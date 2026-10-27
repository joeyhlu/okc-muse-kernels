"""vsl_patches.py: VSL = shorter merge chains for the SLIDING-window attention kernels KS1 / KS3 (notes/study/vsl_attn.md).

KS1 / KS3 end with: partial store -> acq_rel ticket -> the last CTA reads all NS partials from L2 and merges them in split order
(~1.7-2.2 us of the 4.2 / 4.9 us post-wait chain, lab.md KS1 row). Two merge forms replace that tail (new template argument MG,
MG = 0 compiles to the shipped kernels unchanged):
  MG 1 "red":     each CTA adds o * 2^M and l * 2^M (absolute exp2 frame; QK-norm bounds the logits, |s| <= ~68 in log2 units)
                  with red.global.add into a zeroed fp32 workspace, ticket, the last CTA reads ONE summed row set (2-4 KB instead of
                  NS partials), finalizes and zeroes it (the public 2.6x attention's merge; also VA1's).
  MG 2 "cluster": the NS splits of a group are one thread-block cluster (NS = 4 / 8 / 16); after the CTA merge every thread pushes its
                  partial row slice into the owner CTA's shared memory (st.shared::cluster), one cluster barrier, each owner CTA
                  finalizes its rows from local smem in split order. No global partials, no ticket, no workspace. KS1 needs the folded
                  split layout for this (new keys on split NS - 1). KS3 MG 2 keeps KS3's merge arithmetic and order (bitwise = KS3 MG 0).
Routing: VSL_ROUTE {(batch, q_len): variant} (read at call time, so one file serves every A/B arm); variants ("ks1", L, NS, NTG, MG)
and ("ks3", NS, MG); only variants that pass prepare()'s self-test (torch reference, cache bytes == r1, tickets / workspace zero
after every call, graph replay) are routable; anything else keeps the shipped KS1 / KS3 / FA4 route.

    python -I tools/vsl_patches.py [--n=1] [--base=k3_vml_3] [--route=...]   # -> kernels/muse_decode_layer/k3_vsl_<n>.py + lab copy
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)

# ---------------------------------------------------------------------------------------------------------------------------- CUDA
HELPERS = r'''template <int ROWS, int NSRC> struct CLB {             // VSL cluster merge buffer in each owner CTA (MG 2)
  static constexpr int RPO = (ROWS + NSRC - 1) / NSRC;     // rows one owner finalizes (owner of row r = r % NSRC)
  static constexpr int OFF_ML = RPO * NSRC * 64 * 4;       // [RPO][NSRC][64] fp32 partial o, then [RPO][NSRC] (max, sum)
  static constexpr int BYTES = OFF_ML + RPO * NSRC * 8;
};

'''

HELPERS2 = r'''__device__ __forceinline__ void vsl_red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void vsl_red2(float* a, float x, float y) {
  asm volatile("red.global.add.v2.f32 [%0], {%1, %2};\n" ::"l"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void vsl_red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ uint32_t vsl_sa(const void* x) { return (uint32_t)__cvta_generic_to_shared(x); }
__device__ __forceinline__ uint32_t vsl_mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ void vsl_st4(uint32_t a, float x, float y, float z, float w) {
  asm volatile("st.shared::cluster.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void vsl_st2(uint32_t a, float x, float y) {
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};\n" ::"r"(a), "f"(x), "f"(y) : "memory");
}
__device__ __forceinline__ void vsl_carrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void vsl_carrive() { asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void vsl_cwait() { asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory"); }

// KS1's last-CTA finalize arithmetic (normalize, bf16, RMSNorm over the row's UPR lanes, * sigmoid(gate)); op == nullptr: no store
template <int DPT, int UPR>
__device__ __forceinline__ void vsl_fin1(const float (&a)[DPT], float den, const uint32_t (&graw)[DPT / 2], float eps, __nv_bfloat16* op) {
  const float dd = (den > 0.f) ? den : 1.f;
  float y[DPT];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < DPT; ++i) { y[i] = bfr(a[i] / dd); ss = fmaf(y[i], y[i], ss); }
#pragma unroll
  for (int d = UPR / 2; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
  const float rs = rsqrtf(ss * (1.f / 64.f) + eps);
  uint32_t ow2[DPT / 2];
#pragma unroll
  for (int i = 0; i < DPT / 2; ++i) {
    const float2 gg = bf2f(graw[i]);
    const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                    y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
    ow2[i] = *reinterpret_cast<const uint32_t*>(&o2);
  }
  if (op == nullptr) return;
  if constexpr (DPT == 4) *reinterpret_cast<uint2*>(op) = make_uint2(ow2[0], ow2[DPT / 2 - 1]);
  else *reinterpret_cast<uint32_t*>(op) = ow2[0];
}

// KS3's last-CTA finalize arithmetic, same expression forms (16 lanes per row)
__device__ __forceinline__ void vsl_fin3(const float4 a, float den, const uint2 graw, float eps, __nv_bfloat16* op) {
  const float dd = (den > 0.f) ? den : 1.f;
  const float y0 = bfr(a.x / dd), y1 = bfr(a.y / dd), y2 = bfr(a.z / dd), y3 = bfr(a.w / dd);
  float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
  ss += __shfl_xor_sync(FULL, ss, 8);
  ss += __shfl_xor_sync(FULL, ss, 4);
  ss += __shfl_xor_sync(FULL, ss, 2);
  ss += __shfl_xor_sync(FULL, ss, 1);
  const float rs = rsqrtf(ss * (1.f / 64.f) + eps);
  const float2 g01 = bf2f(graw.x), g23 = bf2f(graw.y);
  const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
  const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
  if (op != nullptr)
    *reinterpret_cast<uint2*>(op) = make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
}

'''

KS1_GRAW_C = r'''  uint32_t graw_c[DPT / 2];
#pragma unroll
  for (int i = 0; i < DPT / 2; ++i) graw_c[i] = 0u;
  if constexpr ((MG & 3) == 2) {                                // VSL cluster finalize: this CTA owns rows s, s + NS, ..
    const int ck = tid / UPR, cmu = tid % UPR, cmr = s + ck * NS;
    if (ck < CLB<ROWS, NS>::RPO && cmr < ROWS) {
      const __nv_bfloat16* gq = p.qkv + (size_t)(tb + tg * TPG + (cmr >> 3)) * sq + (size_t)(HQ + 2 * HKV) * 64 +
                                (size_t)(h * 8 + (cmr & 7)) * 64 + DPT * cmu;
      if constexpr (DPT == 4) {
        const uint2 v = __ldcg(reinterpret_cast<const uint2*>(gq));
        graw_c[0] = v.x;
        graw_c[DPT / 2 - 1] = v.y;
      } else {
        graw_c[0] = __ldcg(reinterpret_cast<const unsigned int*>(gq));
      }
    }
  }
'''

KS1_MG = r'''    if constexpr ((MG & 3) == 1) {   // VSL red merge: o 2^M, l 2^M into the zeroed workspace, ticket, the last CTA finalizes one row set
      const float f = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
      float* wa = p.part_o + ((size_t)grp * ROWS + mr) * 64 + DPT * mu;
      float* wl = reinterpret_cast<float*>(p.part_ml) + (size_t)grp * ROWS + mr;
      if constexpr (DPT == 4) vsl_red4(wa, acc[0] * f, acc[1] * f, acc[DPT - 2] * f, acc[DPT - 1] * f);
      else vsl_red2(wa, acc[0] * f, acc[1] * f);
      if (mu == 0) vsl_red1(wl, lt * f);
      KS1P(7);
      __syncthreads();
      if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + grp, 1) == NSP - 1) ? 1 : 0;
      __syncthreads();
      KS1P(8);
      if (misc[0] == 0) return;
      float a[DPT];
      float den = (mu == 0) ? __ldcg(wl) : 0.f;
      if constexpr (DPT == 4) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(wa));
        a[0] = v.x; a[1] = v.y; a[DPT - 2] = v.z; a[DPT - 1] = v.w;
        __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
      } else {
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wa));
        a[0] = v.x; a[1] = v.y;
        __stcg(reinterpret_cast<float2*>(wa), make_float2(0.f, 0.f));
      }
      if (mu == 0) __stcg(wl, 0.f);
      den = __shfl_sync(FULL, den, lane & ~(UPR - 1));
      vsl_fin1<DPT, UPR>(a, den, graw, p.eps, p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64) + (size_t)(h * 8 + (mr & 7)) * 64 + DPT * mu);
      if (tid == 0) p.counters[grp] = 0;
      KS1P(9);
      return;
    }
    if constexpr ((MG & 3) == 2) {   // VSL cluster merge: push the row slice to its owner CTA, one cluster barrier, owners finalize
      using CB = CLB<ROWS, NS>;
      uint8_t* cb = smem + G::OFF_CB;
      vsl_cwait();                                                // entry phase: every CTA of the cluster is running
      {
        const int own = mr % NS, lr = mr / NS;
        const uint32_t ra = vsl_mapa(vsl_sa(cb + 4 * ((lr * NS + s) * 64 + DPT * mu)), (uint32_t)own);
        if constexpr (DPT == 4) vsl_st4(ra, acc[0], acc[1], acc[DPT - 2], acc[DPT - 1]);
        else vsl_st2(ra, acc[0], acc[1]);
        if (mu == 0) vsl_st2(vsl_mapa(vsl_sa(cb + CB::OFF_ML + 8 * (lr * NS + s)), (uint32_t)own), M, lt);
      }
      vsl_carrive();
      vsl_cwait();
      if (s < ROWS && warp < (CB::RPO * UPR + 31) / 32) {         // (ROWS 8 < NS 16: CTAs 8.. own no row)
        const int ck = tid / UPR, cmu = tid % UPR, cmr = s + ck * NS;
        const bool cact = ck < CB::RPO && cmr < ROWS;
        const int ckk = cact ? ck : 0;
        const float* co = reinterpret_cast<const float*>(cb) + (size_t)(ckk * NS) * 64 + DPT * cmu;
        const float2* cm = reinterpret_cast<const float2*>(cb + CB::OFF_ML) + ckk * NS;
        float2 ml[NS];
        float ov[NS][DPT];
#pragma unroll
        for (int q = 0; q < NS; ++q) {
          ml[q] = cm[q];
          if constexpr (DPT == 4) {
            const float4 v = *reinterpret_cast<const float4*>(co + q * 64);
            ov[q][0] = v.x; ov[q][1] = v.y; ov[q][DPT - 2] = v.z; ov[q][DPT - 1] = v.w;
          } else {
            const float2 v = *reinterpret_cast<const float2*>(co + q * 64);
            ov[q][0] = v.x; ov[q][1] = v.y;
          }
        }
        float Mx = neg_inf();
#pragma unroll
        for (int q = 0; q < NS; ++q) Mx = fmaxf(Mx, ml[q].x);
        const float Mu = (Mx == neg_inf()) ? 0.f : Mx;
        float den = 0.f;
        float a[DPT];
#pragma unroll
        for (int i = 0; i < DPT; ++i) a[i] = 0.f;
#pragma unroll
        for (int q = 0; q < NS; ++q) {
          const float w = ex2(ml[q].x - Mu);
          const float wl = w * ml[q].y;
          den += wl;
#pragma unroll
          for (int i = 0; i < DPT; ++i) a[i] = fmaf(w, ov[q][i], a[i]);
        }
        vsl_fin1<DPT, UPR>(a, den, graw_c, p.eps,
                           cact ? p.out + (size_t)(tb + tg * TPG + (cmr >> 3)) * (HQ * 64) + (size_t)(h * 8 + (cmr & 7)) * 64 + DPT * cmu
                                : nullptr);
      }
      return;
    }
'''

KS1_LAUNCH_OLD = r'''template <int L, int NS, int NTG>
static void launch(const Params& P, bool pdl, cudaStream_t st) {
  using G = Geo<L, NS, NTG>;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(NS + (P.fold ? 0 : 1)), (unsigned)HKV, (unsigned)(NTG * P.B));
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks1_kernel<L, NS, NTG>, P));
}

template <int L, int NS, int NTG>
static std::vector<int64_t> info_t() {
  using G = Geo<L, NS, NTG>;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks1_kernel<L, NS, NTG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks1_kernel<L, NS, NTG>, NTHR, G::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G::SMEM, (int64_t)occ};
}
'''

KS1_LAUNCH_NEW = r'''template <int L, int NS, int NTG, int MG = 0>
static void launch(const Params& P, bool pdl, cudaStream_t st) {
  using G = Geo<L, NS, NTG>;
  constexpr int SM = ((MG & 3) == 2) ? G::SMEM_CL : G::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    if ((MG & 3) == 2 && NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(NS + (P.fold ? 0 : 1)), (unsigned)HKV, (unsigned)(NTG * P.B));
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  unsigned na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if ((MG & 3) == 2) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)NS;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks1_kernel<L, NS, NTG, MG>, P));
}

template <int L, int NS, int NTG, int MG = 0>
static std::vector<int64_t> info_t() {
  using G = Geo<L, NS, NTG>;
  constexpr int SM = ((MG & 3) == 2) ? G::SMEM_CL : G::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks1_kernel<L, NS, NTG, MG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks1_kernel<L, NS, NTG, MG>, NTHR, SM));
  if (MG == 0) return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
  int ncl = 0;
  if ((MG & 3) == 2) {
    if (NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks1_kernel<L, NS, NTG, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)NS, (unsigned)HKV, (unsigned)(NTG * 64));
    cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
    cfg.dynamicSmemBytes = SM;
    cudaLaunchAttribute a1;
    a1.id = cudaLaunchAttributeClusterDimension;
    a1.val.clusterDim.x = (unsigned)NS;
    a1.val.clusterDim.y = 1;
    a1.val.clusterDim.z = 1;
    cfg.attrs = &a1;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl, ks1_kernel<L, NS, NTG, MG>, &cfg));
  }
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ, (int64_t)ncl};
}
'''

KS3_GRAW_C = r'''  uint2 graw_c = make_uint2(0u, 0u);
  if constexpr (MG == 2) {
    const int ck = tid >> 4, cmr = s + ck * NS;
    if (ck < CLB<16, NS>::RPO && cmr < 16)
      graw_c = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)cmr * 64 + 4 * (tid & 15)));
  }
'''

KS3_MG = r'''    if constexpr (MG == 1) {   // VSL red merge (see KS1)
      const float f = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
      float* wa = p.part_o + ((size_t)b * 16 + mr) * 64 + 4 * mu;
      float* wl = reinterpret_cast<float*>(p.part_ml) + (size_t)b * 16 + mr;
      vsl_red4(wa, acc.x * f, acc.y * f, acc.z * f, acc.w * f);
      if (mu == 0) vsl_red1(wl, lt * f);
      KS3P(5);
      __syncthreads();
      if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
      __syncthreads();
      KS3P(6);
      if (misc[0] == 0) return;
      float den = (mu == 0) ? __ldcg(wl) : 0.f;
      const float4 v = __ldcg(reinterpret_cast<const float4*>(wa));
      __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
      if (mu == 0) __stcg(wl, 0.f);
      den = __shfl_sync(FULL, den, lane & ~15);
      vsl_fin3(v, den, graw, p.eps, p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 4 * mu);
      if (tid == 0) p.counters[b] = 0;
      KS3P(7);
      return;
    }
    if constexpr (MG == 2) {   // VSL cluster merge (see KS1); KS3's merge order and arithmetic
      using CB = CLB<16, NS>;
      uint8_t* cb = smem + G::OFF_CB;
      vsl_cwait();
      {
        const int own = mr % NS, lr = mr / NS;
        vsl_st4(vsl_mapa(vsl_sa(cb + 4 * ((lr * NS + s) * 64 + 4 * mu)), (uint32_t)own), acc.x, acc.y, acc.z, acc.w);
        if (mu == 0) vsl_st2(vsl_mapa(vsl_sa(cb + CB::OFF_ML + 8 * (lr * NS + s)), (uint32_t)own), M, lt);
      }
      vsl_carrive();
      vsl_cwait();
      if (warp < (CB::RPO * 16 + 31) / 32) {
        const int ck = tid >> 4, cmu = tid & 15, cmr = s + ck * NS;
        const bool cact = ck < CB::RPO && cmr < 16;
        const int ckk = cact ? ck : 0;
        const float* co = reinterpret_cast<const float*>(cb) + (size_t)(ckk * NS) * 64 + 4 * cmu;
        const float2* cm = reinterpret_cast<const float2*>(cb + CB::OFF_ML) + ckk * NS;
        float2 ml[NS];
        float4 ov[NS];
#pragma unroll
        for (int q = 0; q < NS; ++q) {
          ml[q] = cm[q];
          ov[q] = *reinterpret_cast<const float4*>(co + q * 64);
        }
        float Mx = neg_inf();
#pragma unroll
        for (int q = 0; q < NS; ++q) Mx = fmaxf(Mx, ml[q].x);
        const float Mu = (Mx == neg_inf()) ? 0.f : Mx;
        float den = 0.f;
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int q = 0; q < NS; ++q) {
          const float w = ex2(ml[q].x - Mu);
          den += w * ml[q].y;
          a.x = fmaf(w, ov[q].x, a.x); a.y = fmaf(w, ov[q].y, a.y); a.z = fmaf(w, ov[q].z, a.z); a.w = fmaf(w, ov[q].w, a.w);
        }
        vsl_fin3(a, den, graw_c, p.eps, cact ? p.out + (size_t)b * (HQ * 64) + (size_t)cmr * 64 + 4 * cmu : nullptr);
      }
      return;
    }
'''

KS3_LAUNCH_OLD = r'''template <int NS>
static void launch3(const Params3& P, bool pdl, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G3<NS>::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G3<NS>::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks3_kernel<NS>, P));
}

template <int NS>
static std::vector<int64_t> info3() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks3_kernel<NS>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G3<NS>::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks3_kernel<NS>, NTHR, G3<NS>::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G3<NS>::SMEM, (int64_t)occ};
}
'''

KS3_LAUNCH_NEW = r'''template <int NS, int MG = 0>
static void launch3(const Params3& P, bool pdl, cudaStream_t st) {
  constexpr int SM = (MG == 2) ? G3<NS>::SMEM_CL : G3<NS>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    if (MG == 2 && NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  unsigned na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (MG == 2) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)NS;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks3_kernel<NS, MG>, P));
}

template <int NS, int MG = 0>
static std::vector<int64_t> info3() {
  constexpr int SM = (MG == 2) ? G3<NS>::SMEM_CL : G3<NS>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks3_kernel<NS, MG>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks3_kernel<NS, MG>, NTHR, SM));
  if (MG == 0) return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
  int ncl = 0;
  if (MG == 2) {
    if (NS > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS, MG>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)NS, 64u, 1u);
    cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
    cfg.dynamicSmemBytes = SM;
    cudaLaunchAttribute a1;
    a1.id = cudaLaunchAttributeClusterDimension;
    a1.val.clusterDim.x = (unsigned)NS;
    a1.val.clusterDim.y = 1;
    a1.val.clusterDim.z = 1;
    cfg.attrs = &a1;
    cfg.numAttrs = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&ncl, ks3_kernel<NS, MG>, &cfg));
  }
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ, (int64_t)ncl};
}
'''

KS1_SPEC = r'''  uint4 kr[G::KPT][G::CPT], vr[G::KPT][G::CPT];
  if constexpr ((MG & 4) != 0) {   // VSL SPEC (one request): page indices for seq = page-table width - L, issued with cache_seqlens
    int prow[G::KPT];
    const int sg = (p.B == 1) ? max(p.pt_cols - L, 0) : seq;
    {
      const int nog = min(sg, WIN - 1), ksg = sg - nog;
#pragma unroll
      for (int kk = 0; kk < G::KPT; ++kk) {
        const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
        const int gs = s * KPS + slot;
        prow[kk] = (oldc && gs < nog && ksg + gs < p.pt_cols) ? __ldg(p.page_table + (size_t)b * p.pt_cols + ksg + gs) : -1;
      }
    }
    if (sg != seq) {                                            // miss: the real length's indices (no K/V is read from a guess)
#pragma unroll
      for (int kk = 0; kk < G::KPT; ++kk) {
        const int slot = tid / G::TPK + kk * (NTHR / G::TPK);
        const int gs = s * KPS + slot;
        prow[kk] = (oldc && gs < n_old && kstart + gs < p.pt_cols) ? __ldg(p.page_table + (size_t)b * p.pt_cols + kstart + gs) : -1;
      }
    }
#pragma unroll
    for (int kk = 0; kk < G::KPT; ++kk) {
      const bool ok = prow[kk] >= 0;
      const uint8_t* kp = p.k_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
      const uint8_t* vp = p.v_cache + (size_t)(ok ? prow[kk] : 0) * (HKV * 64) + h * 64 + 16 * part * G::CPT;
#pragma unroll
      for (int c = 0; c < G::CPT; ++c) {
        kr[kk][c] = ok ? ldg_nc16(kp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
        vr[kk][c] = ok ? ldg_nc16(vp + 16 * c) : make_uint4(0u, 0u, 0u, 0u);
      }
    }
  } else
  {
    int prow[G::KPT];
'''

# (old, new) pairs applied to KS1_CUDA_SRC; each old must occur exactly once in the base text
CUDA_EDITS = [
    ("helpers", "template <int L, int NS, int NTG> struct Geo {", HELPERS + "template <int L, int NS, int NTG> struct Geo {"),
    ("geo smem", "  static constexpr int SMEM = OFF_MISC + 128;\n",
     "  static constexpr int SMEM = OFF_MISC + 128;\n  static constexpr int OFF_CB = SMEM;\n"
     "  static constexpr int SMEM_CL = OFF_CB + CLB<ROWS, (NS < 16 ? NS : 16)>::BYTES;\n"),
    ("helpers2", "#define KS1P(k) do {", HELPERS2 + "#define KS1P(k) do {"),
    ("ks1 template", "template <int L, int NS, int NTG>\n__global__ void __launch_bounds__(NTHR, 2) ks1_kernel(const __grid_constant__ Params p) {",
     "template <int L, int NS, int NTG, int MG = 0>\n__global__ void __launch_bounds__(NTHR, 2) ks1_kernel(const __grid_constant__ Params p) {"),
    ("ks1 entry arrive", "  if (p.trig == 2) pdl_trigger();\n  KS1P(0);\n",
     "  if constexpr ((MG & 3) == 2) vsl_carrive_relaxed();\n  if (p.trig == 2) pdl_trigger();\n  KS1P(0);\n"),
    ("ks1 owner gate", "      graw[0] = __ldcg(reinterpret_cast<const unsigned int*>(gp));\n    }\n  }\n",
     "      graw[0] = __ldcg(reinterpret_cast<const unsigned int*>(gp));\n    }\n  }\n" + KS1_GRAW_C),
    ("ks1 merge", "    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;\n    if (p.phalf) {",
     KS1_MG + "    const size_t prow = ((size_t)grp * (NS + 1) + s) * ROWS + mr;\n    if (p.phalf) {"),
    ("ks1 launch", KS1_LAUNCH_OLD, KS1_LAUNCH_NEW),
    ("ks1 spec", "  uint4 kr[G::KPT][G::CPT], vr[G::KPT][G::CPT];\n  {\n    int prow[G::KPT];\n", KS1_SPEC),
    ("g3 smem", "  static constexpr int SMEM = OFF_MISC + 64;\n",
     "  static constexpr int SMEM = OFF_MISC + 64;\n  static constexpr int OFF_CB = SMEM;\n"
     "  static constexpr int SMEM_CL = OFF_CB + CLB<16, NS>::BYTES;\n"),
    ("ks3 template", "template <int NS>\n__global__ void __launch_bounds__(NTHR, 2) ks3_kernel(const __grid_constant__ Params3 p) {",
     "template <int NS, int MG = 0>\n__global__ void __launch_bounds__(NTHR, 2) ks3_kernel(const __grid_constant__ Params3 p) {"),
    ("ks3 entry arrive", "  if (p.trig == 2) pdl_trigger();\n  KS3P(0);\n",
     "  if constexpr (MG == 2) vsl_carrive_relaxed();\n  if (p.trig == 2) pdl_trigger();\n  KS3P(0);\n"),
    ("ks3 owner gate",
     "  const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));\n",
     "  const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));\n"
     + KS3_GRAW_C),
    ("ks3 merge", "    const size_t prow = ((size_t)b * NS + s) * 16 + mr;\n    __stcg(",
     KS3_MG + "    const size_t prow = ((size_t)b * NS + s) * 16 + mr;\n    __stcg("),
    ("ks3 launch", KS3_LAUNCH_OLD, KS3_LAUNCH_NEW),
    ("insts macros", "#define KS1_INSTS @INSTS@\n",
     "#define KS1_INSTS @INSTS@\n#define KS1_MINSTS @MINSTS@\n#define KS3_MINSTS @M3INSTS@\n"),
    ("ks1 fwd mg", "  P.fold = (B > 1 && fold) ? 1 : 0;\n  P.trig = (int)trig;\n  P.sync = (int)sync;\n",
     "  const int vmg = (sync >= 3) ? (int)(sync - 2) : 0;   // VSL: sync 3 = red merge (MG 1), 4 = cluster merge (MG 2, folded)\n"
     "  P.fold = ((vmg & 3) == 2) ? 1 : ((B > 1 && fold) ? 1 : 0);\n  P.trig = (int)trig;\n  P.sync = vmg ? 1 : (int)sync;\n"),
    ("ks1 fwd dispatch",
     "#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_>(P, pdl != 0, st); return; }\n",
     "  if (vmg) {\n"
     "#define KS1_XM(L_, NS_, NTG_, MG_) if (vmg == MG_ && L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_, MG_>(P, pdl != 0, st); return; }\n"
     "    KS1_MINSTS\n#undef KS1_XM\n    TORCH_CHECK(false, \"ks1: VSL variant not built\");\n  }\n"
     "#define KS1_X(L_, NS_, NTG_) if (L == L_ && NS == NS_ && NTG == NTG_) { ks1::launch<L_, NS_, NTG_>(P, pdl != 0, st); return; }\n"),
    ("ks1 info dispatch", "std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG) {\n",
     "std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG) {\n"
     "  if (NTG >= 100) {\n    const int64_t vmg = NTG / 100;\n    NTG %= 100;\n"
     "#define KS1_XM(L_, NS_, NTG_, MG_) if (vmg == MG_ && L == L_ && NS == NS_ && NTG == NTG_) return ks1::info_t<L_, NS_, NTG_, MG_>();\n"
     "    KS1_MINSTS\n#undef KS1_XM\n    return {};\n  }\n"),
    ("ks3 fwd decode", "                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {\n  const int64_t B = cache_seqlens.numel();\n",
     "                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {\n"
     "  const int vmg3 = (int)(NS / 100);   // VSL: NS + 100 MG\n  NS %= 100;\n  const int64_t B = cache_seqlens.numel();\n"),
    ("ks3 fwd dispatch", "  if (NS == 4) { ks3::launch3<4>(P, pdl != 0, st); return; }\n",
     "  if (vmg3) {\n#define KS3_XM(NS_, MG_) if (vmg3 == MG_ && NS == NS_) { ks3::launch3<NS_, MG_>(P, pdl != 0, st); return; }\n"
     "    KS3_MINSTS\n#undef KS3_XM\n    TORCH_CHECK(false, \"ks3: VSL variant not built\");\n  }\n"
     "  if (NS == 4) { ks3::launch3<4>(P, pdl != 0, st); return; }\n"),
    ("ks3 info dispatch", "std::vector<int64_t> ks3_info(int64_t NS) {\n",
     "std::vector<int64_t> ks3_info(int64_t NS) {\n  if (NS >= 100) {\n    const int64_t vmg = NS / 100;\n    NS %= 100;\n"
     "#define KS3_XM(NS_, MG_) if (vmg == MG_ && NS == NS_) return ks3::info3<NS_, MG_>();\n"
     "    KS3_MINSTS\n#undef KS3_XM\n    return {};\n  }\n"),
]

# ------------------------------------------------------------------------------------------------------------------------- Python
PY_CONST_ANCHOR = "KS1_PREPARE_BY_S = 230.0\n"
PY_CONST = r'''VSL_ENABLE = True
VSL_ROUTE = @ROUTE@
VSL_KS1_INSTS = @KS1I@
VSL_KS3_INSTS = @KS3I@
VSL_WS_FLOATS = 1 << 20
VSL_PREPARE_BY_S = 240.0
_VSL = {"ws": {}, "seen": set()}
'''

PY_BUILD_OLD = '    cuda = KS1_CUDA_SRC.replace("@INSTS@", insts) + _dg1_fold("cuda")\n'
PY_BUILD_NEW = ('    minsts = " ".join(f"KS1_XM({l}, {n}, {g}, {m})" for l, n, g, m in VSL_KS1_INSTS) if VSL_ENABLE else ""\n'
                '    m3insts = " ".join(f"KS3_XM({n}, {m})" for n, m in VSL_KS3_INSTS) if VSL_ENABLE else ""\n'
                '    cuda = (KS1_CUDA_SRC.replace("@INSTS@", insts).replace("@MINSTS@", minsts).replace("@M3INSTS@", m3insts)\n'
                '            + _dg1_fold("cuda"))\n')

PY_WANTED_OLD = '    if not (geo is not None and (L, *geo) in passed and cfg["window"] == 2048 and cfg["rope_theta"] == ROPE_THETA\n'
PY_WANTED_NEW = ('    if not (((geo is not None and (L, *geo) in passed) or _vsl_pick(cfg, state) is not None) and cfg["window"] == 2048\n'
                 '            and cfg["rope_theta"] == ROPE_THETA\n')

PY_RUN_OLD = '''def _ks1_run(qkv, data, state):
    cfg = data["config"]
'''
PY_RUN_NEW = '''def _ks1_run(qkv, data, state):
    cfg = data["config"]
    v = _vsl_pick(cfg, state)
    if v is not None:
        return _vsl_call(qkv, data, state["cos_sin"], v)
'''

PY_PREP_OLD = '''    state["ks1"] = set(ok_geo)
    if ok_geo and KS1_P1_CARVEOUT >= 0'''
PY_PREP_NEW = '''    state["ks1"] = set(ok_geo)
    _vsl_prepare(state, device, t0)
    if ok_geo and KS1_P1_CARVEOUT >= 0'''

PY_FUNCS_ANCHOR = "\n\ndef _ks1_case(ctx, length, seed, row0, device):\n"
PY_FUNCS = r'''

def _vsl_ws(dev):
    key = str(dev)
    if key not in _VSL["ws"]:
        _VSL["ws"][key] = (torch.zeros(VSL_WS_FLOATS, dtype=torch.float32, device=dev),
                           torch.zeros(VSL_WS_FLOATS // 8, dtype=torch.float32, device=dev))
    return _VSL["ws"][key]


def _vsl_pick(cfg, state):
    if not VSL_ENABLE:
        return None
    v = VSL_ROUTE.get((cfg["batch"], cfg["q_len"]))
    if v is None:
        return None
    v = tuple(v)
    return v if v in (state.get("vsl") or ()) else None


def _vsl_call(qkv, data, cos_sin, v):
    cfg = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    dev = qkv.device
    po, pml = _vsl_ws(dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    qk = float(cfg["sm_scale"] * cfg["qk_scale_factor"])
    if v[0] == "ks1":
        _, l, ns, ntg, mg = v
        out = torch.empty((B * L, 16 * 64), dtype=torch.bfloat16, device=dev)
        _KS1["mod"].ks1_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                                data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                                _ks1_counters(dev), out, prof, L, ns, ntg, qk, KS1_EPS, int(KS1_TRIG.get(L, 0)),
                                1 if KS1_PDL else 0, 2 + mg, 0, 1 if (mg & 3) == 2 else 0)
    else:
        _, ns, mg = v
        out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
        _KS1["mod"].ks3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                                data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                                _ks1_counters(dev), out, prof, int(ns + 100 * mg), qk, KS1_EPS,
                                int(KS3_TRIG.get((B, 1), 1)), 1 if KS1_PDL else 0, 1 if cfg["rope_theta"] else 0)
    if v not in _VSL["seen"]:
        _VSL["seen"].add(v)
        print(f"[vsl] first call {v} B={B} L={L}", flush=True)
    return out


def _vsl_variants():
    return ([("ks1", l, n, g, m) for l, n, g, m in VSL_KS1_INSTS] + [("ks3", n, m) for n, m in VSL_KS3_INSTS])


def _vsl_cases(v):
    if v[0] == "ks1":
        return [(1, ctx, length, seed + 50, row0) for ctx, length, seed, row0 in KS1_SELF_TEST_CASES if length == v[1]]
    return ([(b, ctx, 1, seed + 50, row0) for b, ctx, seed, row0 in KS3_SELF_TEST_CASES]
            + [(1, 3000, 1, 991, False), (1, 2047, 1, 992, False), (1, 100, 1, 993, False), (1, 1, 1, 994, False),
               (1, 4000, 1, 995, True), (16, 2049, 1, 996, False)])


def _vsl_self_test(cos_sin, device):
    cnt = _ks1_counters(device)
    po, pml = _vsl_ws(device)
    ok = set()
    for v in _vsl_variants():
        good, worst, nbit, nb = True, 0.0, 0, 0
        cnt.zero_()
        po.zero_()
        pml.zero_()
        try:
            for B, ctx, L, seed, row0 in _vsl_cases(v):
                T = B * L
                ref = _k1fa4_reference(_k1fa4_case(B, ctx, L, 2048, seed, row0, device)).view(T, -1)
                d1 = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                o1 = _k1fa4_r1_path(d1).float().view(T, -1)
                torch.cuda.synchronize(device)
                e1 = float((o1 - ref).norm() / ref.norm())
                rows = d1["cache_loc"]
                k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                dk = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                qkv = dk["qkv"].view(T, -1)
                o_a = _vsl_call(qkv, dk, cos_sin, v).clone()
                o_b = _vsl_call(qkv, dk, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                ek = float((o_a.float() - ref).norm() / ref.norm())
                kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
                clean = int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0 and float(pml.abs().sum()) == 0.0
                if (v[-1] & 3) == 2:
                    rep = bool(torch.equal(o_a, o_b))
                else:
                    rep = float((o_a.float() - o_b.float()).abs().max()) <= 0.02 * float(o_a.float().abs().max()) + 1e-3
                if v[0] == "ks3" and v[-1] == 2:   # KS3 MG 2 keeps KS3's merge: bitwise equal to the shipped kernel
                    ds = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                    o_s = _ks3_call(ds["qkv"].view(T, -1), ds, cos_sin, v[1], None, None)
                    torch.cuda.synchronize(device)
                    nb += 1
                    nbit += int(torch.equal(o_s, o_a))
                    del ds, o_s
                if v[0] == "ks1" and (v[-1] & 4):   # SPEC miss: a page table wider than seq + L (guess wrong -> real length)
                    dm = _k1fa4_case(B, ctx, L, 2048, seed, row0, device)
                    pt = dm["page_table"]
                    dm["page_table"] = torch.cat([pt, torch.full((B, 37), -1, dtype=pt.dtype, device=device)], 1).contiguous()
                    o_m = _vsl_call(dm["qkv"].view(T, -1), dm, cos_sin, v)
                    torch.cuda.synchronize(device)
                    em = float((o_m.float() - ref).norm() / ref.norm())
                    km = dm["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                    vm = dm["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                    nbad += int((km != k1b).sum()) + int((vm != v1b).sum())
                    ek = max(ek, em)
                    clean = clean and int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0
                    del dm, o_m
                c_ok = (bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and clean and rep)
                worst = max(worst, ek)
                if not c_ok:
                    good = False
                    print(f"[vsl] self-test {v} B{B} ctx{ctx} L{L}{' row0' if row0 else ''}: FAIL err {ek:.4e} (r1 {e1:.4e}) "
                          f"cache bytes != r1 {nbad} clean {clean} repeat {rep}", flush=True)
                    cnt.zero_()
                    po.zero_()
                    pml.zero_()
                del dk, o_a, o_b, d1, o1, ref
                if not good:
                    break
            if good:   # graph replay: captured call twice, state clean, output as eager
                B, ctx, L = (1, 3000, v[1]) if v[0] == "ks1" else (32, 3000, 1)
                d = _k1fa4_case(B, ctx, L, 2048, 977, False, device)
                qkv = d["qkv"].view(B * L, -1)
                o1 = _vsl_call(qkv, d, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    og = _vsl_call(qkv, d, cos_sin, v)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize(device)
                same = (bool(torch.equal(og, o1)) if (v[-1] & 3) == 2 else
                        float((og.float() - o1.float()).abs().max()) <= 0.02 * float(o1.float().abs().max()) + 1e-3)
                clean = int(cnt.abs().sum()) == 0 and float(po.abs().sum()) == 0.0
                if not (same and clean):
                    good = False
                    print(f"[vsl] graph check {v}: FAIL same {same} clean {clean}", flush=True)
                del gr, og, d
        except Exception as ex:
            good = False
            print(f"[vsl] self-test {v} raised {repr(ex)[-600:]}", flush=True)
            torch.cuda.synchronize(device)
            cnt.zero_()
            po.zero_()
            pml.zero_()
        if good:
            ok.add(v)
        bits = f", bitwise = KS3 {nbit}/{nb}" if nb else ""
        print(f"[vsl] variant {v}: {'ok' if good else 'FAILED'} (worst err {worst:.3e}{bits})", flush=True)
    return ok


def _vsl_prepare(state, device, t0):
    state["vsl"] = set()
    if not (VSL_ENABLE and _KS1["mod"] is not None):
        return
    if time.time() - t0 > VSL_PREPARE_BY_S:
        print(f"[vsl] skipped: prepare already took {time.time() - t0:.1f}s: KS1 / KS3 stay", flush=True)
        return
    t1 = time.time()
    try:
        mod = _KS1["mod"]
        for l, n, g, m in VSL_KS1_INSTS:
            print(f"[vsl] ks1 L{l} NS{n} NTG{g} MG{m}: regs / local / smem / CTAs per SM / clusters {list(mod.ks1_info(l, n, g + 100 * m))}",
                  flush=True)
        for n, m in VSL_KS3_INSTS:
            print(f"[vsl] ks3 NS{n} MG{m}: regs / local / smem / CTAs per SM / clusters {list(mod.ks3_info(n + 100 * m))}", flush=True)
        state["vsl"] = _vsl_self_test(state["cos_sin"], device)
    except Exception as ex:
        print(f"[vsl] prepare FAILED {repr(ex)[-1500:]}: KS1 / KS3 stay", flush=True)
        state["vsl"] = set()
    live = {k: tuple(v) for k, v in VSL_ROUTE.items() if tuple(v) in state["vsl"]}
    print(f"[vsl] variants ok {sorted(state['vsl'])}; route {live or 'none (KS1 / KS3 / FA4 as shipped)'} "
          f"({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
'''

PY_EDITS = [
    ("vsl consts", PY_CONST_ANCHOR, None),  # filled in apply()
    ("vsl build", PY_BUILD_OLD, PY_BUILD_NEW),
    ("vsl wanted", PY_WANTED_OLD, PY_WANTED_NEW),
    ("vsl run", PY_RUN_OLD, PY_RUN_NEW),
    ("vsl prepare", PY_PREP_OLD, PY_PREP_NEW),
    ("vsl funcs", PY_FUNCS_ANCHOR, PY_FUNCS + PY_FUNCS_ANCHOR),
]


def check(ok, msg):
    if not ok:
        raise SystemExit(f"[vsl_patches] FAIL: {msg}")


KS1I_V1 = ((1, 16, 1, 1), (1, 16, 1, 2), (1, 8, 1, 2), (8, 16, 4, 1), (8, 16, 4, 2), (8, 8, 4, 2))
KS3I_V1 = ((16, 2), (8, 2), (4, 2), (8, 1), (16, 1))


def apply(text, route=None, log=None, ks1_insts=KS1I_V1, ks3_insts=KS3I_V1):
    """text = a stripped layer file containing KS1 / KS3 (k3_vml_3 lineage); route = VSL_ROUTE literal (default {} = ship)."""
    route = route if route is not None else "{}"
    i0 = text.index('KS1_CUDA_SRC = r"""')
    i1 = text.index('"""', i0 + len('KS1_CUDA_SRC = r"""'))
    cu = text[i0:i1]
    lean = not any((g[3] & 3) == 2 for g in ks1_insts) and not any(g[1] == 2 for g in ks3_insts)
    for what, old, new in CUDA_EDITS:
        if what == "ks1 spec" and not any(g[3] & 4 for g in ks1_insts):
            continue
        if lean and what in ("ks1 owner gate", "ks3 owner gate"):   # no cluster instance: no cluster code text
            continue
        if lean and what in ("ks1 merge", "ks3 merge"):
            mark = ("    if constexpr ((MG & 3) == 2) {   // VSL cluster merge" if what == "ks1 merge"
                    else "    if constexpr (MG == 2) {   // VSL cluster merge")
            j0 = new.index(mark)
            j1 = new.index("    const size_t prow")
            new = new[:j0] + new[j1:]
        check(cu.count(old) == 1, f"CUDA anchor {what!r} found {cu.count(old)} times")
        cu = cu.replace(old, new)
    t = text[:i0] + cu + text[i1:]
    for what, old, new in PY_EDITS:
        if new is None:
            new = (old + PY_CONST.replace("@ROUTE@", route).replace("@KS1I@", repr(tuple(ks1_insts)))
                   .replace("@KS3I@", repr(tuple(ks3_insts))))
        check(t.count(old) == 1, f"Python anchor {what!r} found {t.count(old)} times")
        t = t.replace(old, new)
    if log:
        log(f"VSL applied: +{len(t.encode()) - len(text.encode())} B, route {route}")
    return t


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    n = args.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_vml_3')}.py"
    route = args.get("route", "{}")
    out = ROOT / f"kernels/muse_decode_layer/k3_vsl_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vsl_{n}_ab.py"
    tag = f"k3-vsl-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    check(fb == b, f"{base.name} is not a finish() fixed point")
    k1i = tuple(tuple(int(x) for x in g.split(",")) for g in args["ks1"].split(";")) if "ks1" in args else KS1I_V1
    k3i = tuple(tuple(int(x) for x in g.split(",")) for g in args["ks3"].split(";")) if "ks3" in args else KS3I_V1
    form = "red.add" if not any((g[3] & 3) == 2 for g in k1i) and not any(g[1] == 2 for g in k3i) else "red.add or cluster-DSMEM"
    nonce = (f"# nonce {tag} ({base.stem} + VSL: sliding attention KS1 / KS3 {form} merge variants, "
             f"route {route.replace(' ', '')}, self-tested with fallback)")
    t = apply(b.replace(bn, nonce), route, rep.append, k1i, k3i)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vsl_patches] OK  {r}")
    out.write_text(s, encoding="utf-8", newline=NL)
    if "--nolab" not in sys.argv:
        lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vsl_patches] wrote {out.relative_to(ROOT)} and {lab.relative_to(ROOT)} ({len(s.encode())} B, headroom "
          f"{1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate kernels/muse_moe_w4a16/k2a_v39.py from k2a_v38.py (design: kernels/muse_moe_w4a16/k2a_v39_design.md).

v3.9 = v3.8 + vids 6 / 7: v3.3's NC1 kernels (vid 0) launched as 2- / 4-CTA thread-block clusters, with the split-K
reduction of every split item whose CTAs all sit in one cluster done through distributed shared memory:
- contributors st.async their FP32 accumulators into the owner's smem slot (complete_tx on its receive mbarrier);
- the owner (the CTA holding the item's last stage) waits on that mbarrier (bounded, error flag), adds
  gs*own + gs*partner in slot order (__fmul_rn/__fadd_rn: the value today's two FP32 reds give for 2-way splits), and
  flushes the item as a whole one.
No h / y reds, no counter, no __threadfence, no readback for those items. Items that span clusters keep the global
protocol (epilogue warp). vids 0-5 are untouched: `gemm` gains a defaulted template parameter CL = 1 whose code is
`if constexpr (CL > 1)` only.

  python tools/gen_k2a_v39.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "kernels/muse_moe_w4a16/k2a_v38.py"
DST = ROOT / "kernels/muse_moe_w4a16/k2a_v39.py"

HELP = r'''
// ---- v3.9: cluster split-K through distributed shared memory (vids 6 / 7) ----
__device__ __forceinline__ bool mbar_try_cl(uint32_t b, uint32_t par) {   // acquire at cluster scope (remote writers)
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait_cl(uint32_t b, uint32_t par, int* err) {
  if (mbar_try_cl(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try_cl(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 2); return; }
  }
}
__device__ __forceinline__ uint32_t mapa_u32(uint32_t la, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_async_v4(uint32_t ra, float a, float b, float c, float d, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];\n"
               :: "r"(ra), "f"(a), "f"(b), "f"(c), "f"(d), "r"(rbar) : "memory");
}
__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
// The CTA whose stream-K range [floor(S c / G), floor(S (c+1) / G)) holds stage x.
__device__ __forceinline__ int cta_of(long long S, int G, int x) {
  int c = (int)(((long long)x * G) / S);
  c = max(0, min(c, G - 1));
  while (c > 0 && (S * c) / G > x) --c;
  while (c + 1 < G && (S * (c + 1)) / G <= x) ++c;
  return c;
}
// Item [a0, a0 + 16) is split and all its CTAs (clo .. chi) sit in one CL-cluster: reduced through DSMEM into chi.
__device__ __forceinline__ bool dsm_item(long long S, int G, int CL, int a0, int& clo, int& chi) {
  clo = cta_of(S, G, a0);
  chi = cta_of(S, G, a0 + 15);
  return chi > clo && (clo / CL) == (chi / CL);
}
constexpr int DSM_SLOT = 8 * 32 * 16;   // one contributor's NC1 accumulators: 8 warps x 32 lanes x float4
'''

FLUSH_OLD = ("    // ---- flush ----\n#pragma unroll\n    for (int q = 0; q < NC; ++q)\n#pragma unroll\n"
             "      for (int i = 0; i < 4; ++i) if (NC == 1) acc[q][i] += acc2[q][i];\n"
             "    const int fmode = P.probe & 15;   // 6: partial items stored like whole ones (timing only); 7: no finalize\n"
             "    if (fmode != 0 && fmode < 6) {")
FLUSH_NEW = ("    // ---- flush ----\n#pragma unroll\n    for (int q = 0; q < NC; ++q)\n#pragma unroll\n"
             "      for (int i = 0; i < 4; ++i) if (NC == 1) acc[q][i] += acc2[q][i];\n"
             "    bool dsm_pre = false, dsm_skip = false;   // v3.9: owner (pre-scaled sum, flushed whole) / contributor (sent)\n"
             "    if constexpr (CL > 1) {\n"
             "      if (!(s0 == 0 && s1 == 16)) {\n"
             "        int clo, chi;\n"
             "        if (dsm_item(Stot, ncta, CL, S.sb[e] + 16 * item, clo, chi)) {\n"
             "          const uint32_t rbar = sa(bars + 2 * NST);\n"
             "          const uint32_t rbuf = sa(bars + 2 * NST) + 16;\n"
             "          if (cta == chi) {\n"
             "            const float gsv = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);\n"
             "            mbar_wait_cl(rbar, 0, P.err);\n"
             "            const float4* rb = reinterpret_cast<const float4*>(reinterpret_cast<const uint8_t*>(bars + 2 * NST) + 16) + warp * 32 + lane;\n"
             "            float r0 = __fmul_rn(acc[0][0], gsv), r1 = __fmul_rn(acc[0][1], gsv);\n"
             "            float r2 = __fmul_rn(acc[0][2], gsv), r3 = __fmul_rn(acc[0][3], gsv);\n"
             "            for (int k = 0; k < chi - clo; ++k) {\n"
             "              const float4 p = rb[k * 256];\n"
             "              r0 = __fadd_rn(r0, __fmul_rn(p.x, gsv)); r1 = __fadd_rn(r1, __fmul_rn(p.y, gsv));\n"
             "              r2 = __fadd_rn(r2, __fmul_rn(p.z, gsv)); r3 = __fadd_rn(r3, __fmul_rn(p.w, gsv));\n"
             "            }\n"
             "            acc[0][0] = r0; acc[0][1] = r1; acc[0][2] = r2; acc[0][3] = r3;\n"
             "            dsm_pre = true;\n"
             "          } else {\n"
             "            const uint32_t rk = (uint32_t)(chi % CL);\n"
             "            const uint32_t off = (uint32_t)((cta - clo) * DSM_SLOT + (warp * 32 + lane) * 16);\n"
             "            st_async_v4(mapa_u32(rbuf + off, rk), acc[0][0], acc[0][1], acc[0][2], acc[0][3], mapa_u32(rbar, rk));\n"
             "            dsm_skip = true;\n"
             "          }\n"
             "        }\n"
             "      }\n"
             "    }\n"
             "    const int fmode = P.probe & 15;   // 6: partial items stored like whole ones (timing only); 7: no finalize\n"
             "    if (dsm_skip) {\n"
             "    } else if (fmode != 0 && fmode < 6) {")

KERN = r'''
// v3.9 (vids 6 / 7): v3.3's NC1 K1 / K2 (vid 0) in CL-CTA clusters. Thread 0 arms the receive barrier of the item this
// CTA owns (the split item holding its first stage, if all of that item's CTAs are in this cluster: chi - clo slots of
// DSM_SLOT bytes), then a cluster barrier, so no contributor writes before the owner's barrier is armed.
template <int CL, int NST>
__device__ __forceinline__ void cl_setup(const Params& P, const Route& S, uint64_t* bars) {
  if (threadIdx.x == 0) {
    const long long Stot = S.sb[32];
    int beg, end;
    my_range2(Stot, (int)blockIdx.x, (int)gridDim.x, beg, end);
    const uint32_t rb = sa(bars + 2 * NST);
    mbar_init(rb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    uint32_t nb = 0;
    if (end > beg) {
      int e = 0;
      while (S.sb[e + 1] <= beg) ++e;
      const int a0 = S.sb[e] + ((beg - S.sb[e]) & ~15);
      int clo, chi;
      if (a0 < beg && dsm_item(Stot, (int)gridDim.x, CL, a0, clo, chi) && chi == (int)blockIdx.x)
        nb = (uint32_t)((chi - clo) * DSM_SLOT);
    }
    mbar_expect_tx(rb, nb);   // nb = 0: the phase completes at once (never waited on)
  }
  __syncthreads();
  cluster_sync_all();
}

template <int CL>
__global__ void __launch_bounds__(320, 2) k1_kernel39(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int NC = 1, NST = 3, STG = SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<NC, NST>(bars);
  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  cl_setup<CL, NST>(P, S, bars);
  gemm<NC, NST, 1, 0, STG, CL>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);
}

template <int CL>
__global__ void __launch_bounds__(320, 2) k2_kernel39(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int NC = 1, NST = 3, STG = SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<NC, NST>(bars);
  route<NC>(P, S, nullptr, s_list, 32, false);
  cl_setup<CL, NST>(P, S, bars);
  gemm<NC, NST, 2, 0, STG, CL>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);
}

'''

HOST = r'''// v3.9 vids 6 / 7: vid 0's launch chain with a (CL, 1, 1) cluster on K1 and K2 (K2 / K3 PDL as before); dynamic smem =
// vid 0's ring + barriers, then the receive barrier (16 B) and CL - 1 contributor slots.
template <int CL>
static constexpr int smem_bytes39() { return smem_bytes3<1, 3, 2, false>() + 16 + (CL - 1) * k3::DSM_SLOT; }

template <int CL>
static void launch39(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes39<CL>();
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  if (mask & 1) {
    cfg.numAttrs = 1;   // K1: cluster only (normal launch, as vid 0)
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel39<CL>, p));
  }
  if (mask & 2) {
    cfg.numAttrs = 2;
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel39<CL>, p));
  }
  if (mask & 4) {
    cfg.attrs = at + 1; cfg.numAttrs = 1;
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

// attrs: (regs, local, smem, occ) per kernel; occ = min(blocks/SM, max active clusters * CL / SMs).
template <int CL>
static std::vector<int64_t> attrs39() {
  constexpr int SM = smem_bytes39<CL>();
  std::vector<int64_t> r;
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0, ncl = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = CL; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(2 * sms); cfg.blockDim = dim3(nt); cfg.dynamicSmemBytes = SM; cfg.attrs = at; cfg.numAttrs = 1;
    if (cudaOccupancyMaxActiveClusters(&ncl, f, &cfg) != cudaSuccess) ncl = -1;
    const int occc = (ncl > 0 && sms > 0) ? (ncl * CL) / sms : -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM),
                       (int64_t)(occc >= 0 ? (occ < occc ? occ : occc) : -1)});
  };
  one((const void*)k3::k1_kernel39<CL>, 320);
  one((const void*)k3::k2_kernel39<CL>, 320);
  (void)cudaGetLastError();
  return r;
}

'''


def sub1(text, old, new):
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor matched {n} times: {old[:100]!r}")
    return text.replace(old, new)


def main():
    full = SRC.read_text(encoding="utf-8")
    a = full.index('KERNEL_SRC_V3 = r"""')
    b = full.index('CPP_SRC_V3 = r"""')
    head, k, tail = full[:a], full[a:b], full[b:]
    n = 0
    # helpers after mbar_wait
    k = sub1(k, "    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }\n  }\n}\n"
                "__device__ __forceinline__ void bulk_g2s(",
             "    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }\n  }\n}\n"
             + HELP + "__device__ __forceinline__ void bulk_g2s("); n += 1
    k = sub1(k, "template <int NC, int NST, int MODE, int HM, int STG>\n__device__ void gemm(",
             "template <int NC, int NST, int MODE, int HM, int STG, int CL = 1>\n__device__ void gemm("); n += 1
    k = sub1(k, "  my_range2t(Stot, cta, ncta, v38_a(P, MODE), beg, end);",
             "  my_range2t(Stot, cta, ncta, (CL > 1) ? 0 : v38_a(P, MODE), beg, end);"); n += 1
    k = sub1(k, "          x += s1 - s0;\n          if (s0 == 0 && s1 == 16) continue;\n          constexpr bool hot = HOTCAP;",
             "          x += s1 - s0;\n          if (s0 == 0 && s1 == 16) continue;\n"
             "          if constexpr (CL > 1) {   // v3.9: cluster-split items never reach the epilogue warp\n"
             "            int clo, chi;\n"
             "            if (dsm_item(Stot, ncta, CL, S.sb[e] + 16 * item, clo, chi)) continue;\n"
             "          }\n"
             "          constexpr bool hot = HOTCAP;"); n += 1
    k = sub1(k, FLUSH_OLD, FLUSH_NEW); n += 1
    k = sub1(k, "      const float gs = v38_b3(P) ? S.gs[e] : __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);\n"
                "      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;",
             "      const float gs = dsm_pre ? 1.f : (v38_b3(P) ? S.gs[e] : __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e));\n"
             "      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6 || dsm_pre;"); n += 1
    k = sub1(k, "    if (!(s0 == 0 && s1 == 16)) ++npart;\n    x += s1 - s0;",
             "    if (!(s0 == 0 && s1 == 16) && !dsm_pre && !dsm_skip) ++npart;\n    x += s1 - s0;"); n += 1
    k = sub1(k, "// P0 (v3.3): warm experts' x rows packed", KERN + "// P0 (v3.3): warm experts' x rows packed"); n += 1
    k = sub1(k, "void run3(int64_t vid, int64_t mask, torch::Tensor x,", HOST + "void run3(int64_t vid, int64_t mask, torch::Tensor x,"); n += 1

    head = sub1(head, "# nonce k2a-v38-1", "# nonce k2a-v39-1")
    head = sub1(head, '"""K2a v3.8 = v3.7 + boundary switches',
                '"""K2a v3.9 = v3.8 + vids 6 / 7: vid 0 (v3.3 NC1) in 2- / 4-CTA clusters, split items whose CTAs share a cluster\n'
                'reduced through DSMEM into their owner (no reds / counters / fences / readback); design k2a_v39_design.md,\n'
                'generated by tools/gen_k2a_v39.py, bench tools/box/k38_bench.py rows cl2 / cl4. vids 0-5 = v3.8.\n\n'
                'K2a v3.8 = v3.7 + boundary switches')
    head = sub1(head, "V3_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4)]",
                "V3_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4),\n"
                "               (1, 3, 2, 4, 2), (1, 3, 2, 4, 4)]   # v3.9: HK 4 = vid 0 in NCH-CTA clusters (DSMEM split-K)")
    tail = sub1(tail, "        elif hk == 3:",
                "        elif hk == 4:\n"
                "            assert nc == 1 and nst == 3 and minb == 2 and nch in (2, 4), v\n"
                "            cases.append(\"    case %d: launch39<%d>(p, (int)G, (int)G3, (int)mask); break;\" % (i, nch))\n"
                "            attrs.append(\"    case %d: return attrs39<%d>();\" % (i, nch))\n"
                "        elif hk == 3:")
    tail = sub1(tail, '            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;")',
                '            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;\\n"\n'
                '            "    case 6: launch39<2>(p, (int)G, (int)G3, (int)mask); break;\\n"\n'
                '            "    case 7: launch39<4>(p, (int)G, (int)G3, (int)mask); break;")')
    tail = sub1(tail, '            "    case 5: return attrs37<4, 3>();")',
                '            "    case 5: return attrs37<4, 3>();\\n"\n'
                '            "    case 6: return attrs39<2>();\\n"\n'
                '            "    case 7: return attrs39<4>();")')
    tail = sub1(tail, 'def v3_prepare(weights, st2=None, variants=V3_VARIANTS, name="k2a_v38", log=print, share=None):',
                'def v3_prepare(weights, st2=None, variants=V3_VARIANTS, name="k2a_v39", log=print, share=None):')
    out = head + k + tail
    DST.write_text(out, encoding="utf-8")
    print(f"[gen_k2a_v39] {n} kernel patches + python; wrote {DST} ({len(out)} B)")


if __name__ == "__main__":
    main()

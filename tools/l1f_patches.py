"""l1f_patches.py: L1 ROUTE-EARLY (FRONT builder, lead_oct29.md L1) on a layer's GD block (base k3_gc_19).
Stage 1 step 0 = router first + routing CTAs in the pooler launch (tail2 / VMS unchanged; routing -> scratch).

GD template parameter RE (default 0 = the shipped kernels: RE = 0 instances are PTX-identical to the base's, checked by
tools/l1f_local_compile.py):
* VF = 2 host (fused pooler + FP32 router rows, T = 1..8): each pooler CTA computes its router share (router32_kernel<8>'s
  arithmetic, verbatim) right after its x32 slice lands, BEFORE its pooler tiles, then releases one arrival on re_ctr[0].
  T = 1 uses the VF host too (ship: GD pooler + side-stream lean router; same per-row arithmetic -> same partial bits).
* GC host (cat GEMV, T = 9..16): each warp runs its 2 virtual hi/lo router units FIRST (stored order unchanged: the loop
  visits units KUP, KUP+1, 0..KUP-1; the per-warp escape list is consumed from its router part, then from 0), the CTA parks
  + stores its K slice's router partial and releases one arrival.
* NRC routing CTAs at the LOWEST blockIdx (VF 1, GC MT = 2) run tail_kernel2's routing warp (one token per warp, bitwise:
  same loads / sum order / sigmoid + bias / redux top-8 / weights / local ids) after re_ctr[0] >= re_np (bounded
  globaltimer spin, sticky error flag re_ctr[2], no trap), write tw / tid / lid; the last one subtracts re_np.
* RE = 2: same + per-CTA globaltimer marks (lab only): re_mk[rc*4 + {0 entry, 1 shares complete, 2 routing done}],
  re_mk[16 + c*4 + {0 entry, 1 post-wait, 2 share released, 3 end}].
Host: gd_re_arm(...) arms ONE following gdr_run (vf 2) / gdc_run launch (RTF-style), which then launches the RE instance
with nbx + NRC CTAs. gd_re_info(k) = regs / local / occupancy / smem of the 4 RE instances.
"""
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[l1f_patches] FAIL {msg}")


def rep(t, old, new, what, n=1):
    check(t.count(old) == n, f"anchor {what}: {t.count(old)} matches (want {n})")
    return t.replace(old, new)


DEVICE = r'''
// RE (L1 route-early): tail_kernel2's routing warp, one token per warp, run by the NRC routing CTAs (lowest blockIdx) of the
// pooler launch once every pooler CTA released its router share (re_ctr[0] >= re_np). Bitwise tail_kernel2 (same sum order).
__device__ __forceinline__ int re_ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
template <int VF, int NRC, int RE>
__device__ __forceinline__ void re_route(const Args& A, int rc) {
  constexpr int TK = 8, NE = 256;
  const int tx = (int)threadIdx.x, lane = tx & 31, warp = tx >> 5;
  const int t = rc * 8 + warp;
  if (RE == 2 && A.re_mk && tx == 0) A.re_mk[rc * 4 + 0] = vf_gt();
  const float4* pbias = reinterpret_cast<const float4*>(A.re_bias) + lane * 2;
  const float4 c0 = pbias[0], c1 = pbias[1];
  const int ls = A.re_ls[0];
  if (tx == 0) {
    const long long g0 = vf_gt();
    while (re_ld_acq(A.re_ctr) < A.re_np) {
      if (vf_gt() - g0 > 4000000ll) { atomicExch(A.re_ctr + 2, 1); break; }   // 4 ms: sticky error flag, no trap
      __nanosleep(32);
    }
    if (RE == 2 && A.re_mk) A.re_mk[rc * 4 + 1] = vf_gt();
  }
  __syncthreads();
  if (t < A.M) {
    const float* la = VF == 2 ? A.part : reinterpret_cast<const float*>(A.out2);
    const long long sstride = VF == 2 ? (long long)A.M * 256 : A.sst2;
    const long long l_ld = VF == 2 ? 256 : (long long)A.ldo2;
    float4 qa[8][2];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long long)t * l_ld) + lane * 2;
      qa[q][0] = __ldcg(pq);
      qa[q][1] = __ldcg(pq + 1);
    }
    const float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
    float s[8], b[8];
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in tail_kernel2's fixed order
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * A.re_mult));
      b[j] = s[j] + bs[j];
    }
    int sel_id[TK];
    float sel_s[TK];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      float bv = b[0], sv = s[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
      const unsigned u = __float_as_uint(bv + 0.f);   // -0 -> +0: equal floats get equal keys
      const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
      const unsigned mx = __reduce_max_sync(0xffffffffu, key);
      const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
      sel_s[k] = __shfl_sync(0xffffffffu, sv, owner);
      if (lane == owner) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (j == bj) b[j] = -INFINITY;
      }
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < TK; ++i) sum += sel_s[i];
      const float inv = A.re_rscale / (sum + A.re_deps);
#pragma unroll
      for (int k = 0; k < TK; ++k) {
        A.re_tw[t * TK + k] = sel_s[k] * inv;
        A.re_tid[t * TK + k] = sel_id[k];
        const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
        A.re_lid[t * TK + k] = rel < 32 ? rel : -1;
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (tx == 0) {
    if (RE == 2 && A.re_mk) A.re_mk[rc * 4 + 2] = vf_gt();
    if (atomicAdd(A.re_ctr + 1, 1) == NRC - 1) {   // the last routing CTA: every routing CTA has passed its spin
      atomicSub(A.re_ctr, A.re_np);
      atomicExch(A.re_ctr + 1, 0);
    }
  }
}

'''

VF_PRE = r'''  if constexpr (VF == 2 && RE > 0) {   // RE: this CTA's router share FIRST (router32_kernel<8>'s arithmetic), then release
    asm volatile("cp.async.wait_all;\n" ::: "memory");
    __syncthreads();
    {
      const float4* wsm = gd_vf_dsm + warp * 256;
      const float4* xs = gd_vf_dsm + 8 * 256;
      float racc[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) racc[j] = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c = i * 32 + lane;
        const float4 wv = wsm[c];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float4 xv = xs[j * 256 + c];
          racc[j] = fmaf(wv.x, xv.x, racc[j]);
          racc[j] = fmaf(wv.y, xv.y, racc[j]);
          racc[j] = fmaf(wv.z, xv.z, racc[j]);
          racc[j] = fmaf(wv.w, xv.w, racc[j]);
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float a = racc[j];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0 && j < A.M) A.part[((long)vf_sl * A.M + j) * 256 + vf_bx * 8 + warp] = a;
      }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(A.re_ctr, 1);
      if (RE == 2 && A.re_mk) A.re_mk[16 + GBX * 4 + 2] = vf_gt();
    }
  }
'''

GC_PARK = r'''    if constexpr (CAT > 0 && RE > 0) {
      if (s == CAT - 1) {   // RE: the router units ran first; park + store this K slice's router partial, release it
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float v = acc[0][mt][i];
            if constexpr (NC == 2) v += acc[1][mt][i];
            redr[warp][mt * 4 + i][lane] = v;
            acc[0][mt][i] = 0.f;
            if constexpr (NC == 2) acc[1][mt][i] = 0.f;
          }
        __syncthreads();
        {
          const int gc_t = GBX >> 3, gc_ks = GBX & 7;
          for (int i = (int)threadIdx.x; i < NO / 2; i += NT) {
            const int l = i & 31, gg = l >> 2, tt = l & 3;
            const int c = (i >> 5) & 1, mt = i >> 6;
            float v = 0.f;
#pragma unroll
            for (int k2 = 0; k2 < WK; ++k2) v += redr[k2][mt * 4 + c][l] + redr[k2][mt * 4 + c + 2][l];
            const int tok = tok0 + mt * 8 + 2 * tt + c;
            if (tok < A.M)
              reinterpret_cast<float*>(A.out2)[(size_t)gc_ks * A.sst2 + (size_t)tok * A.ldo2 + gc_t * 8 + gg] = v;
          }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          atomicAdd(A.re_ctr, 1);
          if (RE == 2 && A.re_mk) A.re_mk[16 + GBX * 4 + 2] = vf_gt();
        }
        ecur = 0;   // the pooler units consume the escape list from its start
        enx = __shfl_sync(0xffffffffu, em, 0);
      }
    }
'''

HOST = r'''
struct ReArm {
  int on; int* ctr; float* tw; int* tid; int* lid; const float* bias; const int* ls; long long* mk; float mult, rscale, deps;
};
static ReArm g_re = {};   // RE: armed by gd_re_arm() right before ONE gdr_run (vf 2) / gdc_run launch, consumed by it

static int re_take(Args& A, int np) {   // returns 0 (not armed), 1 (RE = 1) or 2 (RE = 2, marks)
  if (!g_re.on) return 0;
  g_re.on = 0;
  A.re_ctr = g_re.ctr; A.re_tw = g_re.tw; A.re_tid = g_re.tid; A.re_lid = g_re.lid; A.re_bias = g_re.bias; A.re_ls = g_re.ls;
  A.re_mk = g_re.mk; A.re_np = np; A.re_mult = g_re.mult; A.re_rscale = g_re.rscale; A.re_deps = g_re.deps;
  return g_re.mk != nullptr ? 2 : 1;
}

void gd_re_arm_raw(void* ctr, void* tw, void* tid, void* lid, const void* bias, const void* ls, void* mk, int on, float mult,
                   float rscale, float deps) {
  g_re.on = on; g_re.ctr = reinterpret_cast<int*>(ctr); g_re.tw = reinterpret_cast<float*>(tw);
  g_re.tid = reinterpret_cast<int*>(tid); g_re.lid = reinterpret_cast<int*>(lid); g_re.bias = reinterpret_cast<const float*>(bias);
  g_re.ls = reinterpret_cast<const int*>(ls); g_re.mk = reinterpret_cast<long long*>(mk);
  g_re.mult = mult; g_re.rscale = rscale; g_re.deps = deps;
}

template <int MT, int KU, int ST, int MB, int XP, int RE>
static int re_launch_gc(const Args& A, int nbx, int pdl, cudaStream_t st) {   // GC RE: nbx pooler CTAs + MT routing CTAs
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(nbx + MT), 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  unsigned na = 0;
  if (pdl & 1) {
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    na = 1;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<0, MT, 1, 8, KU, ST, 1, MB, XP, 0, 2, RE>, A);
}

int gd_re_info_raw(int k, int64_t* res) {   // 0 / 1: VF RE 1 / 2; 2 / 3: GC (2,1,8,18,2,1,0) RE 1 / 2
  cudaFuncAttributes fa;
  auto fn = k == 0 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2, 0, 1> : k == 1 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2, 0, 2>
                   : k == 2 ? gd_kernel<0, 2, 1, 8, 18, 2, 1, 2, 0, 0, 2, 1> : gd_kernel<0, 2, 1, 8, 18, 2, 1, 2, 0, 0, 2, 2>;
  const int ds = k < 2 ? 2 * 8 * 256 * 16 : 0;
  int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ds);
  if (e) return e;
  e = (int)cudaFuncGetAttributes(&fa, fn);
  if (e) return e;
  int occ = 0;
  e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, ds);
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes;
  return e;
}
'''

CPP_ADD = r'''void gd_re_arm_raw(void* ctr, void* tw, void* tid, void* lid, const void* bias, const void* ls, void* mk, int on, float mult,
                   float rscale, float deps);
int gd_re_info_raw(int k, int64_t* res);
void gd_re_arm(torch::Tensor ctr, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor bias, torch::Tensor ls,
               torch::Tensor mk, int64_t on, double mult, double rscale, double deps) {
  TORCH_CHECK(ctr.scalar_type() == at::kInt && ctr.numel() >= 4 && tw.scalar_type() == at::kFloat && tid.scalar_type() == at::kInt &&
              lid.scalar_type() == at::kInt && tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() &&
              tw.numel() == tid.numel() && tw.numel() == lid.numel() && bias.scalar_type() == at::kFloat && bias.numel() == 256 &&
              bias.is_contiguous() && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 && ls.scalar_type() == at::kInt &&
              ls.numel() >= 1 && mk.scalar_type() == at::kLong, "gd_re_arm: ctr / tw / tid / lid / bias / local_start / marks");
  gd_re_arm_raw(ctr.data_ptr(), tw.data_ptr(), tid.data_ptr(), lid.data_ptr(), bias.data_ptr(), ls.data_ptr(),
                mk.numel() >= 16 + 4 * 256 ? mk.data_ptr() : nullptr, (int)on, (float)mult, (float)rscale, (float)deps);
}
std::vector<int64_t> gd_re_info(int64_t k) {
  std::vector<int64_t> r(4, 0);
  const int e = gd_re_info_raw((int)k, r.data());
  TORCH_CHECK(e == 0, "gd_re_info: ", k, " error ", e);
  return r;
}
'''


def cuda(c):
    """Patch the GD_CUDA_SRC text c."""
    c = rep(c, "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n};",
            "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
            "  int* re_ctr = nullptr; const float* re_bias = nullptr; const int* re_ls = nullptr; float* re_tw = nullptr;   // RE\n"
            "  int* re_tid = nullptr; int* re_lid = nullptr; long long* re_mk = nullptr; int re_np = 0;\n"
            "  float re_mult = 0.f, re_rscale = 0.f, re_deps = 0.f;\n};", "Args RE fields")
    c = rep(c, "#define GC_XP(S_)", DEVICE + "#define GC_XP(S_)", "device re_route")
    c = rep(c, "#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))\n",
            "#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))\n"
            "#define GD_UM(S_) ((RE > 0 && CAT > 0) ? (((S_) < CAT) ? KUP + (S_) : (S_) - CAT) : (S_))   // RE: GC visit order\n"
            "#define GBX (RE > 0 ? (int)blockIdx.x - NRC : (int)blockIdx.x)   // RE: the pooler CTAs follow the routing CTAs\n",
            "GD_UM / GBX macros")
    c = rep(c, "#undef GC_XP\n", "#undef GC_XP\n#undef GD_UM\n#undef GBX\n", "undef GD_UM / GBX")
    c = rep(c, "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n"
            "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(",
            "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int RE = 0>\n"
            "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(", "template RE")
    c = rep(c, "  constexpr int XB = XP ? 2 : 1;\n",
            "  constexpr int XB = XP ? 2 : 1;\n"
            "  constexpr int NRC = RE > 0 ? (VF == 2 ? 1 : MT) : 0;   // RE: routing CTAs at the lowest blockIdx\n"
            "  static_assert(RE == 0 || (S == 1 && (VF == 2 || CAT > 0)), \"RE: VF = 2 / GC hosts only\");\n", "NRC")
    c = rep(c, "  if (A.trig) pdl_trigger();\n  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;\n",
            "  if (A.trig) pdl_trigger();\n"
            "  if constexpr (RE > 0) {\n"
            "    if ((int)blockIdx.x < NRC) {\n"
            "      re_route<VF, NRC, RE>(A, (int)blockIdx.x);\n"
            "      return;\n"
            "    }\n"
            "  }\n"
            "  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;\n", "routing CTA branch")
    c = rep(c, "  const int G = (int)(gridDim.x * gridDim.y) * NW;\n"
            "  const int gw = ((int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x) * NW + warp;\n",
            "  const int G = (RE > 0 ? (int)((gridDim.x - NRC) * gridDim.y) : (int)(gridDim.x * gridDim.y)) * NW;\n"
            "  const int gw = ((int)blockIdx.y * (RE > 0 ? (int)gridDim.x - NRC : (int)gridDim.x) + GBX) * NW + warp;\n", "G / gw")
    c = rep(c, "      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)i * wstep + p * 32);\n",
            "      for (int p = 0; p < NP; ++p) R[i][p] = ldw(wp + (size_t)GD_UM(i) * wstep + p * 32);\n", "initial prefetch")
    c = rep(c, "  const int vf_c = (int)blockIdx.x, vf_bx = vf_c & 31, vf_sl = vf_c >> 5;\n",
            "  const int vf_c = GBX, vf_bx = vf_c & 31, vf_sl = vf_c >> 5;\n"
            "  if constexpr (RE == 2) {\n"
            "    if (A.re_mk && threadIdx.x == 0) A.re_mk[16 + GBX * 4 + 0] = vf_gt();\n"
            "  }\n", "vf_c + entry mark")
    c = rep(c, "  pdl_wait();\n  if constexpr (VF >= 1) {\n    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();\n  }\n",
            "  pdl_wait();\n  if constexpr (VF >= 1) {\n    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();\n  }\n"
            "  if constexpr (RE == 2) {\n"
            "    if (A.re_mk && threadIdx.x == 0) A.re_mk[16 + GBX * 4 + 1] = vf_gt();\n"
            "  }\n", "post-wait mark")
    c = rep(c, "(A.x + (size_t)(((int)blockIdx.x & 7) * 16 + 2 * wk) * 64 + 16 * t);   // GC",
            "(A.x + (size_t)((GBX & 7) * 16 + 2 * wk) * 64 + 16 * t);   // GC", "xbr")
    c = rep(c, "    X[0][mt][0] = ldx(xb + xo[mt]);\n    X[0][mt][1] = ldx(xb + xo[mt] + 1);\n  }\n"
            "  int ecur = 0;\n  uint32_t enx = __shfl_sync(0xffffffffu, em, 0);\n",
            "    X[0][mt][0] = ldx((RE > 0 && CAT > 0 ? GC_XP(GD_UM(0)) : xb) + xo[mt]);\n"
            "    X[0][mt][1] = ldx((RE > 0 && CAT > 0 ? GC_XP(GD_UM(0)) : xb) + xo[mt] + 1);\n  }\n" + VF_PRE +
            "  int ecur = 0;\n"
            "  uint32_t enx;\n"
            "  if constexpr (RE > 0 && CAT > 0) {   // RE: the router units' escapes first\n"
            "    ecur = __popc(__ballot_sync(0xffffffffu, (int)(em >> 20) < KUP));\n"
            "    enx = ecur < 32 ? __shfl_sync(0xffffffffu, em, ecur & 31) : 0xFFFFFFFFu;\n"
            "  } else {\n"
            "    enx = __shfl_sync(0xffffffffu, em, 0);\n"
            "  }\n", "x0 / VF router first / escapes")
    c = rep(c, "          X[(s + 1) & (XB - 1)][mt][0] = ldx(GC_XP(s + 1) + xo[mt]);\n"
            "          X[(s + 1) & (XB - 1)][mt][1] = ldx(GC_XP(s + 1) + xo[mt] + 1);\n",
            "          X[(s + 1) & (XB - 1)][mt][0] = ldx(GC_XP(GD_UM(s + 1)) + xo[mt]);\n"
            "          X[(s + 1) & (XB - 1)][mt][1] = ldx(GC_XP(GD_UM(s + 1)) + xo[mt] + 1);\n", "XP x")
    c = rep(c, "        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);\n        X[0][mt][1] = ldx(GC_XP(s) + xo[mt] + 1);\n",
            "        X[0][mt][0] = ldx(GC_XP(GD_UM(s)) + xo[mt]);\n        X[0][mt][1] = ldx(GC_XP(GD_UM(s)) + xo[mt] + 1);\n", "x")
    c = rep(c, "    decode(R[slot], (CAT && s >= KUP) ? A.tbr : A.tb, a);\n",
            "    decode(R[slot], (CAT && GD_UM(s) >= KUP) ? A.tbr : A.tb, a);\n", "decode base")
    c = rep(c, "      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)(s + ST) * wstep + p * 32);\n",
            "      for (int p = 0; p < NP; ++p) R[slot][p] = ldw(wp + (size_t)GD_UM(s + ST) * wstep + p * 32);\n", "prefetch")
    c = rep(c, "    while ((int)(enx >> 20) == s) {\n", "    while ((int)(enx >> 20) == GD_UM(s)) {\n", "escape step")
    c = rep(c, "    if constexpr (CAT > 0) {\n      if (s == KUP - 1) {",
            GC_PARK + "    if constexpr (CAT > 0 && RE == 0) {\n      if (s == KUP - 1) {", "GC park")
    c = rep(c, "      if constexpr (CAT > 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;\n",
            "      if constexpr (CAT > 0 && RE == 0) redr[warp][mt * 4 + i][lane] = v; else red[warp][mt * 4 + i][lane] = v;\n", "final acc")
    c = rep(c, "    const int row = ((int)blockIdx.x * WR + w2) * 16 + gg + 8 * (c >> 1);\n",
            "    const int row = (GBX * WR + w2) * 16 + gg + 8 * (c >> 1);\n", "store row")
    c = rep(c, "    if constexpr (CAT > 0) {   // GC: this K slice's router partial",
            "    if constexpr (CAT > 0 && RE == 0) {   // GC: this K slice's router partial", "GC epilogue")
    c = rep(c, "      const int gc_t = (int)blockIdx.x >> 3, gc_ks = (int)blockIdx.x & 7;\n",
            "      const int gc_t = GBX >> 3, gc_ks = GBX & 7;\n", "gc_t")
    c = rep(c, "    if constexpr (VF == 2) {   // router32_kernel<8>'s arithmetic",
            "    if constexpr (VF == 2 && RE == 0) {   // router32_kernel<8>'s arithmetic", "VF epilogue router")
    c = rep(c, "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n  } else {",
            "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n"
            "    if constexpr (RE == 2) {\n"
            "      if (A.re_mk) {\n        __syncthreads();\n        if (threadIdx.x == 0) A.re_mk[16 + GBX * 4 + 3] = vf_gt();\n      }\n"
            "    }\n  } else {", "end mark")
    c = rep(c, "}  // namespace gd\n\nusing namespace gd;\n", "}  // namespace gd\n\nusing namespace gd;\n" + HOST, "host RE")
    # gdc_run_raw: an armed call launches the GC RE instance
    c = rep(c, "  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n  const int nz = 1;\n",
            "  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n"
            "  if (const int re = re_take(A, nbx)) {   // RE: the T = 9..16 GC route (2,1,8,18,2,1,0) only (checked by the host)\n"
            "    return re == 2 ? re_launch_gc<2, 18, 2, 2, 0, 2>(A, nbx, pdl, st) : re_launch_gc<2, 18, 2, 2, 0, 1>(A, nbx, pdl, st);\n"
            "  }\n"
            "  const int nz = 1;\n", "gdc_run_raw RE")
    # gdr_run_raw: vf 2 + armed -> the VF RE instance
    c = rep(c, "  if (vf == 2) {\n    static bool set = false;\n",
            "  if (vf == 2) {\n"
            "    if (const int re = re_take(A, nbx)) {   // RE: nbx pooler CTAs + 1 routing CTA\n"
            "      auto fn = re == 2 ? gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2, 0, 2> : gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 2, 0, 1>;\n"
            "      static bool set_re[3] = {false, false, false};\n"
            "      if (!set_re[re]) {\n"
            "        const int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, GDR_DS);\n"
            "        if (e) return e;\n"
            "        set_re[re] = true;\n"
            "      }\n"
            "      cfg.gridDim = dim3((unsigned)nbx + 1u, 1, 1);\n"
            "      cfg.dynamicSmemBytes = GDR_DS;\n"
            "      return (int)cudaLaunchKernelEx(&cfg, fn, A);\n"
            "    }\n"
            "    static bool set = false;\n", "gdr_run_raw RE")
    return c


def cpp(t):
    a = "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n"
    return rep(t, a, CPP_ADD + a, "CPP bindings")


PY_GLOBALS = r'''RE_MODE = {1: 0, 8: 0, 16: 0}   # L1 ROUTE-EARLY per GD bucket: 0 off (ship), 1 inert (router first + routing CTAs -> scratch)
RE_MK_ON = False   # lab: RE = 2 instances write per-CTA globaltimer marks into _RE["mk"]
RE_SELF_TEST_T = {1: (1,), 8: (8, 5, 2), 16: (16, 9)}
RE_GC_ROUTE = (2, 1, 8, 18, 2, 1, 0)   # the GC route the RE instance is compiled for (T = 9..16)
_RE = {"ok": {}, "ctr": None, "scr": None, "mk": None, "nomk": None, "ls": None, "seen": set()}
'''

PY_FUNCS = r'''

def _re_on(T):
    b = _gd_bucket(T)
    m = RE_MODE.get(b, 0) if b in (1, 8, 16) else 0
    return m if (m == 1 and _RE["ok"].get(b) and _RE["ls"] is not None) else 0


def _re_arm(mod, state, ls, T, outs=None):
    v8 = state["v8"]
    o = outs if outs is not None else tuple(s[:T] for s in _RE["scr"])
    mod.gd_re_arm(_RE["ctr"], o[0], o[1], o[2], v8["bias"], ls, _RE["mk"] if RE_MK_ON else _RE["nomk"], 1,
                  0.1767766952966369, 1.0, 1e-15)
    if (T, RE_MK_ON) not in _RE["seen"]:
        _RE["seen"].add((T, RE_MK_ON))
        print(f"[re] first arm T={T} marks {RE_MK_ON}", flush=True)


def _re_test(state, b, T, ls, seed, pdl, mk):
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn((T, HIDDEN), generator=gen, device=dev)
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
    hb = h.to(torch.bfloat16).contiguous()
    lst = torch.tensor([ls], dtype=torch.int32, device=dev)
    outs = (torch.full((T, TOP_K), float("nan"), dtype=torch.float32, device=dev),
            torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev), torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev))
    p1 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
    y1 = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
    global RE_MK_ON
    if b in (1, 8):
        r = _gd_live(state, "pooler", T)
        if r is None or not r["desc"].startswith("m1(1,8,16,1)s2x1") or r["nbx"] != 256:
            return f"GD pooler route {None if r is None else r['desc']} is not (1,8,16,1)s2x1"
        if T == 1:   # ship T = 1: GD pooler + lean router (vf_router32_kernel<1>)
            y0 = _gd(state, "pooler", hb)
            p0 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
            _SBB1["mod"].vf_router32(h, state["router"], p0, 0)
        else:   # ship T = 2..8: the fused VF launch
            y0, p0 = _vfr_gdr(state, hb, h, 2)
        saved, RE_MK_ON = RE_MK_ON, mk
        try:
            _re_arm(r["mod"], state, lst, T, outs)
        finally:
            RE_MK_ON = saved
        r["mod"].gdr_run(2, r["w"], r["esc"], hb, y1, T, r["K"], r["N"], r["tb"], r["nbx"], pdl, h, state["router"], p1, _RE["nomk"])
    else:
        r = (state.get("gd_cat") or {}).get(16)
        if r is None or not r.get("re"):
            return "GC cat/16 route not live / not the RE geometry"
        y0 = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
        p0 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        _gd_cat_run(r, hb, y0, p0, T, GD_CAT_PDL)
        saved, RE_MK_ON = RE_MK_ON, mk
        try:
            _re_arm(r["mod"], state, lst, T, outs)
        finally:
            RE_MK_ON = saved
        _gd_cat_run(r, hb, y1, p1, T, pdl)
    ref = _v8_tail(y1, p1, state, {"local_start": lst})
    torch.cuda.synchronize()
    c = _RE["ctr"].tolist()
    if any(c[:4]):
        return f"counters {c[:4]} not reset / error flag"
    if not torch.equal(y1, y0):
        return "pooler output differs from the ship kernel"
    if not torch.equal(p1, p0):
        return "router partials differ from the ship producer"
    tw, tid, lid = outs
    if not (torch.equal(tid, ref[2]) and torch.equal(lid, ref[3]) and torch.equal(tw, ref[1])):
        return (f"routing differs (tid/lid mismatches {int((tid != ref[2]).sum()) + int((lid != ref[3]).sum())}, "
                f"tw equal {bool(torch.equal(tw, ref[1]))})")
    return None


def _re_prepare(state):
    _RE["ok"] = {}
    mod = _GD.get("mod")
    if state.get("v8") is None or mod is None or not hasattr(mod, "gd_re_arm") or not _VFR["ok"]:
        print("[re] FALLBACK: GD module / v8 tail / VF missing -> RE off", flush=True)
        return
    dev = state["router"].device
    _RE["ctr"] = torch.zeros((8,), dtype=torch.int32, device=dev)
    _RE["scr"] = (torch.empty((32, TOP_K), dtype=torch.float32, device=dev), torch.empty((32, TOP_K), dtype=torch.int32, device=dev),
                  torch.empty((32, TOP_K), dtype=torch.int32, device=dev))
    _RE["mk"] = torch.zeros((16 + 4 * 256,), dtype=torch.int64, device=dev)
    _RE["nomk"] = torch.empty((0,), dtype=torch.int64, device=dev)
    try:
        infos = [list(mod.gd_re_info(k)) for k in range(4)]
    except Exception as ex:
        print(f"[re] FALLBACK: gd_re_info raised {ex!r} -> RE off", flush=True)
        return
    print(f"[re] instances regs / local / occ / smem: vf {infos[0]} vf_mk {infos[1]} gc {infos[2]} gc_mk {infos[3]}", flush=True)
    gc = (state.get("gd_cat") or {}).get(16)
    if gc is not None:
        gc["re"] = tuple(GD_CAT_ROUTE.get(16) or ()) == RE_GC_ROUTE and gc["nbx"] == 256 and infos[2][2] >= 2 and infos[2][1] == 0
    for b, Ts in RE_SELF_TEST_T.items():
        k = 0 if b < 16 else 2
        msgs = []
        if infos[k][1] > 0 or infos[k][2] < 2 or infos[k + 1][2] < 2:
            msgs.append(f"instance spills / occupancy {infos[k]} {infos[k + 1]}")
        try:
            for T in Ts:
                if msgs:
                    break
                for j, ls in enumerate((0, 224)):
                    for rep in range(3):
                        e = _re_test(state, b, T, ls, 9300 + 31 * T + 7 * j + rep, (0, 3, 3)[rep], rep == 2)
                        if e is not None:
                            msgs.append(f"T={T} ls={ls} rep {rep}: {e}")
                            break
        except Exception as ex:
            msgs.append(f"EXC {ex!r}")
        finally:
            _RE["ctr"].zero_()
        _RE["ok"][b] = not msgs
        print(f"[re] self-test bucket {b}: {'ok (pooler y / partials bitwise vs ship, tid / lid / tw bitwise vs the tail)' if not msgs else 'FALLBACK ' + msgs[0]}",
              flush=True)
'''

PY_PATCHES = [
    ("\n_GD_CAT_SEEN = set()\n", "\n" + PY_GLOBALS + "_GD_CAT_SEEN = set()\n", "RE globals"),
    ("\n\ndef _gd_cat_live(state, tokens):\n", PY_FUNCS + "\n\ndef _gd_cat_live(state, tokens):\n", "RE functions"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gd_re_arm", "gd_re_info"]',
     "GD module exports"),
    ("    r = _gd_live(state, \"pooler\", T)\n    if r is None or _gd_bucket(T) != 8:\n        return None\n",
     "    r = _gd_live(state, \"pooler\", T)\n"
     "    if r is None or (_gd_bucket(T) != 8 and not (T == 1 and vf == 2 and _re_on(1) and r[\"desc\"].startswith(\"m1(1,8,16,1)s2x1\"))):\n"
     "        return None\n", "_vfr_gdr bucket"),
    ("    mk = _VFR[\"marks\"] if VF_GDM else _VFR[\"nomk\"]\n",
     "    mk = _VFR[\"marks\"] if VF_GDM else _VFR[\"nomk\"]\n"
     "    if vf == 2 and _re_on(T):\n"
     "        _re_arm(r[\"mod\"], state, _RE[\"ls\"], T)\n", "_vfr_gdr arm"),
    ("    if not (2 <= ffn_input.shape[0] <= 8 and router_input.is_contiguous() and ffn_input.is_contiguous()):\n",
     "    if not ((2 <= ffn_input.shape[0] <= 8 or (ffn_input.shape[0] == 1 and _re_on(1))) and router_input.is_contiguous()\n"
     "            and ffn_input.is_contiguous()):\n", "_vfr_fused T=1"),
    ("        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
     "        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        if r.get(\"re\") and _re_on(tokens):\n"
     "            _re_arm(r[\"mod\"], state, _RE[\"ls\"], tokens)\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n", "_gd_cat arm"),
    ("    mode, place = _v8_router_mode(tokens, state)\n",
     "    _RE[\"ls\"] = data[\"local_start\"] if _RE[\"ok\"] else None\n    mode, place = _v8_router_mode(tokens, state)\n",
     "forward: RE per-call local_start"),
    ("    torch.cuda.synchronize()\n    _pt1_join_all()\n", "    torch.cuda.synchronize()\n    _pt1_join_all()\n    _re_prepare(state)\n",
     "prepare: RE self-tests"),
]


def py(t):
    for a, b, what in PY_PATCHES:
        t = rep(t, a, b, what)
    return t


def apply(text):
    a0 = text.index("GD_CUDA_SRC = r'''")
    a1 = text.index(NL + "'''" + NL, a0)
    t = text[:a0] + cuda(text[a0:a1]) + text[a1:]
    return py(cpp(t))

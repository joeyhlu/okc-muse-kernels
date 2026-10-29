"""l1b_patches.py: L1b ROUTE-EARLY (FRONT builder; lead variant after the L1 in-launch KILL R-l1f-0 / R-re-0) on a layer's GD
block (base k3_gc_20). Step 0 at T = 1..8 only (VF host buckets 1 / 8; T = 9..16 later if T <= 8 passes):

The pooler launch is NOT changed in arithmetic; the router moves in front of it, in the same stream:
  sbb1 -> R (reb_router_kernel: 128 CTAs x 2 VF router pieces; router weights -> smem BEFORE its PDL wait, i.e. during the
          attention boundary; after the wait: h-ready flag, x32 slice, router32_kernel<8>'s arithmetic verbatim -> partials)
       -> K (reb_route_kernel: ceil(T/8) CTAs, triggers at entry, PDL-waits on R, tail_kernel2's routing warp per token ->
          tw / tid / lid, routing-done flag)
       -> P (gd_kernel FG = 1 = the GD pooler (1,8,16) instance: no griddepcontrol.wait; thread 0 acquire-polls R's h-ready flag,
          X loads ld.global.cg; every CTA leaves only after K's routing-done flag, so P complete => K complete => R complete)
       -> tail2 (unchanged in step 0: routing + norm on R's partials + P's y) -> VMS (unchanged).
Flags (int32 x 4, _L1B["fl"]): [0] h ready (reset by R CTA 0 before its trigger), [1] routing done (reset by K before its
trigger), [2] sticky error. Causality: P launches only after every K CTA triggered, K only after every R CTA triggered, so
both resets precede any poll; R sets [0] after griddepcontrol.wait returned (sbb1 complete and visible).
FG = 0 instances are PTX-identical to the base (tools/l1f_local_compile.py). Marks (L1B_MK_ON): mk[0..2] K entry / wait /
done, mk[16 + c*4 + {0 entry, 1 flag pass, 3 end}] P CTA c, mk[1040 + c*2 + {0 post-wait, 1 end}] R CTA c.
"""
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[l1b_patches] FAIL {msg}")


def rep(t, old, new, what, n=1):
    check(t.count(old) == n, f"anchor {what}: {t.count(old)} matches (want {n})")
    return t.replace(old, new)


ROUTER_MATH = r'''      float racc[8];
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
        if (lane == 0 && j < B.M) B.part[((long)vf_sl * B.M + j) * 256 + vf_bx * 8 + warp] = a;
      }
'''

DEVICE = r'''
// L1b (route-early in front of the pooler): R = router kernel, K = routing kernel, P = gd_kernel FG = 1 (flag-gated pooler)
__device__ __forceinline__ uint4 ldx_cg(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ int reb_ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void reb_st_rel(int* p, int v) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void reb_poll(const int* f, int* err) {   // bounded acquire spin (globaltimer), sticky error, no trap
  const long long g0 = vf_gt();
  while (reb_ld_acq(f) != 1) {
    if (vf_gt() - g0 > 4000000ll) { atomicExch(err, 1); break; }
    __nanosleep(32);
  }
}
struct ReB {
  const float* x32; const float* w32; float* part; int M, trig;
  const float* bias; const int* ls; float* tw; int* tid; int* lid; float mult, rscale, deps;
  int* fl; long long* mk;
};
constexpr int REB_DS = (2 * 8 * 256 + 8 * 256) * 16;   // R: 2 pieces of 8 router rows x 1024 K + the x slice [8][1024] f32

__global__ void __launch_bounds__(256) reb_router_kernel(const __grid_constant__ ReB B) {
  const int cb = (int)blockIdx.x, tx = (int)threadIdx.x, lane = tx & 31, warp = tx >> 5;
  const int vf_sl = (2 * cb) >> 5;
  if (cb == 0 && tx == 0) { *reinterpret_cast<volatile int*>(B.fl) = 0; __threadfence(); }
  __syncthreads();
  if (B.trig) pdl_trigger();
  for (int idx = tx; idx < 2 * 8 * 256; idx += 256) {   // router weights: no producer dependency (streams during the boundary)
    const int pc = 2 * cb + (idx >> 11), r = (idx >> 8) & 7, col = idx & 255;
    vf_cp16(gd_vf_dsm + idx, reinterpret_cast<const float4*>(B.w32 + (size_t)((pc & 31) * 8 + r) * 8192 + vf_sl * 1024) + col);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  pdl_wait();
  if (cb == 0 && tx == 0) { __threadfence(); reb_st_rel(B.fl, 1); }   // sbb1 complete: the pooler may read h
  if (B.mk && tx == 0) B.mk[1040 + cb * 2] = vf_gt();
  float4* xs = gd_vf_dsm + 2 * 8 * 256;
  for (int idx = tx; idx < 8 * 256; idx += 256) {
    const int j = idx >> 8, col = idx & 255;
    if (j < B.M) vf_cp16(xs + idx, reinterpret_cast<const float4*>(B.x32 + (size_t)j * 8192 + vf_sl * 1024) + col);
    else xs[idx] = make_float4(0.f, 0.f, 0.f, 0.f);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
#pragma unroll 1
  for (int pi = 0; pi < 2; ++pi) {   // router32_kernel<8>'s arithmetic for row vf_bx * 8 + warp, K slice vf_sl (verbatim VF = 2)
    const int vf_bx = (2 * cb + pi) & 31;
    const float4* wsm = gd_vf_dsm + pi * 2048 + warp * 256;
    {
''' + ROUTER_MATH + r'''    }
  }
  if (B.mk) {
    __syncthreads();
    if (tx == 0) B.mk[1040 + cb * 2 + 1] = vf_gt();
  }
}

__global__ void __launch_bounds__(256) reb_route_kernel(const __grid_constant__ ReB B) {
  constexpr int TK = 8, NE = 256;
  const int tx = (int)threadIdx.x, lane = tx & 31, warp = tx >> 5;
  const int t = (int)blockIdx.x * 8 + warp;
  if (tx == 0) { *reinterpret_cast<volatile int*>(B.fl + 1) = 0; __threadfence(); }
  __syncthreads();
  if (B.trig) pdl_trigger();
  if (B.mk && tx == 0) B.mk[blockIdx.x * 4 + 0] = vf_gt();
  const float4* pbias = reinterpret_cast<const float4*>(B.bias) + lane * 2;
  const float4 c0 = pbias[0], c1 = pbias[1];
  const int ls = B.ls[0];
  pdl_wait();   // R complete: its partials are visible
  if (B.mk && tx == 0) B.mk[blockIdx.x * 4 + 1] = vf_gt();
  if (t < B.M) {   // tail_kernel2's routing warp, verbatim (8 slices summed in order, + 0 of the absent lb)
    const long long sstride = (long long)B.M * 256;
    float4 qa[8][2];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      const float4* pq = reinterpret_cast<const float4*>(B.part + q * sstride + (long long)t * 256) + lane * 2;
      qa[q][0] = __ldcg(pq);
      qa[q][1] = __ldcg(pq + 1);
    }
    const float4 qb0 = make_float4(0.f, 0.f, 0.f, 0.f), qb1 = qb0;
    float s[8], b[8];
    float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
    for (int q = 1; q < 8; ++q) {
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += qb0.x; a0.y += qb0.y; a0.z += qb0.z; a0.w += qb0.w; a1.x += qb1.x; a1.y += qb1.y; a1.z += qb1.z; a1.w += qb1.w;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * B.mult));
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
      const float inv = B.rscale / (sum + B.deps);
#pragma unroll
      for (int k = 0; k < TK; ++k) {
        B.tw[t * TK + k] = sel_s[k] * inv;
        B.tid[t * TK + k] = sel_id[k];
        const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
        B.lid[t * TK + k] = rel < 32 ? rel : -1;
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (tx == 0) {
    if (B.mk) B.mk[blockIdx.x * 4 + 2] = vf_gt();
    reb_st_rel(B.fl + 1, 1);
  }
}

'''

HOST = r'''
static int reb_launch(void (*fn)(ReB), const ReB& B, int grid, int ds, int pdl, cudaStream_t st) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)grid, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = (size_t)ds;
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
  return (int)cudaLaunchKernelEx(&cfg, fn, B);
}

// L1b: R + K + P on stream st (P = the GD pooler (1,8,16,2,1,2,1) instance with FG = 1, nbx = 256 row tiles)
int gd_reb_run_raw(const float* x32, const float* w32, float* part, const void* w, const void* esc, const void* h, void* y, int M,
                   uint32_t tb, const float* bias, const int* ls, float* tw, int* tid, int* lid, float mult, float rscale, float deps,
                   int* fl, long long* mk, int pdl, cudaStream_t st) {
  static bool set = false;
  if (!set) {
    const int e = (int)cudaFuncSetAttribute(reb_router_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, REB_DS);
    if (e) return e;
    set = true;
  }
  ReB B;
  B.x32 = x32; B.w32 = w32; B.part = part; B.M = M; B.trig = (pdl >> 1) & 1;
  B.bias = bias; B.ls = ls; B.tw = tw; B.tid = tid; B.lid = lid; B.mult = mult; B.rscale = rscale; B.deps = deps;
  B.fl = fl; B.mk = mk;
  int e = reb_launch(reb_router_kernel, B, 128, REB_DS, pdl, st);
  if (e) return e;
  e = reb_launch(reb_route_kernel, B, (M + 7) / 8, 0, pdl, st);
  if (e) return e;
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(h);
  A.out = y;
  A.M = M; A.K = 8192; A.ldo = 4096; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;
  A.re_flag = fl; A.re_mk = mk;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(256, 1, 1);
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
  return (int)cudaLaunchKernelEx(&cfg, gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 0, 0, 1>, A);
}

int gd_reb_info_raw(int k, int64_t* res) {   // 0: R, 1: K, 2: P
  cudaFuncAttributes fa;
  int e = 0, occ = 0, ds = 0;
  if (k == 0) {
    ds = REB_DS;
    e = (int)cudaFuncSetAttribute(reb_router_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ds);
    if (!e) e = (int)cudaFuncGetAttributes(&fa, reb_router_kernel);
    if (!e) e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reb_router_kernel, 256, ds);
  } else if (k == 1) {
    e = (int)cudaFuncGetAttributes(&fa, reb_route_kernel);
    if (!e) e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, reb_route_kernel, 256, 0);
  } else {
    auto fn = gd_kernel<0, 1, 1, 8, 16, 2, 1, 2, 1, 0, 0, 1>;
    e = (int)cudaFuncGetAttributes(&fa, fn);
    if (!e) e = (int)cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, 0);
  }
  if (e) return e;
  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = occ; res[3] = (int64_t)fa.sharedSizeBytes + ds;
  return 0;
}
'''

CPP_ADD = r'''int gd_reb_run_raw(const float* x32, const float* w32, float* part, const void* w, const void* esc, const void* h, void* y, int M,
                   uint32_t tb, const float* bias, const int* ls, float* tw, int* tid, int* lid, float mult, float rscale, float deps,
                   int* fl, long long* mk, int pdl, cudaStream_t st);
int gd_reb_info_raw(int k, int64_t* res);
void gd_reb_run(torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor w, torch::Tensor esc, torch::Tensor h,
                torch::Tensor y, int64_t tb, torch::Tensor bias, torch::Tensor ls, torch::Tensor tw, torch::Tensor tid,
                torch::Tensor lid, torch::Tensor fl, torch::Tensor mk, double mult, double rscale, double deps, int64_t pdl) {
  const int64_t M = h.size(0);
  TORCH_CHECK(M >= 1 && M <= 8 && x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.size(0) == M && x32.size(1) == 8192 &&
              w32.scalar_type() == at::kFloat && w32.is_contiguous() && w32.size(0) == 256 && w32.size(1) == 8192 &&
              part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == 8 * M * 256 &&
              h.scalar_type() == at::kBFloat16 && h.is_contiguous() && h.size(1) == 8192 && y.scalar_type() == at::kBFloat16 &&
              y.is_contiguous() && y.size(0) == M && y.size(1) == 4096 && bias.scalar_type() == at::kFloat && bias.numel() == 256 &&
              ls.scalar_type() == at::kInt && tw.scalar_type() == at::kFloat && tid.scalar_type() == at::kInt &&
              lid.scalar_type() == at::kInt && tw.numel() == M * 8 && tid.numel() == M * 8 && lid.numel() == M * 8 &&
              tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() && fl.scalar_type() == at::kInt && fl.numel() >= 4 &&
              mk.scalar_type() == at::kLong &&
              reinterpret_cast<uintptr_t>(x32.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w32.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(h.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "gd_reb_run: shapes / dtypes / alignment");
  const int e = gd_reb_run_raw(x32.data_ptr<float>(), w32.data_ptr<float>(), part.data_ptr<float>(), w.data_ptr(), esc.data_ptr(),
                               h.data_ptr(), y.data_ptr(), (int)M, (uint32_t)tb, bias.data_ptr<float>(), ls.data_ptr<int>(),
                               tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(), (float)mult, (float)rscale, (float)deps,
                               fl.data_ptr<int>(), mk.numel() >= 2048 ? reinterpret_cast<long long*>(mk.data_ptr()) : nullptr,
                               (int)pdl, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_reb_run: CUDA error ", e, " ", cudaGetErrorString((cudaError_t)e));
}
std::vector<int64_t> gd_reb_info(int64_t k) {
  std::vector<int64_t> r(4, 0);
  const int e = gd_reb_info_raw((int)k, r.data());
  TORCH_CHECK(e == 0, "gd_reb_info: ", k, " error ", e);
  return r;
}
'''


def cuda(c):
    c = rep(c, "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n};",
            "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
            "  int* re_flag = nullptr; long long* re_mk = nullptr;   // L1b FG pooler: [0] h ready, [1] routing done, [2] error; marks\n};",
            "Args L1b fields")
    c = rep(c, "#define GC_XP(S_)", DEVICE + "#define GC_XP(S_)", "device L1b")
    c = rep(c, "#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))\n",
            "#define GC_XP(S_) (((S_) < KUP) ? (xb + (S_) * 8) : (xbr + ((S_) - KUP) * 8))\n"
            "#define GD_LDX(P_) (FG ? ldx_cg(P_) : ldx(P_))   // L1b: the flag-gated pooler reads h through L2\n", "GD_LDX")
    c = rep(c, "#undef GC_XP\n", "#undef GC_XP\n#undef GD_LDX\n", "undef GD_LDX")
    c = rep(c, "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n"
            "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(",
            "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int FG = 0>\n"
            "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(", "template FG")
    c = rep(c, "  constexpr int XB = XP ? 2 : 1;\n",
            "  constexpr int XB = XP ? 2 : 1;\n"
            "  static_assert(FG == 0 || (S == 1 && VF == 0 && CAT == 0), \"FG: plain S = 1 pooler only\");\n", "FG assert")
    c = rep(c, "  pdl_wait();\n  if constexpr (VF >= 1) {\n    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();\n  }\n",
            "  if constexpr (FG) {   // L1b: h is complete once R passed its PDL wait (flag); this launch never griddepcontrol.waits\n"
            "    if (A.re_mk && threadIdx.x == 0) A.re_mk[16 + (int)blockIdx.x * 4 + 0] = vf_gt();\n"
            "    if (threadIdx.x == 0) reb_poll(A.re_flag, A.re_flag + 2);\n"
            "    __syncthreads();\n"
            "    if (A.re_mk && threadIdx.x == 0) A.re_mk[16 + (int)blockIdx.x * 4 + 1] = vf_gt();\n"
            "  } else {\n"
            "    pdl_wait();\n"
            "  }\n"
            "  if constexpr (VF >= 1) {\n    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2] = vf_gt();\n  }\n", "FG flag gate")
    c = rep(c, "    X[0][mt][0] = ldx(xb + xo[mt]);\n    X[0][mt][1] = ldx(xb + xo[mt] + 1);\n",
            "    X[0][mt][0] = GD_LDX(xb + xo[mt]);\n    X[0][mt][1] = GD_LDX(xb + xo[mt] + 1);\n", "x0")
    c = rep(c, "          X[(s + 1) & (XB - 1)][mt][0] = ldx(GC_XP(s + 1) + xo[mt]);\n"
            "          X[(s + 1) & (XB - 1)][mt][1] = ldx(GC_XP(s + 1) + xo[mt] + 1);\n",
            "          X[(s + 1) & (XB - 1)][mt][0] = GD_LDX(GC_XP(s + 1) + xo[mt]);\n"
            "          X[(s + 1) & (XB - 1)][mt][1] = GD_LDX(GC_XP(s + 1) + xo[mt] + 1);\n", "XP x")
    c = rep(c, "        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);\n        X[0][mt][1] = ldx(GC_XP(s) + xo[mt] + 1);\n",
            "        X[0][mt][0] = GD_LDX(GC_XP(s) + xo[mt]);\n        X[0][mt][1] = GD_LDX(GC_XP(s) + xo[mt] + 1);\n", "x")
    c = rep(c, "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n  } else {",
            "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n"
            "    if constexpr (FG) {   // L1b: leave after K's routing-done flag, so P complete => routing (and R) complete\n"
            "      __syncthreads();\n"
            "      if (threadIdx.x == 0) {\n"
            "        reb_poll(A.re_flag + 1, A.re_flag + 2);\n"
            "        if (A.re_mk) A.re_mk[16 + (int)blockIdx.x * 4 + 3] = vf_gt();\n"
            "      }\n"
            "    }\n  } else {", "FG end")
    c = rep(c, "}  // namespace gd\n\nusing namespace gd;\n", "}  // namespace gd\n\nusing namespace gd;\n" + HOST, "host L1b")
    return c


def cpp(t):
    a = "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n"
    return rep(t, a, CPP_ADD + a, "CPP bindings")


PY_GLOBALS = r'''L1B_MODE = {1: 0, 8: 0}   # L1b per GD bucket: 0 ship, 1 inert (R + K -> scratch routing + flag-gated P; tail2 / VMS unchanged)
L1B_MK_ON = False   # lab: K / P / R globaltimer marks into _L1B["mk"]
L1B_SELF_TEST_T = (1, 2, 5, 8)
_L1B = {"ok": {}, "fl": None, "scr": None, "mk": None, "nomk": None, "ls": None, "seen": set()}
'''

PY_FUNCS = r'''

def _reb_on(T):
    b = _gd_bucket(T)
    return b in (1, 8) and L1B_MODE.get(b, 0) == 1 and bool(_L1B["ok"].get(b)) and _L1B["ls"] is not None


def _reb_launch(state, r, h, x32, ls, outs, pdl, mk):
    T = h.shape[0]
    y = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=h.device)
    part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=h.device)
    r["mod"].gd_reb_run(x32, state["router"], part, r["w"], r["esc"], h, y, r["tb"], state["v8"]["bias"], ls, outs[0], outs[1],
                        outs[2], _L1B["fl"], _L1B["mk"] if mk else _L1B["nomk"], 0.1767766952966369, 1.0, 1e-15, pdl)
    return y, part


def _reb_run(state, h, x32):
    T = h.shape[0]
    r = _gd_live(state, "pooler", T)
    if r is None or not r["desc"].startswith("m1(1,8,16,1)s2x1") or r["nbx"] != 256:
        return None
    outs = tuple(s[:T] for s in _L1B["scr"])
    pdl = (1 if GD_PDL.get("pooler") else 0) | (2 if GD_TRIG.get("pooler") else 0)
    if (T, L1B_MK_ON) not in _L1B["seen"]:
        _L1B["seen"].add((T, L1B_MK_ON))
        print(f"[l1b] first call T={T} pdl {pdl} marks {L1B_MK_ON}", flush=True)
    return _reb_launch(state, r, h, x32, _L1B["ls"], outs, pdl, L1B_MK_ON)


def _reb_test(state, T, ls, seed, pdl, mk):
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn((T, HIDDEN), generator=gen, device=dev)
    z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
    hb = (z * 0.7).to(torch.bfloat16).contiguous()
    lst = torch.tensor([ls], dtype=torch.int32, device=dev)
    r = _gd_live(state, "pooler", T)
    if r is None or not r["desc"].startswith("m1(1,8,16,1)s2x1") or r["nbx"] != 256:
        return f"GD pooler route {None if r is None else r['desc']} is not (1,8,16,1)s2x1"
    if T == 1:   # ship T = 1: GD pooler + lean router (vf_router32_kernel<1>)
        y0 = _gd(state, "pooler", hb)
        p0 = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        _SBB1["mod"].vf_router32(z, state["router"], p0, 0)
    else:   # ship T = 2..8: the fused VF launch
        y0, p0 = _vfr_gdr(state, hb, z, 2)
    outs = (torch.full((T, TOP_K), float("nan"), dtype=torch.float32, device=dev),
            torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev), torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev))
    y1, p1 = _reb_launch(state, r, hb, z, lst, outs, pdl, mk)
    ref = _v8_tail(y1, p1, state, {"local_start": lst})
    torch.cuda.synchronize()
    f = _L1B["fl"].tolist()
    if f[:3] != [1, 1, 0]:
        return f"flags {f[:3]} (want [1, 1, 0])"
    if not torch.equal(y1, y0):
        return "pooler output differs from the ship kernel"
    if not torch.equal(p1, p0):
        return "router partials differ from the ship producer"
    tw, tid, lid = outs
    if not (torch.equal(tid, ref[2]) and torch.equal(lid, ref[3]) and torch.equal(tw, ref[1])):
        return (f"routing differs (tid/lid mismatches {int((tid != ref[2]).sum()) + int((lid != ref[3]).sum())}, "
                f"tw equal {bool(torch.equal(tw, ref[1]))})")
    return None


def _reb_prepare(state):
    _L1B["ok"] = {}
    mod = _GD.get("mod")
    if state.get("v8") is None or mod is None or not hasattr(mod, "gd_reb_run") or not _VFR["ok"] or _SBB1.get("mod") is None:
        print("[l1b] FALLBACK: GD module / v8 tail / VF / SBB1 missing -> L1b off", flush=True)
        return
    dev = state["router"].device
    _L1B["fl"] = torch.zeros((4,), dtype=torch.int32, device=dev)
    _L1B["scr"] = (torch.empty((8, TOP_K), dtype=torch.float32, device=dev), torch.empty((8, TOP_K), dtype=torch.int32, device=dev),
                   torch.empty((8, TOP_K), dtype=torch.int32, device=dev))
    _L1B["mk"] = torch.zeros((2048,), dtype=torch.int64, device=dev)
    _L1B["nomk"] = torch.empty((0,), dtype=torch.int64, device=dev)
    try:
        infos = [list(mod.gd_reb_info(k)) for k in range(3)]
    except Exception as ex:
        print(f"[l1b] FALLBACK: gd_reb_info raised {ex!r} -> L1b off", flush=True)
        return
    print(f"[l1b] regs / local / occ / smem: R {infos[0]} K {infos[1]} P {infos[2]}", flush=True)
    msgs = []
    if infos[2][1] > 0 or infos[2][2] < 2 or infos[0][1] > 0 or infos[0][2] < 1:
        msgs.append(f"spills / occupancy R {infos[0]} P {infos[2]}")
    try:
        for T in L1B_SELF_TEST_T:
            if msgs:
                break
            for j, ls in enumerate((0, 224)):
                for rep in range(3):
                    e = _reb_test(state, T, ls, 9500 + 31 * T + 7 * j + rep, (0, 3, 3)[rep], rep == 2)
                    if e is not None:
                        msgs.append(f"T={T} ls={ls} rep {rep}: {e}")
                        break
    except Exception as ex:
        msgs.append(f"EXC {ex!r}")
    finally:
        _L1B["fl"].zero_()
    for b in (1, 8):
        _L1B["ok"][b] = not msgs
    print(f"[l1b] self-test T {L1B_SELF_TEST_T}: {'ok (y / partials bitwise vs ship, tid / lid / tw bitwise vs the tail, flags)' if not msgs else 'FALLBACK ' + msgs[0]}",
          flush=True)
'''

PY_PATCHES = [
    ("\n_GD_CAT_SEEN = set()\n", "\n" + PY_GLOBALS + "_GD_CAT_SEEN = set()\n", "L1b globals"),
    ("\n\ndef _gd_cat_live(state, tokens):\n", PY_FUNCS + "\n\ndef _gd_cat_live(state, tokens):\n", "L1b functions"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gd_reb_run", "gd_reb_info"]',
     "GD module exports"),
    ("def _vfr_fused(state, ffn_input, router_input):\n",
     "def _vfr_fused(state, ffn_input, router_input):\n"
     "    if (_reb_on(ffn_input.shape[0]) and router_input is not None and router_input.dtype == torch.float32\n"
     "            and router_input.is_contiguous() and ffn_input.is_contiguous()):\n"
     "        out = _reb_run(state, ffn_input, router_input)\n"
     "        if out is not None:\n"
     "            return out\n", "_vfr_fused L1b hook"),
    ("    mode, place = _v8_router_mode(tokens, state)\n",
     "    _L1B[\"ls\"] = data[\"local_start\"] if _L1B[\"ok\"] else None\n    mode, place = _v8_router_mode(tokens, state)\n",
     "forward: L1b per-call local_start"),
    ("    torch.cuda.synchronize()\n    _pt1_join_all()\n", "    torch.cuda.synchronize()\n    _pt1_join_all()\n    _reb_prepare(state)\n",
     "prepare: L1b self-tests"),
]


def apply(text):
    a0 = text.index("GD_CUDA_SRC = r'''")
    a1 = text.index(NL + "'''" + NL, a0)
    t = text[:a0] + cuda(text[a0:a1]) + text[a1:]
    t = rep(t, "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n",
            CPP_ADD + "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n",
            "CPP bindings")
    for a, b, what in PY_PATCHES:
        t = rep(t, a, b, what)
    return t

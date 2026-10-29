"""cf_patches.py: L2 COMB-FOLD (BACK builder, notes/study/lead_oct29.md s3) source patches.

GD (GD_CUDA_SRC): template parameter XC (default 0 = today's kernel). XC = 1 is the T <= 8 unpooler route (16 warps, MT 1,
K 4096) fed by VMS-CF instead of VMS's combined output: after griddepcontrol.wait the CTA builds
    y[t][k] = bf16(nw[k] * sum_s c[t][s] * yb[t * 8 + s][k]),   c[t][s] = tw[t][s] * rsqrt(ss[t * 8 + s] / 4096 + eps)
for every local slot s of token t (VMS's combine, from bf16 per-route rows = SGLang's expert-output rounding point) into
shared memory, then runs today's mma loop with its x fragments read from shared memory. Each thread stages (cp.async 16 B)
and later reads only its own 16 B of every route row, so staging needs no CTA barrier; route rows are staged in rounds of
whole tokens (<= XRC rows). The local-slot mask comes from the router ids read BEFORE the wait (tail2's output, complete
before VMS triggered); it is re-checked after the wait from the same loads as ss / tw and a mismatch restages (never taken
unless the hint was stale). ss is VMS-CF's double buffer, parity from VMS-CF's arrival counter ctr[1] (= (gen + 1) * G).
GD_XMK (bench builds only): per-CTA %globaltimer marks entry / wait / x ready / end in A.marks[cta * 4 + i].

VMS (VMS_CUDA_SRC): VMS_CF (default 0). VMS_CF = 1 drops the end grid barrier and the combine: each phase-B epilogue
stores bf16(acc * gs) (the per-route expert output, SGLang's rounding point) at row t * 8 + s (S.ent) and red.adds
sum((bf16 * nw)^2) into ss[gen & 1][t * 8 + s]; CTAs arrive on ctr[1] and exit. gen = ctr[1] / G read at entry (only this
launch's own arrivals can precede the read, < G of them); CTA 0 zeroes the next launch's ss half and doneA bank
(ctr[8 + 64 * ((gen + 1) & 1) ..]) after its griddepcontrol.wait. The CF build needs its own scratch (ctr >= 136 ints,
ss 256 floats, yb bf16 [64, 4096]).
"""

GD = [
    # Args: the CF inputs
    ("  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n};\n",
     "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
     "  const uint16_t* yb = nullptr; const float* xss = nullptr; const float* xtw = nullptr; const int* xids = nullptr;   // CF\n"
     "  const float* xnw = nullptr; const int* xctr = nullptr; float xeps = 0.f; int xlo = 0; int xvg = 1;\n};\n"),
    ("__device__ __forceinline__ uint32_t lde(const uint32_t* p) {\n",
     "#ifndef GD_XRC\n#define GD_XRC 16   // CF: route rows (8 KB) staged per round\n#endif\n"
     "__device__ __forceinline__ uint4 lds4(const uint4* p) {   // CF: x fragment from shared memory\n"
     "  uint4 r;\n"
     "  asm volatile(\"ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\\n\" : \"=r\"(r.x), \"=r\"(r.y), \"=r\"(r.z), \"=r\"(r.w)\n"
     "               : \"r\"((uint32_t)__cvta_generic_to_shared(p)));\n"
     "  return r;\n"
     "}\n"
     "__device__ __forceinline__ uint32_t lde(const uint32_t* p) {\n"),
    ("template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n",
     "#define GD_LDX(P_) (XC ? lds4(P_) : ldx(P_))\n"
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int XC = 0>\n"
     "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {\n"),
    ("  constexpr int XB = XP ? 2 : 1;\n",
     "  constexpr int XB = XP ? 2 : 1;\n"
     "  constexpr int XROW = 4096 + 32, XRC = GD_XRC;   // CF: smem y row stride (bf16, +64 B), route rows staged per round\n"
     "  static_assert(XRC >= 8, \"CF: a round must fit a whole token (<= 8 routes)\");\n"
     "  static_assert(XC == 0 || (NW * 32 == 512 && MT == 1 && S == 1 && CAT == 0 && VF == 0 && XP == 0),\n"
     "                \"CF: the T <= 8 unpooler route (16 warps x 8 K per thread = 4096)\");\n"),
    ("  if (A.trig) pdl_trigger();\n  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;\n",
     "  if (A.trig) pdl_trigger();\n"
     "#ifdef GD_XMK\n"
     "  if (A.marks && threadIdx.x == 0) A.marks[blockIdx.x * 4 + 0] = vf_gt();\n"
     "#endif\n"
     "  const int lane = (int)threadIdx.x & 31, warp = (int)threadIdx.x >> 5;\n"),
    # pre-wait: nw chunk (constant) + local-slot mask hint from the router ids
    ("  pdl_wait();\n  if constexpr (VF >= 1) {\n",
     "  float xnw8[8];\n"
     "  unsigned long long xmask = 0ull;\n"
     "  if constexpr (XC) {\n"
     "    const float4* nwp = reinterpret_cast<const float4*>(A.xnw) + 2 * (int)threadIdx.x;\n"
     "    const float4 n0 = __ldg(nwp), n1 = __ldg(nwp + 1);\n"
     "    xnw8[0] = n0.x; xnw8[1] = n0.y; xnw8[2] = n0.z; xnw8[3] = n0.w;\n"
     "    xnw8[4] = n1.x; xnw8[5] = n1.y; xnw8[6] = n1.z; xnw8[7] = n1.w;\n"
     "    const int m8 = A.M * 8;\n"
     "    const bool l0 = lane < m8 && (((__ldcg(A.xids + lane) - A.xlo) & 255) < 32);\n"
     "    const bool l1 = lane + 32 < m8 && (((__ldcg(A.xids + lane + 32) - A.xlo) & 255) < 32);\n"
     "    xmask = (unsigned long long)__ballot_sync(0xffffffffu, l0) | ((unsigned long long)__ballot_sync(0xffffffffu, l1) << 32);\n"
     "  }\n"
     "  pdl_wait();\n"
     "#ifdef GD_XMK\n"
     "  if (A.marks && threadIdx.x == 0) A.marks[blockIdx.x * 4 + 1] = vf_gt();\n"
     "#endif\n"
     "  if constexpr (VF >= 1) {\n"),
    # post-wait: build y in smem (CF), then the x fragments come from smem
    ("  const uint4* xb = reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n",
     "  uint16_t* xys = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(gd_vf_dsm) + 1024);\n"
     "  if constexpr (XC) {\n"
     "    unsigned char* xsm = reinterpret_cast<unsigned char*>(gd_vf_dsm);\n"
     "    float* xcs = reinterpret_cast<float*>(xsm);\n"
     "    int* xloc = reinterpret_cast<int*>(xsm + 256);\n"
     "    unsigned char* xst = xsm + 1024 + (size_t)A.M * XROW * 2;\n"
     "    const int tid = (int)threadIdx.x, m8 = A.M * 8, M = A.M;\n"
     "    auto rend = [&](unsigned long long mk, int ta) -> int {\n"
     "      int tb = ta, nr = 0;\n"
     "      while (tb < M) {\n"
     "        const int n = __popcll((mk >> (8 * tb)) & 255ull);\n"
     "        if (nr + n > XRC) break;\n"
     "        nr += n;\n"
     "        ++tb;\n"
     "      }\n"
     "      return tb;\n"
     "    };\n"
     "    auto stage = [&](unsigned long long mk, int ta, int tb) {\n"
     "      int e = 0;\n"
     "      for (int tk = ta; tk < tb; ++tk) {\n"
     "        unsigned bits = (unsigned)((mk >> (8 * tk)) & 255ull);\n"
     "        while (bits) {\n"
     "          const int s = __ffs(bits) - 1;\n"
     "          bits &= bits - 1;\n"
     "          vf_cp16(xst + (size_t)e * 8192 + 16 * tid, A.yb + (size_t)(tk * 8 + s) * 4096 + 8 * tid);\n"
     "          ++e;\n"
     "        }\n"
     "      }\n"
     "      asm volatile(\"cp.async.commit_group;\\n\" ::: \"memory\");\n"
     "    };\n"
     "    unsigned long long mk = xmask;\n"
     "    int ta = 0, tb = rend(mk, 0);\n"
     "    stage(mk, 0, tb);\n"
     "    if (tid < m8) {\n"
     "      const int id = __ldcg(A.xids + tid), c1 = __ldcg(A.xctr + 1);\n"
     "      const float s0 = __ldcg(A.xss + tid), s1 = __ldcg(A.xss + 128 + tid), tw = __ldcg(A.xtw + tid);\n"
     "      const bool loc = ((id - A.xlo) & 255) < 32;\n"
     "      const float ssv = ((c1 / A.xvg - 1) & 1) ? s1 : s0;\n"
     "      xcs[tid] = loc ? tw * rsqrtf(ssv * (1.f / 4096) + A.xeps) : 0.f;\n"
     "      xloc[tid] = loc ? 1 : 0;\n"
     "    }\n"
     "    __syncthreads();\n"
     "    {\n"
     "      const bool v0 = lane < m8 && xloc[lane] != 0, v1 = lane + 32 < m8 && xloc[lane + 32] != 0;\n"
     "      const unsigned long long vm = (unsigned long long)__ballot_sync(0xffffffffu, v0) |\n"
     "                                    ((unsigned long long)__ballot_sync(0xffffffffu, v1) << 32);\n"
     "      if (vm != mk) {   // stale hint: restage from the verified ids (uniform: every warp sees the same smem)\n"
     "        asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\");\n"
     "        mk = vm;\n"
     "        tb = rend(mk, 0);\n"
     "        stage(mk, 0, tb);\n"
     "      }\n"
     "    }\n"
     "    for (;;) {\n"
     "      asm volatile(\"cp.async.wait_all;\\n\" ::: \"memory\");\n"
     "      int e = 0;\n"
     "      for (int tk = ta; tk < tb; ++tk) {\n"
     "        float a[8];\n"
     "#pragma unroll\n"
     "        for (int i = 0; i < 8; ++i) a[i] = 0.f;\n"
     "        unsigned bits = (unsigned)((mk >> (8 * tk)) & 255ull);\n"
     "        while (bits) {\n"
     "          const int s = __ffs(bits) - 1;\n"
     "          bits &= bits - 1;\n"
     "          const float c = xcs[tk * 8 + s];\n"
     "          const uint4 v = *reinterpret_cast<const uint4*>(xst + (size_t)e * 8192 + 16 * tid);\n"
     "          const uint32_t vw[4] = {v.x, v.y, v.z, v.w};\n"
     "#pragma unroll\n"
     "          for (int i = 0; i < 4; ++i) {\n"
     "            a[2 * i] += c * __uint_as_float(vw[i] << 16);\n"
     "            a[2 * i + 1] += c * __uint_as_float(vw[i] & 0xffff0000u);\n"
     "          }\n"
     "          ++e;\n"
     "        }\n"
     "        uint32_t pw[4];\n"
     "#pragma unroll\n"
     "        for (int i = 0; i < 4; ++i) {\n"
     "          __nv_bfloat162 h = __floats2bfloat162_rn(a[2 * i] * xnw8[2 * i], a[2 * i + 1] * xnw8[2 * i + 1]);\n"
     "          pw[i] = *reinterpret_cast<uint32_t*>(&h);\n"
     "        }\n"
     "        *reinterpret_cast<uint4*>(xys + (size_t)tk * XROW + 8 * tid) = make_uint4(pw[0], pw[1], pw[2], pw[3]);\n"
     "      }\n"
     "      if (tb >= M) break;\n"
     "      ta = tb;\n"
     "      tb = rend(mk, ta);\n"
     "      stage(mk, ta, tb);\n"
     "    }\n"
     "    __syncthreads();\n"
     "  }\n"
     "  const uint4* xb = XC ? reinterpret_cast<const uint4*>(xys + (size_t)u0 * 64 + 16 * t)\n"
     "                      : reinterpret_cast<const uint4*>(A.x + (size_t)u0 * 64 + 16 * t);\n"),
    ("    xo[mt] = tok * (A.K >> 3);\n", "    xo[mt] = tok * (XC ? (XROW >> 3) : (A.K >> 3));\n"),
    ("    X[0][mt][0] = ldx(xb + xo[mt]);\n    X[0][mt][1] = ldx(xb + xo[mt] + 1);\n  }\n",
     "    X[0][mt][0] = GD_LDX(xb + xo[mt]);\n    X[0][mt][1] = GD_LDX(xb + xo[mt] + 1);\n  }\n"
     "#ifdef GD_XMK\n"
     "  if (A.marks && threadIdx.x == 0) A.marks[blockIdx.x * 4 + 2] = vf_gt();\n"
     "#endif\n"),
    ("        X[0][mt][0] = ldx(GC_XP(s) + xo[mt]);\n        X[0][mt][1] = ldx(GC_XP(s) + xo[mt] + 1);\n",
     "        X[0][mt][0] = GD_LDX(GC_XP(s) + xo[mt]);\n        X[0][mt][1] = GD_LDX(GC_XP(s) + xo[mt] + 1);\n"),
    ("    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n",
     "    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
     "#ifdef GD_XMK\n"
     "    if (A.marks) {\n"
     "      __syncthreads();\n"
     "      if (threadIdx.x == 0) A.marks[blockIdx.x * 4 + 3] = vf_gt();\n"
     "    }\n"
     "#endif\n"),
    # CF launcher (dynamic smem) + entry point gdx_run_raw / gdx_info_raw (instances @XINST@ / @XINFO@)
    ("int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,\n",
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>\n"
     "static int launch_xc(const Args& A, int nbx, int pdl, cudaStream_t st) {\n"
     "  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, 0, 1>;\n"
     "  static int attr = 0;   // opt-in size set so far (host side, outside graph capture after the first call per size)\n"
     "  const int ds = 1024 + A.M * 4128 * 2 + (8 * A.M < GD_XRC ? 8 * A.M : GD_XRC) * 8192;\n"
     "  if (ds > attr) {\n"
     "    const int e = (int)cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, ds);\n"
     "    if (e) return e;\n"
     "    attr = ds;\n"
     "  }\n"
     "  cudaLaunchConfig_t cfg = {};\n"
     "  cfg.gridDim = dim3((unsigned)nbx, 1, 1);\n"
     "  cfg.blockDim = dim3((unsigned)(WR * WK * 32), 1, 1);\n"
     "  cfg.dynamicSmemBytes = (size_t)ds;\n"
     "  cfg.stream = st;\n"
     "  cudaLaunchAttribute attr1[1];\n"
     "  unsigned na = 0;\n"
     "  if (pdl & 1) {\n"
     "    attr1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
     "    attr1[0].val.programmaticStreamSerializationAllowed = 1;\n"
     "    na = 1;\n"
     "  }\n"
     "  cfg.attrs = attr1;\n"
     "  cfg.numAttrs = na;\n"
     "  return (int)cudaLaunchKernelEx(&cfg, fn, A);\n"
     "}\n"
     "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP>\n"
     "static int info_xc(int64_t* res) {\n"
     "  cudaFuncAttributes fa;\n"
     "  auto fn = gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, 0, 1>;\n"
     "  int e = (int)cudaFuncGetAttributes(&fa, fn);\n"
     "  if (e) return e;\n"
     "  res[0] = fa.numRegs; res[1] = (int64_t)fa.localSizeBytes; res[2] = 1; res[3] = (int64_t)fa.sharedSizeBytes;\n"
     "  return 0;\n"
     "}\n"
     "#define GD_XINST(ID, ...) \\\n"
     "  case ID: return launch_xc<__VA_ARGS__>(A, nbx, pdl, st);\n"
     "#define GD_XINFO(ID, ...) \\\n"
     "  case ID: return info_xc<__VA_ARGS__>(res);\n"
     "int gdx_run_raw(int64_t inst, const void* w, const void* esc, const void* yb, const float* ss, const float* tw, const int* ids,\n"
     "                const float* nw, const int* ctr, void* out, int M, int K, int ldo, uint32_t tb, int nbx, int pdl, float eps,\n"
     "                int lo, int vg, long long* marks, cudaStream_t st) {\n"
     "  if (M < 1 || M > 8 || K != 4096) return -2;\n"
     "  Args A;\n"
     "  A.w = reinterpret_cast<const uint4*>(w);\n"
     "  A.esc = reinterpret_cast<const uint32_t*>(esc);\n"
     "  A.x = nullptr;\n"
     "  A.out = out;\n"
     "  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;\n"
     "  A.trig = (pdl >> 1) & 1;\n"
     "  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = marks;\n"
     "  A.yb = reinterpret_cast<const uint16_t*>(yb); A.xss = ss; A.xtw = tw; A.xids = ids; A.xnw = nw; A.xctr = ctr;\n"
     "  A.xeps = eps; A.xlo = lo; A.xvg = vg;\n"
     "  switch (inst) {\n"
     "@XINST@\n"
     "    default: return -1;\n"
     "  }\n"
     "}\n"
     "int gdx_info_raw(int64_t inst, int64_t* res) {\n"
     "  switch (inst) {\n"
     "@XINFO@\n"
     "    default: return -1;\n"
     "  }\n"
     "}\n\n"
     "int gdc_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, void* out2, int M, int K, int ldo,\n"),
]

# GD_CPP_SRC: the CF entry points
GD_CPP = [
    ("int gd_info_raw(int64_t inst, int64_t* res);\n",
     "int gd_info_raw(int64_t inst, int64_t* res);\n"
     "int gdx_run_raw(int64_t inst, const void* w, const void* esc, const void* yb, const float* ss, const float* tw, const int* ids,\n"
     "                const float* nw, const int* ctr, void* out, int M, int K, int ldo, uint32_t tb, int nbx, int pdl, float eps,\n"
     "                int lo, int vg, long long* marks, cudaStream_t st);\n"
     "int gdx_info_raw(int64_t inst, int64_t* res);\n"
     "void gdx_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor yb, torch::Tensor ss, torch::Tensor tw,\n"
     "             torch::Tensor ids, torch::Tensor nw, torch::Tensor ctr, torch::Tensor out, int64_t M, int64_t ldo, int64_t tb,\n"
     "             int64_t nbx, int64_t pdl, double eps, int64_t lo, int64_t vg, torch::Tensor marks) {\n"
     "  TORCH_CHECK(yb.scalar_type() == at::kBFloat16 && yb.is_contiguous() && yb.size(1) == 4096 && yb.size(0) >= 8 * M &&\n"
     "              ss.scalar_type() == at::kFloat && ss.numel() >= 256 && tw.scalar_type() == at::kFloat && tw.numel() >= 8 * M &&\n"
     "              ids.scalar_type() == at::kInt && ids.numel() >= 8 * M && nw.scalar_type() == at::kFloat && nw.numel() == 4096 &&\n"
     "              ctr.scalar_type() == at::kInt && out.scalar_type() == at::kBFloat16, \"gdx_run: CF inputs\");\n"
     "  long long* mk = marks.numel() >= 4 * nbx ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;\n"
     "  const int e = gdx_run_raw(inst, w.data_ptr(), esc.data_ptr(), yb.data_ptr(), ss.data_ptr<float>(), tw.data_ptr<float>(),\n"
     "                            ids.data_ptr<int>(), nw.data_ptr<float>(), ctr.data_ptr<int>(), out.data_ptr(), (int)M, 4096,\n"
     "                            (int)ldo, (uint32_t)tb, (int)nbx, (int)pdl, (float)eps, (int)lo, (int)vg, mk,\n"
     "                            at::cuda::getCurrentCUDAStream());\n"
     "  TORCH_CHECK(e == 0, \"gdx_run: inst \", inst, \" error \", e);\n"
     "}\n"
     "std::vector<int64_t> gdx_info(int64_t inst) {\n"
     "  std::vector<int64_t> r(4, 0);\n"
     "  const int e = gdx_info_raw(inst, r.data());\n"
     "  TORCH_CHECK(e == 0, \"gdx_info: inst \", inst, \" error \", e);\n"
     "  return r;\n"
     "}\n"),
]


def _apply(src, pats, what):
    for old, new in pats:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"[cf_patches] {what}: anchor matched {n}x: {old[:90]!r}")
        src = src.replace(old, new)
    return src


def apply_gd(src):
    return _apply(src, GD, "GD")


def apply_gd_cpp(src):
    return _apply(src, GD_CPP, "GD_CPP")


def gd_instances(keys, xkeys):
    """@INST@ / @INFO@ / @XINST@ / @XINFO@ fill for the patched GD source."""
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    xinst = "\n".join("    GD_XINST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(xkeys))
    xinfo = "\n".join("    GD_XINFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(xkeys))
    return {"@INST@": inst, "@INFO@": info, "@XINST@": xinst, "@XINFO@": xinfo}


VMS = [
    ("#ifndef VMS_G\n",
     "#ifndef VMS_CF\n#define VMS_CF 0   // COMB-FOLD (tools/cf_patches.py): no end barrier / combine; the GD unpooler combines\n#endif\n"
     "#if VMS_CF\n#define VMS_DA(j_) (p.ctr + 8 + 64 * (S.gen & 1) + (j_))   // doneA bank of this launch\n"
     "#else\n#define VMS_DA(j_) (p.ctr + 8 + (j_))\n#endif\n"
     "#ifndef VMS_G\n"),
    ("if (tid == 0) S.gen = ld_rlx(p.ctr);",
     "if (tid == 0) S.gen = VMS_CF ? ld_rlx(p.ctr + 1) / G : ld_rlx(p.ctr);"),
    ("  const int njobs = S.njobs;\n",
     "#if VMS_CF\n"
     "  if (c == 0) {   // the next launch's doneA bank and ss half (last used by the previous launch, complete)\n"
     "    const int nb = (S.gen + 1) & 1;\n"
     "    if (tid < MAXJ) p.ctr[8 + 64 * nb + tid] = 0;\n"
     "    if (tid < MAXP) p.ss[nb * MAXP + tid] = 0.f;\n"
     "  }\n"
     "#endif\n"
     "  const int njobs = S.njobs;\n"),
    ("        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;\n"
     "#pragma unroll\n"
     "        for (int h = 0; h < 2; ++h) {\n"
     "          const int tk = 2 * t + h;\n"
     "          const float y0 = (h ? sm.y : sm.x) * w0, y1 = (h ? sm.w : sm.z) * w1;\n"
     "          float sq = y0 * y0 + y1 * y1;\n"
     "          if (tk < cnt) {\n"
     "            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;\n"
     "            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;\n"
     "          }\n",
     "#if VMS_CF\n"
     "        const float w0 = __ldg(p.nw + n0), w1 = __ldg(p.nw + n0 + 8);\n"
     "        __nv_bfloat16* yb16 = reinterpret_cast<__nv_bfloat16*>(p.ybuf);\n"
     "#pragma unroll\n"
     "        for (int h = 0; h < 2; ++h) {\n"
     "          const int tk = 2 * t + h;\n"
     "          const __nv_bfloat16 e0 = __float2bfloat16_rn((h ? sm.y : sm.x) * gs), e1 = __float2bfloat16_rn((h ? sm.w : sm.z) * gs);\n"
     "          const float y0 = __bfloat162float(e0) * w0, y1 = __bfloat162float(e1) * w1;\n"
     "          float sq = y0 * y0 + y1 * y1;\n"
     "          const int er = tk < cnt ? S.ent[st + tk] : 0;   // slot row t * 8 + s\n"
     "          if (tk < cnt) {\n"
     "            yb16[(size_t)er * HD + n0] = e0;\n"
     "            yb16[(size_t)er * HD + n0 + 8] = e1;\n"
     "          }\n"
     "#else\n"
     "        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;\n"
     "#pragma unroll\n"
     "        for (int h = 0; h < 2; ++h) {\n"
     "          const int tk = 2 * t + h;\n"
     "          const float y0 = (h ? sm.y : sm.x) * w0, y1 = (h ? sm.w : sm.z) * w1;\n"
     "          float sq = y0 * y0 + y1 * y1;\n"
     "          const int er = st + tk;\n"
     "          if (tk < cnt) {\n"
     "            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;\n"
     "            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;\n"
     "          }\n"
     "#endif\n"),
    ("          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);\n",
     "          if (g == 0 && tk < cnt) atomicAdd(ssc + er, sq);\n"),
    ("#if VMS_FX & 2\n    asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n#else\n    __threadfence();\n#endif\n"
     "    if (mk && lane == 0) mrow[7] = gt();\n",
     "#if !VMS_CF   // CF: kernel completion orders the stores for the unpooler (no in-kernel reader)\n"
     "#if VMS_FX & 2\n    asm volatile(\"fence.acq_rel.gpu;\\n\" ::: \"memory\");\n#else\n    __threadfence();\n#endif\n"
     "#endif\n"
     "    if (mk && lane == 0) mrow[7] = gt();\n"),
    ("  const int gen = S.gen;\n",
     "  const int gen = S.gen;\n"
     "#if VMS_CF\n"
     "  if (tid == 0) {\n"
     "    atomicAdd(p.ctr + 1, 1);   // arrival: gen = ctr[1] / G for the next launch and the unpooler's ss parity\n"
     "    if (mk) { mrow[8] = gt(); mrow[9] = mrow[8]; }\n"
     "#if VMS_MK2\n"
     "    if (mk) mrow[17] = clock64();\n"
     "#endif\n"
     "  }\n"
     "  return;\n"
     "#endif\n"),
]
VMS_SUBS = [("p.ctr + 8 + jj", "VMS_DA(jj)", 3), ("atomicAdd(p.ctr + 8 + j,", "atomicAdd(VMS_DA(j),", 1)]


def apply_vms(src):
    src = _apply(src, VMS, "VMS")
    for old, new, n in VMS_SUBS:
        if src.count(old) != n:
            raise SystemExit(f"[cf_patches] VMS sub {old!r}: {src.count(old)} != {n}")
        src = src.replace(old, new)
    return src

"""vms_abs.py: VMS-ABS = VMS absorbs tail_kernel2 at T <= 8 (notes/study/vm_small.md s6; parent lever from vf_front.md s9).

Kernel (text edits on the frozen v1 VMS source, compile-time VMS_ABS = 1, a separate build; VMS_ABS = 0 preprocesses back to v1):
  * before griddepcontrol.wait: warps t < T each replay tail_kernel2's routing warp for token t verbatim (router32 FP32
    partial slices summed in slice order with zeros past ns, sigmoid via expf, + bias, top-8 by orderable key with the
    lowest index on ties, weights s / (sum + deps) * rscale, local id from local_start) into smem; warp 0's job table reads
    them; the first NS weight copies are issued. Legal: the router32 partials come from the side stream, joined by a full
    graph edge (only the pooler is the PDL primary).
  * after griddepcontrol.wait: every CTA computes each token's rstd in tail_kernel2's exact order (virtual thread 32 w' + l
    sums its 8 squares in order, xor tree 16..1, 16 partials in order; rsqrtf(tot / 4096 + eps)), writes its 1/G slice of
    xn = bf16((v * rstd) * w) to a scratch buffer, fence + counter; compute warps poll the counter before the loop.
  * the only compute-loop change: phase-A x fragments load with ld.cg (xn is written in this launch); final pass reads tw
    from smem.
Layer (apply_layer on a stripped layer file, default k3_slim_3): run_abs binding, PT1 build "vmsabs", _vms_abs_prepare
(self-test vs tail_kernel2 + VMS v1: ids / weights / xn bitwise, output rel < 1e-4 (VMS's FP32 sum-of-squares atomics
are not bitwise run to run), incl. near-tie top-8 cases, two calls each, BF16 and FP32 pooler outputs;
loud fallback), forward: T <= VMS_ABS_MAXT with the side-stream router (3-D logits) -> one VMS-ABS launch instead of
tail_kernel2 + VMS. VMS_ABS_ENABLE is read at call time (A/B arms).

    python -I tools/vms_abs.py [--n=5] [--base=k3_slim_3] [--check]      # -> kernels/muse_decode_layer/k3_vms_<n>.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)
Q3 = '"' * 3

ROUTE = r'''#if VMS_ABS
  if (warp < p.m) {   // tail_kernel2's routing warp for token t = warp (same arithmetic, same order)
    const int t = warp;
    const float4* pbias = reinterpret_cast<const float4*>(p.rbias) + lane * 2;
    const float4 c0 = __ldg(pbias), c1 = __ldg(pbias + 1);
    float4 qa[8][2];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
      if (q < p.ns) {
        const float4* pq = reinterpret_cast<const float4*>(p.lg + q * p.sstride + (long)t * p.l_ld) + lane * 2;
        qa[q][0] = __ldcg(pq);
        qa[q][1] = __ldcg(pq + 1);
      } else {
        qa[q][0] = make_float4(0.f, 0.f, 0.f, 0.f);
        qa[q][1] = qa[q][0];
      }
    }
    float4 a0 = qa[0][0], a1 = qa[0][1];
    const int nsm = p.ns > 1 ? 8 : 1;
    for (int q = 1; q < nsm; ++q) {
      a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
      a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
    }
    a0.x += 0.f; a0.y += 0.f; a0.z += 0.f; a0.w += 0.f; a1.x += 0.f; a1.y += 0.f; a1.z += 0.f; a1.w += 0.f;
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    float s[8], b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * p.mult));
      b[j] = s[j] + bs[j];
    }
    int sel_id[8];
    float sel_s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bv = b[0], sv = s[0];
      int bj = 0;
#pragma unroll
      for (int j = 1; j < 8; ++j)
        if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
      const unsigned u = __float_as_uint(bv + 0.f);
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
      for (int i = 0; i < 8; ++i) sum += sel_s[i];
      const float inv = p.rscale / (sum + p.deps);
      const int ls = __ldg(p.lstart);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        S.twv[t * 8 + k] = sel_s[k] * inv;
        const int rel = ((sel_id[k] - ls) % 256 + 256) % 256;
        S.lidv[t * 8 + k] = rel < 32 ? rel : -1;
      }
    }
  }
  __syncthreads();
  if (c == 0 && tid < p.m * 8) {   // self-test view: ids / weights in xnb rows 8.. (unused at T <= 8)
    reinterpret_cast<int*>(p.xnb + 8 * 4096)[tid] = S.lidv[tid];
    reinterpret_cast<float*>(p.xnb + 8 * 4096)[128 + tid] = S.twv[tid];
  }
#elif __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  if (warp == 0) {
'''

NORM = r'''#if VMS_ABS
    {   // xn = tail_kernel2's pre-expert norm, every CTA computes rstd, writes a 1/G slice, then all wait for every slice
#if __CUDA_ARCH__ >= 900
      asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
      for (int vw = warp; vw < p.m * 16; vw += NW) {
        const int tk = vw >> 4, wq = vw & 15;
        float v[8];
        vms_y8(p, tk, (32 * wq + lane) * 8, v);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        if (lane == 0) S.redn[tk][wq] = ss;
      }
      cta_sync_compute();
      if (tid < p.m) {
        float tot = 0.f;
        for (int i = 0; i < 16; ++i) tot += S.redn[tid][i];
        S.rst[tid] = rsqrtf(tot / 4096.f + p.eps_n);
      }
      cta_sync_compute();
      const int n16 = p.m * 512, i0 = c * n16 / G, i1 = (c + 1) * n16 / G;
      for (int i = i0 + tid; i < i1; i += NT) {
        const int tk = i >> 9;
        float v[8];
        vms_y8(p, tk, (i & 511) * 8, v);
        const uint4 wq4 = __ldg(reinterpret_cast<const uint4*>(p.nwx) + (i & 511));
        const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq4);
        __align__(16) __nv_bfloat16 o8[8];
        const float rs = S.rst[tk];
#pragma unroll
        for (int k = 0; k < 8; ++k) o8[k] = __float2bfloat16((v[k] * rs) * __bfloat162float(w[k]));
        reinterpret_cast<uint4*>(p.xnb)[i] = *reinterpret_cast<const uint4*>(o8);
      }
      __threadfence();
      cta_sync_compute();
      if (tid == 0) atomicAdd(p.ctr + 3, 1);
      if (lane == 0) {
        const int target = (S.xgen + 1) * G;
        int n = 0;
        while (ld_acq(p.ctr + 3) < target) {
          if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 3); break; }
          __nanosleep(32);
        }
      }
      __syncwarp();
      (void)ld_acq(p.ctr + 3);
    }
#endif
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);'''

Y8 = r'''#if VMS_ABS
__device__ __forceinline__ void vms_y8(const Params& p, int tk, int e0, float* v) {
  if (p.y32) {
    const float4* q = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(p.yin) + (long)tk * p.y_ld + e0);
    const float4 a = __ldcg(q), b = __ldcg(q + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = __ldcg(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(p.yin) + (long)tk * p.y_ld + e0));
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
}
#define VMS_TW(i) (S.twv[(i)])
#else
#define VMS_TW(i) __ldg(p.tw + (i))
#endif
'''

RUN_ABS = r'''
void run_abs(torch::Tensor xnb, torch::Tensor y, torch::Tensor lg, torch::Tensor rbias, torch::Tensor lstart, torch::Tensor nwx,
             torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter,
             torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr, torch::Tensor out, torch::Tensor marks, int64_t m,
             double eps, double eps_n, double mult, double rscale, double deps, int64_t pdl) {
  TORCH_CHECK(m >= 1 && m <= 8 && lg.dim() == 3 && lg.size(0) <= 8 && lg.stride(2) == 1 && y.stride(1) == 1, "vms abs: shapes");
  TORCH_CHECK(lg.scalar_type() == at::kFloat && rbias.scalar_type() == at::kFloat && lstart.scalar_type() == at::kInt, "vms abs: dtypes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(lg.data_ptr()) % 16 == 0
              && lg.stride(1) % 4 == 0 && lg.stride(0) % 4 == 0 && y.stride(0) % 8 == 0, "vms abs: alignment");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vms::vms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vms::SMEM_BYTES));
    attr = true;
  }
  vms::Params p = {};
  p.x = reinterpret_cast<const __nv_bfloat16*>(xnb.data_ptr()); p.ids = nullptr; p.tw = nullptr;
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = 0;
  p.lg = lg.data_ptr<float>(); p.rbias = rbias.data_ptr<float>(); p.lstart = lstart.data_ptr<int>(); p.yin = y.data_ptr();
  p.nwx = reinterpret_cast<const __nv_bfloat16*>(nwx.data_ptr()); p.xnb = reinterpret_cast<__nv_bfloat16*>(xnb.data_ptr());
  p.l_ld = lg.stride(1); p.sstride = lg.stride(0); p.y_ld = y.stride(0); p.ns = (int)lg.size(0);
  p.y32 = y.scalar_type() == at::kFloat ? 1 : 0;
  p.mult = (float)mult; p.rscale = (float)rscale; p.deps = (float)deps; p.eps_n = (float)eps_n;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vms::G); lc.blockDim = dim3(vms::NTL); lc.dynamicSmemBytes = vms::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vms::vms_kernel, p));
}
'''
RUN_ABS_DECL = ('void run_abs(torch::Tensor xnb, torch::Tensor y, torch::Tensor lg, torch::Tensor rbias, torch::Tensor lstart, '
                'torch::Tensor nwx, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw, '
                'torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr, torch::Tensor out, '
                'torch::Tensor marks, int64_t m, double eps, double eps_n, double mult, double rscale, double deps, int64_t pdl);\n')

CUDA_EDITS = (
    ("abs define", "#ifndef VMS_G\n", "#ifndef VMS_ABS\n#define VMS_ABS 0\n#endif\n#ifndef VMS_G\n"),
    ("params", "  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};",
     "  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n"
     "  const float* lg; const float* rbias; const int* lstart; const void* yin; const __nv_bfloat16* nwx; __nv_bfloat16* xnb;\n"
     "  long l_ld, sstride, y_ld; int ns, y32; float mult, rscale, deps, eps_n;\n};"),
    ("smem", "  float4 red[NRB][NW][32];\n};",
     "  float4 red[NRB][NW][32];\n#if VMS_ABS\n  float twv[MAXP]; int lidv[MAXP]; float rst[16]; float redn[8][16]; int xgen;\n#endif\n};"),
    ("y8 helper", "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {",
     Y8 + "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {"),
    ("route", '#if __CUDA_ARCH__ >= 900\n  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n#endif\n  if (warp == 0) {\n', ROUTE),
    ("ids", "    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }",
     "#if VMS_ABS\n    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? (S.lidv[i] & 255) : 255; }\n#else\n"
     "    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }\n#endif"),
    ("xgen", "    if (tid == 0) S.gen = ld_rlx(p.ctr);\n",
     "    if (tid == 0) S.gen = ld_rlx(p.ctr);\n#if VMS_ABS\n    if (tid == 0) S.xgen = ld_rlx(p.ctr + 4);\n#endif\n"),
    ("norm", "    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);", NORM),
    ("x cg", "            xr[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);\n"
             "            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);",
     "            xr[2 * cc] = (PB || VMS_ABS) ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);\n"
     "            xr[2 * cc + 1] = (PB || VMS_ABS) ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);"),
    ("final tw", "__ldg(p.tw + tk * 8 + s)", "VMS_TW(tk * 8 + s)"),
    ("abs count", "    if (tid == 0) atomicAdd(p.ctr, 1);\n",
     "    if (tid == 0) atomicAdd(p.ctr, 1);\n#if VMS_ABS\n    if (tid == 1) atomicAdd(p.ctr + 4, 1);\n#endif\n"),
    ("run_abs", "std::vector<int64_t> attrs() {", RUN_ABS + "\nstd::vector<int64_t> attrs() {"),
)
CPP_EDIT = ("std::vector<int64_t> attrs();", RUN_ABS_DECL + "std::vector<int64_t> attrs();")


def edit_sources(cuda, cpp):
    for what, old, new in CUDA_EDITS:
        n = cuda.count(old)
        assert n == 1, f"vms_abs cuda anchor {what!r} found {n} times"
        cuda = cuda.replace(old, new, 1)
    assert cpp.count(CPP_EDIT[0]) == 1
    cpp = cpp.replace(CPP_EDIT[0], CPP_EDIT[1], 1)
    return cuda, cpp


def lab_sources():
    t = (ROOT / "kernels/lab/vms_moe_lab_v1.py").read_text(encoding="utf-8")
    cuda = t.split("CUDA_SRC = r'''", 1)[1].split("'''", 1)[0]
    cuda = cuda.replace("  // ---- routing (warp 0, redundant per CTA): pairs sorted by expert, jobs of <= 8 pairs" + NL, "")
    cpp = t.split("CPP_SRC = r'''", 1)[1].split("'''", 1)[0]
    return edit_sources(cuda, cpp)


# ---------------------------------------------------------------- layer patch
PY_BLOCK = r'''
VMS_ABS_ENABLE = True
VMS_ABS_MAXT = 8


def _vms_abs_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_ABS=1", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_ABS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vmsabs_{tag}", cpp_sources=VMS_ABS_CPP_SRC, cuda_sources=VMS_ABS_CUDA_SRC, functions=["run_abs", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["Gabs"] = G
    _VMS["mod_abs"] = mod
    _K3_BUILD_S["vmsabs"] = time.time() - t1
    return mod


def _vms_abs_call(state, x, logits, data):
    vs = state["k2a33"]["st"]["vms"]
    v8 = state["v8"]
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod_abs"].run_abs(vs["xnb"], x, logits, v8["bias"], data["local_start"], v8["nw"], vs["wA"], vs["wB"], vs["gsA"],
                          vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"], out, vs["nomarks"], m,
                          K2A_POST_NORM_EPS, RMS_EPS, 0.1767766952966369, 1.0, 1e-15, 1)
    return out


def _vms_abs_live(state, tokens, place, logits):
    return (VMS_ABS_ENABLE and state.get("vms_abs_ok") and tokens <= VMS_ABS_MAXT and place != "main"
            and logits is not None and logits.dim() == 3 and logits.shape[0] <= 8 and logits.stride(2) == 1)


def _vms_abs_prepare(state):
    state["vms_abs_ok"] = False
    if not VMS_ABS_ENABLE:
        return
    t1 = time.time()
    try:
        k33 = state.get("k2a33")
        st = k33["st"] if k33 is not None else None
        v8 = state.get("v8")
        if st is None or not st.get("vms_ok") or v8 is None or not TL1_TAIL2:
            print(f"[vmsabs] OFF (vms_ok={st.get('vms_ok') if st else None}, tail={'yes' if v8 else 'no'}, TL1={TL1_TAIL2})",
                  flush=True)
            return
        _pt1_wait("vmsabs")
        mod = _VMS.get("mod_abs") if _VMS.get("mod_abs") is not None else _vms_abs_build()
        if _VMS.get("Gabs") != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vmsabs] built for a different SM count: OFF", flush=True)
            return
        dev = st["vms"]["wA"].device
        st["vms"]["mod_abs"] = mod
        st["vms"]["xnb"] = torch.zeros((16, K2A_D_HIDDEN), dtype=torch.bfloat16, device=dev)
        good = True
        for T, seed, kind, ydt in ((1, 71, "normal", torch.bfloat16), (1, 72, "normal", torch.float32), (5, 73, "tie", torch.bfloat16),
                                   (8, 74, "normal", torch.bfloat16), (8, 75, "tie", torch.float32), (2, 76, "normal", torch.float32)):
            g = torch.Generator(device=dev).manual_seed(seed)
            y = torch.randn((T, K2A_D_HIDDEN), generator=g, device=dev).to(ydt)
            lg = torch.randn((8, T, N_EXPERTS), generator=g, device=dev) * 0.5
            ls = int(torch.randint(0, 8, (1,), generator=g).item()) * 32
            if kind == "tie":   # exact ties at the top-8 boundary: identical logit sums and bias for a run of experts
                lg[:, :, ls:ls + 40] = 0.7
                v8b = v8["bias"].clone()
                v8["bias"][ls:ls + 40] = v8b[ls]
            data = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
            try:
                for k in range(2):
                    xn, tw, tid, lid = _v8_tail(y, lg, state, data)
                    ref = _vms_call(st, xn, lid, tw).clone()
                    st["vms"]["ctr"][2] = 0
                    o = _vms_abs_call(state, y, lg, data)
                    torch.cuda.synchronize()
                    errf = int(st["vms"]["ctr"][2])
                    xb = st["vms"]["xnb"]
                    xs = bool(torch.equal(xb[:T], xn))
                    dv = xb[8:].reshape(-1).view(torch.int32)
                    ids_ok = bool(torch.equal(dv[:T * 8], lid.reshape(-1)))
                    tw_ok = bool(torch.equal(dv[128:128 + T * 8].view(torch.float32), tw.reshape(-1)))
                    rel = float((o.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))
                    same = xs and ids_ok and tw_ok and rel < 1e-4 and errf == 0 and bool(torch.isfinite(o.float()).all())
                    if not same or k == 0:
                        print(f"[vmsabs] self-test T={T} {kind} y={str(ydt)[6:]} call {k}: xn {xs} ids {ids_ok} weights {tw_ok} "
                              f"out rel {rel:.1e} err={errf} -> {'ok' if same else 'FAIL'}", flush=True)
                    good &= same
            finally:
                if kind == "tie":
                    v8["bias"].copy_(v8b)
            _k2a33_reset(st)
        state["vms_abs_ok"] = good
        print(f"[vmsabs] VMS-ABS {'LIVE at T <= ' + str(VMS_ABS_MAXT) if good else 'OFF: self-test failed, tail_kernel2 + VMS'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        state["vms_abs_ok"] = False
        print(f"[vmsabs] prepare raised {ex!r}: OFF (tail_kernel2 + VMS)", flush=True)


'''
FWD_TAIL_OLD = "        if tail is not None:\n            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)\n"
FWD_TAIL_NEW = ("        if tail is not None and _vms_abs_live(state, tokens, place, logits):\n"
                "            abs_out = _vms_abs_call(state, expert_input, logits, data)\n"
                "        elif tail is not None:\n"
                "            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)\n")
FWD_EXP_OLD = "        expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)\n"
FWD_EXP_NEW = ("        expert_output = abs_out if abs_out is not None else _v8_experts(expert_x, local_ids, topk_weights, state, cfg,"
               " tokens)\n")
FWD_INIT_OLD = "    g2out = _g2_prt(ffn_input, router_input, state, data, tokens, mode, tail)\n"
FWD_INIT_NEW = "    abs_out = None\n" + FWD_INIT_OLD
PT1_OLD = '    if VMS_ENABLE:\n        jobs.append(("vms", _vms_build))\n'
PT1_NEW = PT1_OLD + '    if VMS_ENABLE and VMS_ABS_ENABLE:\n        jobs.append(("vmsabs", _vms_abs_build))\n'
PREP_OLD = "    _p206_prepare(state, t0)\n"
PREP_NEW = "    _p206_prepare(state, t0)\n    _vms_abs_prepare(state)\n"
ANCHOR = "def _v8_tail(y, logits, state, data):\n"


def _swap(text, name, new):
    head = f"{name} = r{Q3}"
    assert text.count(head) == 1, name
    i = text.index(head)
    j = text.index(Q3, i + len(head))
    return text[:i] + head + new + text[j:], text[i + len(head):j]


def _get(text, name):
    head = f"{name} = r{Q3}"
    assert text.count(head) == 1, name
    i = text.index(head) + len(head)
    return text[i:text.index(Q3, i)]


def apply_layer(text):
    """VMS_CUDA_SRC / VMS_CPP_SRC stay byte-identical (the shipped v1 module is untouched: the edits alone change its
    VMS_ABS=0 codegen, 1952 vs 2016 SASS instructions); VMS-ABS builds from edited copies VMS_ABS_CUDA_SRC / _CPP_SRC."""
    assert "run_abs" not in text and "VMS_ABS_ENABLE" not in text
    cuda, cpp = edit_sources(_get(text, "VMS_CUDA_SRC"), _get(text, "VMS_CPP_SRC"))
    block = "VMS_ABS_CUDA_SRC = r" + Q3 + cuda + Q3 + NL + "VMS_ABS_CPP_SRC = r" + Q3 + cpp + Q3 + NL
    assert text.count(ANCHOR) == 1
    text = text.replace(ANCHOR, block + ANCHOR, 1)
    for what, old, new in (("fwd tail", FWD_TAIL_OLD, FWD_TAIL_NEW), ("fwd experts", FWD_EXP_OLD, FWD_EXP_NEW),
                           ("fwd init", FWD_INIT_OLD, FWD_INIT_NEW), ("pt1", PT1_OLD, PT1_NEW), ("prepare", PREP_OLD, PREP_NEW),
                           ("py block", ANCHOR, PY_BLOCK + ANCHOR)):
        n = text.count(old)
        assert n == 1, f"vms_abs layer anchor {what!r} found {n} times"
        text = text.replace(old, new, 1)
    return text


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    n = int(args.get("n", 5))
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_slim_3')}.py"
    out = ROOT / f"kernels/muse_decode_layer/k3_vms_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vms_{n}_ab.py"
    tag = f"k3-vms-{n}"
    sys.path.insert(0, str(ROOT / "tools"))
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    rep = [f"base {base.name} {len(b.encode())} B is a finish() fixed point"]
    nonce = (f"# nonce {tag} ({base.stem} + VMS-ABS: VMS absorbs tail_kernel2 at T <= 8 (in-kernel top-8 + pre-expert norm, "
             "first weight copies before the PDL wait), bitwise self-test vs tail_kernel2 + VMS with fallback)")
    s = GM.finish(apply_layer(b.replace(bn, nonce)), nonce, tag, chk_base, rep)
    rep.append(f"size: {len(b.encode())} -> {len(s.encode())} B ({len(s.encode()) - len(b.encode()):+d} B, headroom "
               f"{1_000_000 - len(s.encode())} B)")
    for r in rep:
        print(f"[vms_abs] OK  {r}")
    if "--check" in sys.argv:
        GM.check(out.exists() and out.read_text(encoding="utf-8") == s, f"{out.name} on disk differs")
        return 0
    out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vms_abs] wrote {out.relative_to(ROOT)} and {lab.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

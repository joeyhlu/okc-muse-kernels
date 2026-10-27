"""gr3_patches.py: GR3b = the routing tail off the critical path at T <= 8, as lab patches on k3_v37 (R-glue-3).

notes/study/glue_fusion_independent.md s3 GR3, the cheap variant (no K2A edits). G2's pooler + FP32 router kernel
(prt_kernel) gets a sibling prt_kernel_b<MT> for T = 1 (MT 1) and T = 2..8 (MT 8):
  - pooler stream and router piece exactly as prt_kernel<MT, 1> (router weights in SMEM before the PDL wait);
  - no global pooler output and no last-CTA tail: each CTA keeps its 16 bf16 rows in registers, publishes their sum of squares
    per token, meets the other 255 CTAs at a sense-reversing grid barrier (all co-resident at 2 CTAs/SM; prepare checks it),
    then scales its own rows into xn (tail_kernel's formula, rstd from the 256 partials in a fixed order);
  - CTA 0 runs the top-8 routing, one warp per token, from the router partials (prt_kernel's / tail_kernel's code).
At T = 8 it replaces router32 (side stream) + P1 pooler + the fork/join + tail_kernel2: one kernel, one stream, PDL from SBB1.
K1 / K2 are untouched (they read xn / tid / lid as today). Every spin is bounded, sets err and never traps (R9).
GR3_ENABLE = False (the default) is k3_v37's behaviour.

    python -I tools/gr3_patches.py [in.py] [out.py]
    (default kernels/muse_decode_layer/k3_v37.py -> kernels/lab/k3_v37_gr3.py)
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_decode_layer/k3_v37.py"
DST = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "kernels/lab/k3_v37_gr3.py"
NONCE = ("# nonce k3-v37-gr3-lab (k3_v37 + GR3b: pooler + router kernel with an in-kernel barrier norm + CTA-0 routing at T <= 8, "
         "tools/gr3_patches.py; lab A/B file for R-glue-3, GR3_ENABLE off by default)")
G2_REG = ('G2_CUDA_SRC = r"""', '\n"""\n\n\ndef _g2_stream_h')


def sub(text, old, new, start=None, end=None, count=1):
    i = 0 if start is None else text.index(start)
    j = len(text) if end is None else text.index(end, i)
    seg = text[i:j]
    n = seg.count(old)
    assert n == count, f"anchor {old[:70]!r}: {n} hits (want {count})"
    return text[:i] + seg.replace(old, new) + text[j:]


def region(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


EPI_OLD = ("  {\n    const int row = rt * 16 + (tid & 15), tok = tid >> 4;\n    if (tid < 16 * m) {\n      float v = 0.f;\n"
           "      for (int q = 0; q < nw; ++q) v += red[q][(tid & 15) * 8 + tok];\n      v *= A.scale[rt];\n"
           "      A.out[(size_t)tok * A.ldo + row] = __float2bfloat16_rn(v);\n    }\n  }\n")
EPI_NEW = r'''  const int row = rt * 16 + (tid & 15), tok = tid >> 4;
  float vb = 0.f;   // GR3: this thread's bf16 pooler output (row, token) stays in a register (no global y)
  if (tid < 16 * m) {
    float v = 0.f;
    for (int q = 0; q < nw; ++q) v += red[q][(tid & 15) * 8 + tok];
    v *= A.scale[rt];
    vb = __bfloat162float(__float2bfloat16_rn(v));
  }
  if (warp < (16 * MT + 31) / 32) {   // 16 consecutive threads hold one token's 16 rows: fixed-order butterfly
    float s = vb * vb;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((tid & 15) == 0 && tok < m) B.ssq[(size_t)rt * MT + tok] = s;
  }
'''
TAIL_NEW = r'''  // 3. GR3: grid barrier (all NRT CTAs are co-resident at 2 per SM; prepare checks), rstd, this CTA's xn rows; CTA 0 routes
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    int g0;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(g0) : "l"(B.bar + 1) : "memory");
    if (atomicAdd(B.bar, 1) == (int)gridDim.x - 1) {
      atomicExch(B.bar, 0);
      __threadfence();
      atomicAdd(B.bar + 1, 1);
    } else {
      int g = g0;
      for (int it = 0; it < (1 << 20) && g == g0; ++it) {
        __nanosleep(32);
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(g) : "l"(B.bar + 1) : "memory");
      }
      if (g == g0) atomicOr(B.err, 1);
    }
  }
  __syncthreads();
  if (warp < m) {   // rstd of token `warp` from the NRT CTA partials, fixed order
    float s = 0.f;
    for (int c = lane; c < NRT; c += 32) s += __ldcg(B.ssq + (size_t)c * MT + warp);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) s_rstd[warp] = rsqrtf(s / (float)PD + G.eps);
  }
  __syncthreads();
  if (tid < 16 * m) G.xn[(size_t)tok * PD + row] = __float2bfloat16((vb * s_rstd[tok]) * __bfloat162float(G.nw[row]));
  if (rt != 0 || warp >= m) return;
'''
HOST_B = r'''
void g2_prt_b(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff, torch::Tensor esc,
              torch::Tensor fmt, torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor nw, torch::Tensor bias,
              torch::Tensor lstart, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor ssq,
              torch::Tensor bar, torch::Tensor err, double eps, double mult, double rscale, double deps, int64_t pdl) {
  // GR3 (tools/gr3_patches.py): prt_kernel_b, T <= 8, in-kernel barrier norm + CTA-0 routing
  const int64_t M = x.size(0);
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(1) == g2::RK &&
              M >= 1 && M <= 8 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "g2_prt_b: x bf16 [M <= 8, 8192] contiguous");
  TORCH_CHECK(pk.scalar_type() == at::kByte && toff.scalar_type() == at::kLong && scale.scalar_type() == at::kFloat &&
              eoff.scalar_type() == at::kInt && esc.scalar_type() == at::kInt && fmt.scalar_type() == at::kByte &&
              toff.numel() >= g2::NRT && scale.numel() >= g2::NRT && eoff.numel() >= g2::NRT + 1, "g2_prt_b: packed pooler");
  TORCH_CHECK(x32.scalar_type() == at::kFloat && x32.is_contiguous() && x32.size(0) == M && x32.size(1) == g2::RK &&
              reinterpret_cast<uintptr_t>(x32.data_ptr()) % 16 == 0, "g2_prt_b: x32 FP32 [M, 8192]");
  TORCH_CHECK(w32.scalar_type() == at::kFloat && w32.is_contiguous() && w32.size(0) == g2::NE && w32.size(1) == g2::RK &&
              reinterpret_cast<uintptr_t>(w32.data_ptr()) % 16 == 0, "g2_prt_b: w32 FP32 [256, 8192]");
  TORCH_CHECK(part.scalar_type() == at::kFloat && part.is_contiguous() && part.numel() == 8 * M * g2::NE &&
              reinterpret_cast<uintptr_t>(part.data_ptr()) % 16 == 0, "g2_prt_b: part [8, M, 256]");
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.numel() == g2::PD && bias.scalar_type() == at::kFloat &&
              bias.numel() == g2::NE && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 &&
              lstart.scalar_type() == at::kInt, "g2_prt_b: nw / bias / lstart");
  TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.size(0) == M && xn.size(1) == g2::PD &&
              tw.scalar_type() == at::kFloat && tw.is_contiguous() && tw.numel() == M * g2::TK && tid.scalar_type() == at::kInt &&
              tid.is_contiguous() && tid.numel() == M * g2::TK && lid.scalar_type() == at::kInt && lid.is_contiguous() &&
              lid.numel() == M * g2::TK, "g2_prt_b: outputs");
  TORCH_CHECK(ssq.scalar_type() == at::kFloat && ssq.numel() >= g2::NRT * 8 && bar.scalar_type() == at::kInt &&
              bar.numel() >= 2 && err.scalar_type() == at::kInt && err.numel() >= 1, "g2_prt_b: barrier scratch");
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  p1::Args A;
  A.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); A.M = (int)M; A.ldx = g2::RK;
  A.pk = pk.data_ptr<uint8_t>(); A.toff = toff.data_ptr<int64_t>(); A.scale = scale.data_ptr<float>();
  A.eoff = eoff.data_ptr<int>(); A.esc = reinterpret_cast<const int2*>(esc.data_ptr<int>()); A.fmt = fmt.data_ptr<uint8_t>();
  A.out = nullptr; A.ldo = 0; A.N = g2::PD; A.nsb_total = g2::RK / 64;
  A.ws = nullptr; A.cnt = nullptr; A.red_mode = 0; A.trig = 0;
  g2::PArgs G;
  G.x32 = x32.data_ptr<float>(); G.w32 = w32.data_ptr<float>(); G.part = part.data_ptr<float>();
  G.nw = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()); G.bias = bias.data_ptr<float>(); G.lstart = lstart.data_ptr<int>();
  G.xn = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()); G.tw = tw.data_ptr<float>(); G.tid = tid.data_ptr<int>();
  G.lid = lid.data_ptr<int>(); G.tcnt = nullptr;
  G.eps = (float)eps; G.mult = (float)mult; G.rscale = (float)rscale; G.deps = (float)deps;
  g2::BArgs Bx;
  Bx.ssq = ssq.data_ptr<float>(); Bx.bar = bar.data_ptr<int>(); Bx.err = err.data_ptr<int>();
  if (M == 1) g2::prt_b_launch<1>(A, G, Bx, (int)pdl);
  else g2::prt_b_launch<8>(A, G, Bx, (int)pdl);
}

std::vector<int64_t> g2_info_b(int64_t mt) {
  TORCH_CHECK(mt == 1 || mt == 8, "g2_info_b: mt 1 or 8");
  return mt == 1 ? g2::prt_b_info<1>() : g2::prt_b_info<8>();
}
'''
LAUNCH_B = r'''
template <int MT>
static void prt_b_launch(const Args& A, const PArgs& G, const BArgs& B, int pdl) {
  const int smem = (8 + MT) * RKS4 * 16;
  static bool attr_set = false;
  if (!attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(prt_kernel_b<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    attr_set = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NRT, 1, 1);
  cfg.blockDim = dim3((unsigned)NT, 1, 1);
  cfg.dynamicSmemBytes = (size_t)smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, prt_kernel_b<MT>, A, G, B));
}

template <int MT>
static std::vector<int64_t> prt_b_info() {
  const int smem = (8 + MT) * RKS4 * 16;
  C10_CUDA_CHECK(cudaFuncSetAttribute(prt_kernel_b<MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, prt_kernel_b<MT>));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, prt_kernel_b<MT>, NT, smem));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)occ, (int64_t)smem};
}

'''


def patch_g2(t):
    g = region(t, *G2_REG)
    a = g.index("template <int MT, int RM>\n__global__ void __launch_bounds__(512, 2) prt_kernel(")
    b = g.index("template <int MT, int RM>\nstatic void prt_launch(")
    k = g[a:b]
    sig_old = ("template <int MT, int RM>\n__global__ void __launch_bounds__(512, 2) prt_kernel(const __grid_constant__ Args A, "
               "const __grid_constant__ PArgs G) {\n")
    sig_new = ("// GR3 (tools/gr3_patches.py): prt_kernel<MT, 1> without the last-CTA tail; barrier norm + CTA-0 routing (see header)\n"
               "template <int MT>\n__global__ void __launch_bounds__(512, 2) prt_kernel_b(const __grid_constant__ Args A, "
               "const __grid_constant__ PArgs G, const __grid_constant__ BArgs B) {\n  constexpr int RM = 1;\n")
    for old, new in ((sig_old, sig_new), (EPI_OLD, EPI_NEW), ("  __shared__ int s_last;\n", "  __shared__ float s_rstd[8];\n")):
        assert k.count(old) == 1, old[:60]
        k = k.replace(old, new)
    c0 = k.index("  // 3. the last CTA (ticket; no spin) runs the tail for every token\n")
    c1 = k.index("  if (warp >= m) return;\n", c0) + len("  if (warp >= m) return;\n")
    k = k[:c0] + TAIL_NEW + k[c1:]
    assert "A.out" not in k and "s_last" not in k and "tred" in k
    k = k.replace("  __shared__ float tred[8][NT / 32];\n", "")
    t = sub(t, "  float eps, mult, rscale, deps;\n};\n",
            "  float eps, mult, rscale, deps;\n};\n\nstruct BArgs {   // GR3: barrier scratch of prt_kernel_b\n"
            "  float* ssq;   // [NRT][MT] per-CTA sums of squares of the bf16 pooler rows (rewritten every call)\n"
            "  int* bar;     // [2]: arrive count (reset by the last arriver), generation (monotonic)\n"
            "  int* err;     // [1]: barrier timeout flag\n};\n", *G2_REG)
    t = sub(t, "}  // namespace g2\n", k + LAUNCH_B + "}  // namespace g2\n", *G2_REG)
    gi = ('std::vector<int64_t> g2_info(int64_t mt, int64_t rt) {\n  TORCH_CHECK(mt == 1 && rt == 1, "g2_info: k3_v19 builds only '
          'prt_kernel<1, 1>");\n  return g2::prt_info<1, 1>();\n}\n')
    t = sub(t, gi, gi + HOST_B, *G2_REG)
    t = sub(t, "std::vector<int64_t> g2_info(int64_t mt, int64_t rt);\n",
            "std::vector<int64_t> g2_info(int64_t mt, int64_t rt);\n"
            "void g2_prt_b(torch::Tensor x, torch::Tensor pk, torch::Tensor toff, torch::Tensor scale, torch::Tensor eoff, torch::Tensor esc,\n"
            "              torch::Tensor fmt, torch::Tensor x32, torch::Tensor w32, torch::Tensor part, torch::Tensor nw, torch::Tensor bias,\n"
            "              torch::Tensor lstart, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor ssq,\n"
            "              torch::Tensor bar, torch::Tensor err, double eps, double mult, double rscale, double deps, int64_t pdl);\n"
            "std::vector<int64_t> g2_info_b(int64_t mt);\n")
    t = sub(t, 'functions=["g2_prt", "g2_info"]', 'functions=["g2_prt", "g2_info", "g2_prt_b", "g2_info_b"]')
    return t


PY_GR3 = r'''
# ============================================================================================ GR3b (tools/gr3_patches.py, lab)
GR3_ENABLE = False    # lab arms set True (read at call time); False = k3_v37
GR3_BUILD = True
GR3_REQUIRE = True    # lab file: prepare raises if a routed GR3 bucket fails its self-test
GR3_ROUTE = {1: True, 8: True}    # P1 bucket (T = 1 / T = 2..8) -> prt_kernel_b
GR3_PREPARE_BY_S = 260.0
GR3_SELF_TEST_CASES = ((1, 0, 811), (1, 224, 812), (2, 64, 813), (5, 160, 814), (8, 32, 815), (8, 96, 816), (8, 0, 817))
_GR3 = {"ok": set(), "st": None, "seen": set()}


def _gr3_run(g2, P, ffn_input, router_input, state, data, pdl):
    tokens, dev, st = ffn_input.shape[0], ffn_input.device, _GR3["st"]
    part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=dev)
    xn = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    v8 = state["v8"]
    g2["mod"].g2_prt_b(ffn_input, P["buf"], P["toff"], P["scale"], P["eoff"], P["esc"], P["fmt"], router_input, state["router"],
                       part, v8["nw"], v8["bias"], data["local_start"], xn, tw, tid, lid, st["ssq"], st["bar"], st["err"], RMS_EPS,
                       0.1767766952966369, 1.0, 1e-15, pdl)
    return xn, tw, tid, lid


def _gr3_selftest(state, g2):
    """prt_kernel_b vs the shipped three-kernel path (router32 + P1 pooler + v8 tail): xn within the tail's own rounding, routing
    weights / ids exact (same partials, same code), repeat bitwise, barrier count back at 0, err 0."""
    dev = state["router"].device
    P = state["p1"]["proj"]["pooler"]
    st = _GR3["st"]
    ok = {1: True, 8: True}
    for T, ls, seed in GR3_SELF_TEST_CASES:
        gen = torch.Generator(device=dev).manual_seed(seed)
        h = torch.randn((T, HIDDEN), generator=gen, device=dev)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
        hb = h.to(torch.bfloat16).contiguous()
        d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
        part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(h, state["router"], part)
        ref = _v8_tail(p1_gemv(state["p1"], "pooler", hb), part, state, d)
        outs = [_gr3_run(g2, P, hb, h, state, d, pdl) for pdl in (1, 0)]
        torch.cuda.synchronize()
        cnt, err = int(st["bar"][0]), int(st["err"][0])
        got = outs[0]
        ex = float((got[0].float() - ref[0].float()).norm() / ref[0].float().norm())
        ew = float((got[1] - ref[1]).abs().max())
        ids = bool(torch.equal(got[2], ref[2])) and bool(torch.equal(got[3], ref[3]))
        rep = all(bool(torch.equal(a, b)) for a, b in zip(outs[0], outs[1]))
        good = ex < 1e-3 and ew < 1e-6 and ids and rep and bool(torch.isfinite(got[0].float()).all()) and cnt == 0 and err == 0
        b = p1_bucket(T)
        ok[b] = ok[b] and good
        print(f"[gr3] self-test T={T} ls={ls}: xn rel {ex:.2e} (bitwise {bool(torch.equal(got[0], ref[0]))}) w maxdiff {ew:.1e} "
              f"ids {ids} rep {rep} barrier count {cnt} err {err} -> {'ok' if good else 'FAIL'}", flush=True)
        st["err"].zero_()
    return ok


def _gr3_prepare(state, t0):
    _GR3["ok"] = set()
    want = sorted(b for b, on in GR3_ROUTE.items() if on)
    if not (GR3_BUILD and want):
        return
    if time.time() - t0 > GR3_PREPARE_BY_S:
        print(f"[gr3] skipped: prepare already at {time.time() - t0:.0f}s", flush=True)
    else:
        try:
            g2, p1 = state.get("g2"), state.get("p1")
            if not (g2 is not None and hasattr(g2["mod"], "g2_prt_b") and p1 is not None and "pooler" in p1["proj"]
                    and p1["proj"]["pooler"]["mode_t8"] == 0 and state.get("v8") is not None):
                raise RuntimeError("G2 module / P1 pooler / v8 tail not live")
            dev = state["router"].device
            info = {mt: [int(v) for v in g2["mod"].g2_info_b(mt)] for mt in (1, 8)}
            print(f"[gr3] prt_kernel_b regs / local bytes / CTAs per SM / dyn smem: {info}", flush=True)
            if any(v[2] < 2 for v in info.values()):
                raise RuntimeError("prt_kernel_b below 2 CTAs/SM: the 256-CTA grid barrier needs one wave")
            _GR3["st"] = {"ssq": torch.zeros(256 * 8, dtype=torch.float32, device=dev),
                          "bar": torch.zeros(2, dtype=torch.int32, device=dev), "err": torch.zeros(1, dtype=torch.int32, device=dev)}
            res = _gr3_selftest(state, g2)
            _GR3["ok"] = {b for b in want if res.get(b)}
        except Exception as ex:
            print(f"[gr3] FAILED -> today's path: {ex!r}", flush=True)
            _GR3["ok"] = set()
    print(f"[gr3] live buckets {sorted(_GR3['ok'])} (prepare so far {time.time() - t0:.1f}s)", flush=True)
    missing = [b for b in want if b not in _GR3["ok"]]
    if GR3_REQUIRE and missing:
        raise RuntimeError(f"GR3_REQUIRE: GR3 not live for buckets {missing}")

'''


def patch_py(t):
    old = ('    g2 = state.get("g2")\n'
           '    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:\n        return None\n')
    new = ('    g2 = state.get("g2")\n'
           '    gr3 = bool(GR3_ENABLE and 1 <= tokens <= 8 and p1_bucket(tokens) in _GR3["ok"] and g2 is not None)   # GR3b\n'
           '    if not gr3 and (not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1):\n        return None\n')
    t = sub(t, old, new)
    old = ('        if G2_PRT_MODE.get(p1_bucket(tokens)) != "fused":\n            return None\n')
    new = ('        if gr3:\n'
           '            if tokens not in _GR3["seen"]:\n'
           '                _GR3["seen"].add(tokens)\n'
           '                print(f"[gr3] first call T={tokens}: prt_kernel_b (barrier norm + CTA-0 routing; no tail kernel, no side stream)",\n'
           '                      flush=True)\n'
           '            return _gr3_run(g2, p1["proj"]["pooler"], ffn_input, router_input, state, data, 1 if G2_PRT_PDL else 0)\n'
           + old)
    t = sub(t, old, new)
    a = t.index("\ndef kernel_fn(data, state):\n")
    t = t[:a] + "\n" + PY_GR3 + t[a:]
    t = sub(t, "    _su1_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n",
            "    _su1_prepare(state, t0)\n    _gr3_prepare(state, t0)\n    torch.cuda.synchronize()\n    _pt1_join_all()\n")
    return t


def main():
    t = SRC.read_text(encoding="utf-8")
    lines = t.split("\n")
    assert lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "line 1 / nonce"
    lines[1] = NONCE
    t = "\n".join(lines)
    t = patch_py(patch_g2(t))
    assert t.isascii(), "non-ASCII"
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(t, encoding="utf-8", newline="\n")
    py_compile.compile(str(DST), doraise=True)
    print(f"gr3: {SRC.name} -> {DST} ({len(t) - len(SRC.read_text(encoding='utf-8')):+d} chars, {len(t.encode())} B), compiles")


if __name__ == "__main__":
    main()

"""vf_patches.py: VF front / glue riders at T <= 16 on a stripped layer file (default kernels/muse_decode_layer/k3_vml_3.py),
notes/study/vf_front.md. Composes like gen_k3_vml.py: the base must be a finish() fixed point; the patch anchors are code
lines of the stripped text; then gen_k3_m32.finish (strips, compiles, ASCII, nonce, no duplicate names, checker scan <= base,
< 1 MB).

Riders (module globals, read at call time, so one file serves every A/B arm; prepare self-tests each one, a failed test
switches that rider off with a loud [vf] line):
  VF_R8    T <= 8 FP32 router (mode f32) on vf_router32: router32_kernel's per-(row, K slice) arithmetic (same FMA order,
           same shuffle tree, bitwise equal), but 128 CTAs x 256 threads, no smem, <= 40 registers, so ONE router CTA fits
           beside the two GD pooler CTAs of an SM (router32: 256 CTAs x 74 regs x 32 KB smem; two of them leave no room
           for a 98-register GD pooler CTA, which is why the T = 8 pooler ends one full pooler-CTA time after router32 ends,
           R-vf-1: 21.5 us for 58.7 MB where the pooler alone takes 13.2).
  VF_PRT   G2 prt (T = 1) tail: every CTA warms one line of the pre-expert norm weight / router bias in L2 at its ticket,
           and the last CTA issues the 8 router partial slices together with the pooler rows (one L2 round trip instead of
           two; slices staged in the dead reduction buffer, summed in the same order: bitwise equal).
  VF_SBB2  FFN boundary on SBB1's kernel (SGLang's arithmetic, constants loaded before griddepcontrol.wait, the residual
           stream L2-prefetched before the wait and loaded after it) instead of SGLang's JIT cluster kernel; per bucket.
Also adds GD_ROUTE ("pooler", 1) = the T = 8 geometry (prepared and self-tested by GD; used only when prt is off, the "t1u"
arm: G2_PRT False -> GD pooler + side-stream vf_router32 + tail_kernel2 at T = 1).

    python -I tools/vf_patches.py [--n=1] [--base=k3_vml_3] [--on=r8,prt,sbb2] [--lab-only]
      -> kernels/muse_decode_layer/k3_vf_<n>.py (defaults from --on; none = base behaviour) + kernels/lab/k3_vf_<n>_ab.py
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    if n != count:
        raise SystemExit(f"[vf] anchor {what or old[:60]!r}: found {n}, expected {count}")
    return t.replace(old, new)


# ----------------------------------------------------------------------------------------------------------------------
# 1. lean FP32 router (SBB1's module: small, built by PT1 in parallel)
# ----------------------------------------------------------------------------------------------------------------------
VF_ROUTER_CU = r'''
// VF_R8 (tools/vf_patches.py): FP32 router partials part[8][m][256], router32_kernel's arithmetic per (row, K slice of 1024)
// (lane-strided float4 FMAs in the same order, same xor tree: bitwise equal), 128 CTAs x 256 threads, no smem, <= 40 regs:
// one CTA fits beside two GD pooler CTAs of an SM. Warp w of CTA (g, s) handles rows 16 g + w and 16 g + 8 + w.
template <int MT>
__global__ void __launch_bounds__(256, 6) vf_router32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                             float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll 1
  for (int h = 0; h < 2; ++h) {
    const int row = blockIdx.x * 16 + h * 8 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
      float4 wv[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) wv[i] = __ldcs(wp + (hh * 2 + i) * 32 + lane);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int c = (hh * 2 + i) * 32 + lane;
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          if (j < m) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
            acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
            acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
            acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
            acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = acc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
    }
  }
}

// VF_ROUTER "lean4" (T = 2..8): 128 threads (4 warps, one row each, all 8 weight float4 of the row in flight like router32),
// x through L1, <= 80 regs (10,240 per CTA: fits beside two GD pooler CTAs), grid (64, 8). Same arithmetic: bitwise.
template <int MT>
__global__ void __launch_bounds__(128, 6) vf_router32b_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS, row = blockIdx.x * 4 + warp;
  const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
  float4 wv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
  float acc[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = i * 32 + lane;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      if (j < m) {
        const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
        acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
        acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
        acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
        acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    float a = acc[j];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
  }
}

// VF_ROUTER "lean5" (T = 2..8): 128 CTAs x 128 threads (one CTA per SM, whatever the launch order), <= 80 regs, no smem; warp w
// of CTA (g, s) handles rows 16 g + 4 q + w (q = 0..3) with all 8 weight float4 of a row in flight; all 4 rows are warmed in
// L2 at entry so rows 1-3 come from L2. Same arithmetic per (row, slice) as router32: bitwise.
template <int MT>
__global__ void __launch_bounds__(128, 6) vf_router32c_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                              float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024;
  const int sl = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int k0 = sl * KS;
#pragma unroll
  for (int q = 1; q < 4; ++q) {
    const float* pl = w + (long)(blockIdx.x * 16 + q * 4 + warp) * K + k0 + lane * 32;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
  }
#pragma unroll 1
  for (int q = 0; q < 4; ++q) {
    const int row = blockIdx.x * 16 + q * 4 + warp;
    const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
    float4 wv[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
    float acc[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = i * 32 + lane;
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        if (j < m) {
          const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (long)j * K + k0) + c);
          acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
          acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
          acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
          acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      float a = acc[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
      if (lane == 0 && j < m) part[((long)sl * m + j) * 256 + row] = a;
    }
  }
}
'''

VF_ROUTER_HOST = r'''
void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind) {
  const int m = (int)x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat &&
              x.is_contiguous() && w.is_contiguous() && part.is_contiguous() && x.size(1) == 8192 && w.size(0) == 256 &&
              w.size(1) == 8192 && m >= 1 && m <= 8 && part.numel() == 8L * m * 256 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "vf_router32: x f32 [m <= 8, 8192], w f32 [256, 8192], part [8, m, 256]");
  auto s = c10::cuda::getCurrentCUDAStream();
  if (kind == 2)
    vf_router32c_kernel<8><<<dim3(16, 8), 128, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else if (kind == 1)
    vf_router32b_kernel<8><<<dim3(64, 8), 128, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else if (m == 1)
    vf_router32_kernel<1><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else
    vf_router32_kernel<8><<<dim3(16, 8), 256, 0, s>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "vf_router32: launch failed");
}
'''
VF_ROUTER_DECL = " void vf_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part, int64_t kind);"


def patch_router(t):
    t = sub(t, "SBB1_HOST = r'''\n", "SBB1_HOST = r'''\n" + VF_ROUTER_CU, what="SBB1_HOST head")
    # the host block ends right before SBB1_DECL; append the router host function to it
    t = sub(t, "  else sbb_go<false, false>(p, T, s);\n}\n'''\n",
            "  else sbb_go<false, false>(p, T, s);\n}\n" + VF_ROUTER_HOST + "'''\n", what="sbb1_run end")
    t = sub(t, "double post_eps, double pre_eps, int64_t input_scale);'\n",
            "double post_eps, double pre_eps, int64_t input_scale);" + VF_ROUTER_DECL + "'\n", what="SBB1_DECL")
    t = sub(t, 'functions=["sbb1_run"],', 'functions=["sbb1_run", "vf_router32"],', what="sbb1 functions")
    t = sub(t, """    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
""", """    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        kind = VF_ROUTER.get(1 if router_input.shape[0] == 1 else 8, "r32")
        if kind != "r32" and _VF["r8"] and _SBB1["mod"] is not None:
            _SBB1["mod"].vf_router32(router_input, state["router"], part, {"lean4": 1, "lean5": 2}.get(kind, 0))
        else:
            state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
""", what="_v8_logits f32")
    return t


# ----------------------------------------------------------------------------------------------------------------------
# 2. prt tail (G2, T = 1)
# ----------------------------------------------------------------------------------------------------------------------
def patch_prt(t):
    t = sub(t, "  float eps, mult, rscale, deps;\n};\n", "  float eps, mult, rscale, deps;\n  int vf;   // VF_PRT tail\n};\n",
            what="PArgs")
    t = sub(t, """  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(G.tcnt, 1) == (int)gridDim.x - 1);
""", """  if (G.vf && tid < 2) {   // VF_PRT: warm the last CTA's constants in L2 (nw: 64 lines, bias: 8 lines)
    const char* pl = tid == 0 ? reinterpret_cast<const char*>(G.nw) + (rt & 63) * 128
                              : reinterpret_cast<const char*>(G.bias) + (rt & 7) * 128;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(pl));
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = (atomicAdd(G.tcnt, 1) == (int)gridDim.x - 1);
""", what="prt ticket")
    t = sub(t, """    yq[t] = t < m ? __ldcg(reinterpret_cast<const uint4*>(A.out + (size_t)t * A.ldo) + tid) : make_uint4(0u, 0u, 0u, 0u);
""", """    yq[t] = t < m ? __ldcg(reinterpret_cast<const uint4*>(A.out + (size_t)t * A.ldo) + tid) : make_uint4(0u, 0u, 0u, 0u);
  float4 vb0 = make_float4(0.f, 0.f, 0.f, 0.f), vb1 = vb0;
  if (MT == 1 && G.vf) {   // VF_PRT: the routing operands ride with the pooler-row loads (red[][] is dead here)
    if (warp < 8) {
      const float4* pq = reinterpret_cast<const float4*>(G.part + (long)warp * m * NE) + lane * 2;
      const float4 u0 = __ldcg(pq), u1 = __ldcg(pq + 1);
      float4* r4 = reinterpret_cast<float4*>(&red[0][0]) + warp * 64 + lane * 2;
      r4[0] = u0;
      r4[1] = u1;
      if (warp == 0) {   // the routing warp (t = 0) keeps the bias in registers
        const float4* pb = reinterpret_cast<const float4*>(G.bias) + lane * 2;
        vb0 = pb[0];
        vb1 = pb[1];
      }
    }
  }
""", what="prt yq loads")
    t = sub(t, """  const float4* pbias = reinterpret_cast<const float4*>(G.bias) + lane * 2;
  const float4 c0b = pbias[0], c1b = pbias[1];
  const long sstride = (long)m * NE;
  const float4* pq0 = reinterpret_cast<const float4*>(G.part + (long)t * NE) + lane * 2;
  float4 a0 = __ldcg(pq0), a1 = __ldcg(pq0 + 1);
#pragma unroll
  for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in a fixed order
    const float4* pq = reinterpret_cast<const float4*>(G.part + q * sstride + (long)t * NE) + lane * 2;
    const float4 b0 = __ldcg(pq), b1 = __ldcg(pq + 1);
""", """  const bool vfs = MT == 1 && G.vf;
  const float4* rsm = reinterpret_cast<const float4*>(&red[0][0]);
  const float4* pbias = reinterpret_cast<const float4*>(G.bias) + lane * 2;
  const float4 c0b = vfs ? vb0 : pbias[0], c1b = vfs ? vb1 : pbias[1];
  const long sstride = (long)m * NE;
  const float4* pq0 = reinterpret_cast<const float4*>(G.part + (long)t * NE) + lane * 2;
  float4 a0 = vfs ? rsm[lane * 2] : __ldcg(pq0), a1 = vfs ? rsm[lane * 2 + 1] : __ldcg(pq0 + 1);
#pragma unroll
  for (int q = 1; q < 8; ++q) {   // FP32 split-K partials summed in a fixed order
    const float4* pq = reinterpret_cast<const float4*>(G.part + q * sstride + (long)t * NE) + lane * 2;
    const float4 b0 = vfs ? rsm[q * 64 + lane * 2] : __ldcg(pq), b1 = vfs ? rsm[q * 64 + lane * 2 + 1] : __ldcg(pq + 1);
""", what="prt routing loads")
    t = sub(t, "  G.eps = (float)eps; G.mult = (float)mult; G.rscale = (float)rscale; G.deps = (float)deps;\n",
            "  G.eps = (float)eps; G.mult = (float)mult; G.rscale = (float)rscale; G.deps = (float)deps;\n"
            "  G.vf = (int)((pdl >> 1) & 1);\n", what="prt host vf")
    t = sub(t, "  g2::prt_launch<1, 1>(A, G, (int)pdl);\n", "  g2::prt_launch<1, 1>(A, G, (int)(pdl & 1));\n",
            what="prt launch pdl")
    t = sub(t, "        return _g2_prt_run(g2, p1[\"proj\"][\"pooler\"], ffn_input, router_input, state, data, 1 if G2_PRT_PDL else 0)\n",
            "        return _g2_prt_run(g2, p1[\"proj\"][\"pooler\"], ffn_input, router_input, state, data,\n"
            "                           (1 if G2_PRT_PDL else 0) | (2 if (VF_PRT and _VF[\"prt\"]) else 0))\n", what="_g2_prt call")
    return t


# ----------------------------------------------------------------------------------------------------------------------
# 3. FFN boundary on SBB1's kernel (stream after the wait, L2-prefetched before it)
# ----------------------------------------------------------------------------------------------------------------------
def patch_sbb2(t):
    t = sub(t, "  if (kPreStream) sb_ld8f(p.stream + (size_t)row * SB_H + e0, st);\n",
            "  if (kPreStream) sb_ld8f(p.stream + (size_t)row * SB_H + e0, st);\n"
            "  else if ((tid & 3) == 0) asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(p.stream + (size_t)row * SB_H + e0));\n",
            what="sbb prefetch")
    t = sub(t, """template <bool B, bool C>
static void sbb_go(const SbbParams& p, int T, cudaStream_t s) {""", """template <bool A, bool B, bool C>
static void sbb_go(const SbbParams& p, int T, cudaStream_t s) {""", what="sbb_go head")
    t = sub(t, "    cudaFuncSetAttribute(sbb_kernel<true, B, C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n",
            "    cudaFuncSetAttribute(sbb_kernel<A, B, C>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n", what="sbb attr")
    t = sub(t, "  cudaLaunchKernelEx(&cfg, sbb_kernel<true, B, C>, p);\n", "  cudaLaunchKernelEx(&cfg, sbb_kernel<A, B, C>, p);\n",
            what="sbb launch")
    t = sub(t, """  const bool r = p.router != nullptr;
  if (r && input_scale) sbb_go<true, true>(p, T, s);
  else if (r) sbb_go<true, false>(p, T, s);
  else if (input_scale) sbb_go<false, true>(p, T, s);
  else sbb_go<false, false>(p, T, s);
""", """  const bool r = p.router != nullptr, post = (input_scale & 2) != 0, isc = (input_scale & 1) != 0;
  TORCH_CHECK(!(post && r), "sbb1: the post-wait stream mode has no router output");
  if (post && isc) sbb_go<false, false, true>(p, T, s);
  else if (post) sbb_go<false, false, false>(p, T, s);
  else if (r && isc) sbb_go<true, true, true>(p, T, s);
  else if (r) sbb_go<true, true, false>(p, T, s);
  else if (isc) sbb_go<true, false, true>(p, T, s);
  else sbb_go<true, false, false>(p, T, s);
""", what="sbb1_run dispatch")
    t = sub(t, "def _v9_bnd(which, stream, branch, router_dtype, state, tokens):\n",
            "def _v9_bnd(which, stream, branch, router_dtype, state, tokens):\n"
            "    if which == \"ffn\" and router_dtype is None and VF_SBB2.get(_k2a_bucket_of(tokens)):\n"
            "        r = _vf_sbb2(stream, branch, tokens)\n"
            "        if r is not None:\n"
            "            return r\n", what="_v9_bnd head")
    return t


VF_PY = r'''

VF_ROUTER = @ROUTER@
VF_PRT = @PRT@
VF_SBB2 = @SBB2@
VF_SELF_TEST_SEEDS = (731, 732, 733)
_VF = {"r8": False, "prt": False, "st2": None, "seen": set()}


def _vf_sbb2(stream, branch, tokens):
    st = _VF["st2"]
    if st is None or not st["ok"].get(_k2a_bucket_of(tokens)) or branch.dtype != torch.bfloat16 or stream.dtype != torch.float32:
        return None
    dev = branch.device
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    pw, qw, al, bt, isc, pe, qe = st["w"]
    st["mod"].sbb1_run(branch.contiguous(), stream.contiguous(), out, ns, st["e"], pw, qw, al, bt, pe, qe, isc | 2)
    if ("sbb2", tokens) not in _VF["seen"]:
        _VF["seen"].add(("sbb2", tokens))
        print(f"[vf] first call sbb2 T={tokens}", flush=True)
    return out, ns, None


def _vf_t1_live(state):
    if not VF_T1:
        return False
    ok = (VF_ROUTER.get(1) == "lean" and _VF["r8"] and _SBB1["mod"] is not None and state.get("v8") is not None
          and _gd_live(state, "pooler", 1) is not None)
    if not ok and "t1off" not in _VF["seen"]:
        _VF["seen"].add("t1off")
        print("[vf] FALLBACK t1: lean router / GD pooler T=1 not live -> G2 prt at T = 1", flush=True)
    return ok


def _vf_test_r8(state, mod):
    dev = state["router"].device
    ok = True
    for m, seed in ((1, 731), (3, 732), (8, 733), (8, 734)):
        gen = torch.Generator(device=dev).manual_seed(seed)
        z = torch.randn((m, HIDDEN), generator=gen, device=dev)
        z = (z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True))).contiguous()
        ref = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        got = torch.full((8, m, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
        state["v8"]["mod"].v8_router32(z, state["router"], ref)
        for kind in ((0,) if m == 1 else (0, 1, 2)):
            got.fill_(float("nan"))
            mod.vf_router32(z, state["router"], got, kind)
            torch.cuda.synchronize()
            eq = bool(torch.equal(ref, got))
            print(f"[vf] r8 self-test m={m} kind {kind}: router partials bitwise {eq}", flush=True)
            ok &= eq
    return ok


def _vf_test_prt(state, g2):
    P = state["p1"]["proj"]["pooler"]
    dev = state["router"].device
    ok = True
    for T, ls, seed in G2_SELF_TEST_CASES[:3]:
        gen = torch.Generator(device=dev).manual_seed(seed + 7)
        h = torch.randn((T, HIDDEN), generator=gen, device=dev)
        h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
        hb = h.to(torch.bfloat16).contiguous()
        d = {"local_start": torch.tensor([ls], dtype=torch.int32, device=dev)}
        a = _g2_prt_run(g2, P, hb, h, state, d, 1)
        b = _g2_prt_run(g2, P, hb, h, state, d, 3)
        c = _g2_prt_run(g2, P, hb, h, state, d, 2)
        torch.cuda.synchronize()
        eq = all(bool(torch.equal(x, y)) for x, y in zip(a, b)) and all(bool(torch.equal(x, y)) for x, y in zip(a, c))
        cnt = int(g2["tcnt"][0])
        print(f"[vf] prt tail self-test T={T} ls={ls}: xn / w / ids / lids bitwise {eq}, ticket {cnt}", flush=True)
        ok &= eq and cnt == 0
    return ok


def _vf_prepare(state, t0):
    t1 = time.time()
    st1 = _SBB1["st"]
    mod = _SBB1["mod"]
    _VF["r8"] = False
    _VF["prt"] = False
    _VF["st2"] = None
    if mod is None or st1 is None:
        print("[vf] SBB1 module missing -> VF riders off", flush=True)
        return
    try:
        if state.get("v8") is not None and state.get("router") is not None:
            _VF["r8"] = _vf_test_r8(state, mod)
    except Exception as ex:
        print(f"[vf] r8 FAILED -> router32 stays: {ex!r}", flush=True)
        _VF["r8"] = False
    try:
        g2 = state.get("g2")
        if g2 is not None and g2.get("prt"):
            _VF["prt"] = _vf_test_prt(state, g2)
    except Exception as ex:
        print(f"[vf] prt tail FAILED -> G2 tail stays: {ex!r}", flush=True)
        _VF["prt"] = False
    try:
        op = state["ffn_boundary"]
        op._ensure_cached_weights(torch.bfloat16)
        dev = state["router"].device
        e = torch.empty(0, device=dev)
        w = (op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e,
             op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e,
             op._cc_alpha.float().contiguous() if op._use_residual_gate else e,
             op._cc_beta.float().contiguous() if op._use_residual_gate else e,
             1 if op._post_norm_is_input_scale else 0, float(op._post_norm.eps), float(op._pre_norm.eps))
        st2 = {"mod": mod, "w": w, "e": e, "ok": {}}
        gen = torch.Generator(device=dev).manual_seed(5254)
        for rep in sorted(VF_SBB2):
            lo, hi = [(a, b) for a, b, r in K2A_BUCKETS if r == rep][0]
            good = True
            for T in sorted({lo, rep, hi}):
                stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
                branch = torch.randn((T, HIDDEN), generator=gen, device=dev).to(torch.bfloat16)
                if T <= SLICED_UNPOOL_MAX_TOKENS:
                    branch[:, UNPOOL_SLICE:] = 0
                ref = op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
                pw, qw, al, bt, isc, pe, qe = w
                out = torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev)
                ns = torch.empty((T, HIDDEN), dtype=torch.float32, device=dev)
                mod.sbb1_run(branch, stream, out, ns, e, pw, qw, al, bt, pe, qe, isc | 2)
                torch.cuda.synchronize()
                e0 = float((out.float() - ref[0].float()).norm() / ref[0].float().norm())
                e1 = float((ns - ref[1]).norm() / ref[1].norm())
                bw = bool(torch.equal(out, ref[0])) and bool(torch.equal(ns, ref[1]))
                good &= e0 < 1e-2 and e1 < 1e-4 and bool(torch.isfinite(ns).all())
                print(f"[vf] sbb2 self-test ffn T={T}: out {e0:.1e} stream {e1:.1e} bitwise {bw} -> {'ok' if good else 'FAIL'}",
                      flush=True)
            st2["ok"][rep] = good
        _VF["st2"] = st2
    except Exception as ex:
        print(f"[vf] sbb2 FAILED -> SGLang FFN boundary stays: {ex!r}", flush=True)
        _VF["st2"] = None
    print(f"[vf] riders router {'LIVE' if _VF['r8'] else 'OFF'} prt {'LIVE' if _VF['prt'] else 'OFF'} sbb2 "
          f"{sorted(k for k, v in (_VF['st2'] or {'ok': {}})['ok'].items() if v)} (switches ROUTER {VF_ROUTER} PRT {VF_PRT} "
          f"SBB2 {VF_SBB2}; {time.time() - t1:.1f}s, prepare so far {time.time() - t0:.1f}s)", flush=True)
'''


def patch_py(t, on):
    blk = (VF_PY.replace("@ROUTER@", repr({1: "lean" if "t1" in on else "r32", 8: ("lean4" if "t8" in on else "r32")})).replace("@PRT@", str("prt" in on))
           .replace("@SBB2@", "{1: True, 8: True, 16: True}" if "sbb2" in on else "{1: False, 8: False, 16: False}"))
    t = sub(t, "\n\nSBB1_ENABLE = True\n", blk + "\n\nSBB1_ENABLE = True\n", what="SBB1_ENABLE")
    t = sub(t, "    _sbb1_prepare(state, t0)\n", "    _sbb1_prepare(state, t0)\n    _vf_prepare(state, t0)\n", what="prepare call")
    # VF_T1: T = 1 on GD pooler + side-stream vf_router32 + tail_kernel2 when that path is live, else G2's prt (fallback)
    t = sub(t, '    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:\n        return None\n',
            '    if not (G2_ENABLE and G2_PRT and g2 is not None and g2.get("prt")) or tokens != 1:\n        return None\n'
            '    if _vf_t1_live(state):\n        return None\n', what="_g2_prt t1")
    t = sub(t, "\nVF_SELF_TEST_SEEDS = ", "\nVF_T1 = " + str("t1" in on) + "\nVF_SELF_TEST_SEEDS = ", what="VF_T1")
    if "ppdl" in on:   # GD pooler launched with PDL (T = 1 after the fork, T = 8)
        t = sub(t, 'GD_PDL = {"qkv": False, "o": True, "pooler": False, "unpooler": True}',
                'GD_PDL = {"qkv": False, "o": True, "pooler": True, "unpooler": True}', what="GD_PDL")
    if "galt" in on:   # alternative GD qkv geometries at T = 16 / 32 that leave one KS3 CTA slot per SM (call-time VF_GD_ON)
        t = sub(t, '            ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)}\n',
                '            ("unpooler", 16): (2, 2, 4, 16, 2, 1, 0), ("unpooler", 32): (4, 2, 4, 16, 1, 1, 0),\n'
                '            ("qkv@vf", 16): (2, 1, 2, 32, 4, 2, 0), ("qkv@vf", 32): (4, 1, 2, 32, 3, 2, 0)}\n', what="GD_ROUTE alt")
        t = sub(t, 'GD_DIMS = {"qkv": (2304, HIDDEN),', 'GD_DIMS = {"qkv@vf": (2304, HIDDEN), "qkv": (2304, HIDDEN),', what="GD_DIMS")
        t = sub(t, 'GD_WEIGHT = {"qkv": "qkv_gate",', 'GD_WEIGHT = {"qkv@vf": "qkv_gate", "qkv": "qkv_gate",', what="GD_WEIGHT")
        t = sub(t, "    T = x.shape[0]\n    r = _gd_live(state, proj, T)\n",
                "    T = x.shape[0]\n    r = ((_gd_live(state, proj + \"@vf\", T) if VF_GD_ON.get((proj, _gd_bucket(T))) else None)\n"
                "         or _gd_live(state, proj, T))\n", what="_gd alt")
        t = sub(t, "\nVF_SELF_TEST_SEEDS = ", "\nVF_GD_ON = {(\"qkv\", 16): False, (\"qkv\", 32): False}\nVF_SELF_TEST_SEEDS = ",
                what="VF_GD_ON")
    if "t8m" in on:   # T = 8 router before the pooler on the main stream, pooler PDL
        t = sub(t, "V8_ROUTER_PLACE = {1: 'side', 8: 'side_first',", "V8_ROUTER_PLACE = {1: 'side', 8: 'main',", what="V8 place")
        t = sub(t, 'GD_PDL = {"qkv": False, "o": True, "pooler": False, "unpooler": True}',
                'GD_PDL = {"qkv": False, "o": True, "pooler": True, "unpooler": True}', what="GD_PDL")
    t = sub(t, '            ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),\n',
            '            ("pooler", 1): (1, 1, 8, 16, 2, 1, 1), ("pooler", 8): (1, 1, 8, 16, 2, 1, 1),\n', what="GD pooler T=1")
    return t


def apply(t, on=()):
    t = patch_router(t)
    t = patch_prt(t)
    t = patch_sbb2(t)
    t = patch_py(t, set(on))
    return t


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    n = ARGS.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vml_3')}.py"
    on = [x for x in ARGS.get("on", "").split(",") if x]
    labonly = "--lab-only" in sys.argv
    out = ROOT / f"kernels/muse_decode_layer/k3_vf_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vf_{n}_ab.py"
    tag = f"k3-vf-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    GM.check(fb == b, f"{base.name} is not a fixed point of finish()")
    rep.append(f"base {base.name} {len(b.encode())} B is a finish() fixed point")
    nonce = (f"# nonce {tag} ({base.stem} + VF front riders: lean T <= 8 FP32 router beside the GD pooler, prt tail one L2 trip, "
             f"FFN boundary on SBB; on: {','.join(on) or 'none'}; self-tested with fallback)")
    t = apply(b.replace(bn, nonce), on)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vf_patches] OK  {r}")
    if not labonly:
        out.write_text(s, encoding="utf-8", newline=NL)
    lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vf_patches] wrote {'' if labonly else out.relative_to(ROOT).as_posix() + ' and '}{lab.relative_to(ROOT).as_posix()} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

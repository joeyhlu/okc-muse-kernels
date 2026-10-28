"""vds_mom_lab.py: lever P6a step 0 (notes/study/gap_oct28.md s6) = the attention boundary's per-token moments produced in
GD o's epilogue, so SBB needs no cross-CTA reduction (VDS builder, 2026-10-28).

GD o (the layer's gd_kernel, source taken from the layer file) gets OUT = 2: it stores bf16 o as OUT 0 does, and each CTA also
writes one partial record per token, {sum v^2, sum (a s)^2, sum (a s)(b mv), sum (b mv)^2} over its rows (v = the bf16 o value,
mv = v [x post_w], a / b = residual-gate coefficients, s = the FP32 stream), the four sums SBB1 reduces over its 16-CTA
cluster. SBBM = SBB1's arithmetic with the cluster exchange replaced by a sum of the token's P partial records (no cluster).
Arms (standalone, CUDA graphs, L2 flushed before every replay, x re-read into L2 as the layer's x is fresh):
  gd0 / gd2           GD o OUT 0 / OUT 2 alone                  -> the epilogue cost (gate: < 0.5 us)
  sbb1 / sbbm         SBB1 / SBBM alone (inputs prepared)         -> the boundary body saving
  chain0 / chain2     GD o + SBB1 / GD o OUT 2 + SBBM, PDL chain  -> the in-layer-like pair
Correctness: GD o OUT 2's o is bitwise OUT 0's; SBBM's outputs vs SBB1's (bf16 out, FP32 stream, FP32 router input).

    python3 kernels/lab/vds_mom_lab.py [--ts=1,8,16,32] [--reps=21] [--rounds=6] [--layer=kernels/lab/k3_slim_5_ab.py]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vds_lab as VL  # noqa: E402  (layer_extract, graph_of, time_many, arg)

H = 8192
O_ROUTES = {1: (1, 2, 4, 4, 2, 1, 0), 8: (1, 1, 8, 2, 2, 1, 0), 16: (2, 2, 4, 4, 2, 1, 0), 32: (4, 2, 4, 4, 1, 1, 0)}


def gdm_patch(cuda, cpp):
    """the layer's GD source + OUT 2 (bf16 store + per-token moment partial records) and a gd_run_mom entry point."""
    def rep(s, old, new, what, regex=False):
        n = len(re.findall(old, s)) if regex else s.count(old)
        if n != 1:
            raise ValueError(f"gdm anchor {what!r} found {n} times")
        return re.sub(old, lambda m: new, s) if regex else s.replace(old, new)
    cuda = rep(cuda, r"int trig;[^\n]*\n};",
               "int trig;\n  const float* ms; const float* mal; const float* mbe; const uint16_t* mpw; float* mom; int mflags;\n};",
               "Args", regex=True)
    cuda = rep(cuda, "  __shared__ __align__(8) unsigned long long mbar;\n",
               "  __shared__ __align__(8) unsigned long long mbar;\n"
               "  constexpr int MJ = (NO + NT - 1) / NT;\n"
               "  __shared__ __align__(16) float mom_s[OUT == 2 ? 8 * MT : 1][4];\n", "smem")
    cuda = rep(cuda, "  pdl_wait();\n  const uint4* xb = ",
               "  pdl_wait();\n"
               "  float mS[OUT == 2 ? MJ : 1], mA[OUT == 2 ? MJ : 1], mB[OUT == 2 ? MJ : 1], mP[OUT == 2 ? MJ : 1];\n"
               "  if constexpr (OUT == 2) {\n"
               "    for (int i = (int)threadIdx.x; i < 8 * MT * 4; i += NT) (&mom_s[0][0])[i] = 0.f;\n"
               "#pragma unroll\n"
               "    for (int j = 0; j < MJ; ++j) {\n"
               "      const int i = (int)threadIdx.x + j * NT;\n"
               "      const int l = i & 31, gg = l >> 2, tt = l & 3, c = (i >> 5) & 3, r = i >> 7, mt = r % MT, w2 = r / MT;\n"
               "      const int row = ((int)blockIdx.x * WR + w2) * 16 + gg + 8 * (c >> 1), tok = tok0 + mt * 8 + 2 * tt + (c & 1);\n"
               "      const bool ok = i < NO && tok < A.M;\n"
               "      mS[j] = ok ? __ldg(A.ms + (size_t)tok * A.ldo + row) : 0.f;\n"
               "      mA[j] = (A.mflags & 4) ? __ldg(A.mal + ((A.mflags & 8) ? row : 0)) : 1.f;\n"
               "      mB[j] = (A.mflags & 4) ? __ldg(A.mbe + ((A.mflags & 8) ? row : 0)) : 1.f;\n"
               "      mP[j] = (A.mflags & 1) ? __uint_as_float((uint32_t)A.mpw[row] << 16) : 1.f;\n"
               "    }\n"
               "  }\n"
               "  const uint4* xb = ", "moment input prefetch")
    cuda = rep(cuda, "    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n",
               "    if constexpr (OUT == 2) {\n"
               "#pragma unroll\n"
               "      for (int j = 0; j < MJ; ++j) {\n"
               "        const int i = (int)threadIdx.x + j * NT;\n"
               "        if (i < NO) {\n"
               "          const float v = cta_value(i);\n"
               "          store(i, v);\n"
               "          const int l = i & 31, tt = l & 3, c = (i >> 5) & 3, mt = (i >> 7) % MT;\n"
               "          const int slot = mt * 8 + 2 * tt + (c & 1);\n"
               "          if (tok0 + slot < A.M) {\n"
               "            float vv = __bfloat162float(__float2bfloat16_rn(v));\n"
               "            if ((A.mflags & 3) == 3) vv = __bfloat162float(__float2bfloat16_rn(vv * mP[j]));\n"
               "            const float mv = ((A.mflags & 3) == 1) ? vv * mP[j] : vv;\n"
               "            const float rr = mA[j] * mS[j], sb = mB[j] * mv;\n"
               "            atomicAdd(&mom_s[slot][0], vv * vv);\n"
               "            atomicAdd(&mom_s[slot][1], rr * rr);\n"
               "            atomicAdd(&mom_s[slot][2], rr * sb);\n"
               "            atomicAdd(&mom_s[slot][3], sb * sb);\n"
               "          }\n"
               "        }\n"
               "      }\n"
               "      __syncthreads();\n"
               "      if ((int)threadIdx.x < 8 * MT && tok0 + (int)threadIdx.x < A.M)\n"
               "        reinterpret_cast<float4*>(A.mom)[(size_t)(tok0 + threadIdx.x) * gridDim.x + blockIdx.x] =\n"
               "            *reinterpret_cast<const float4*>(&mom_s[threadIdx.x][0]);\n"
               "    } else {\n"
               "      for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
               "    }\n", "S == 1 store loop")
    cuda = rep(cuda, "int gd_info_raw(int64_t inst, int64_t* res) {",
               "int gd_run_mom_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo,\n"
               "                   uint32_t tb, int nbx, int nz, int pdl, const float* ms, const float* mal, const float* mbe,\n"
               "                   const uint16_t* mpw, float* mom, int mflags, cudaStream_t st) {\n"
               "  Args A;\n"
               "  A.w = reinterpret_cast<const uint4*>(w);\n"
               "  A.esc = reinterpret_cast<const uint32_t*>(esc);\n"
               "  A.x = reinterpret_cast<const uint16_t*>(x);\n"
               "  A.out = out;\n"
               "  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;\n"
               "  A.trig = (pdl >> 1) & 1;\n"
               "  A.ms = ms; A.mal = mal; A.mbe = mbe; A.mpw = mpw; A.mom = mom; A.mflags = mflags;\n"
               "  switch (inst) {\n@INST@\n    default: return -1;\n  }\n}\n\n"
               "int gd_info_raw(int64_t inst, int64_t* res) {", "host entry")
    cpp = rep(cpp, "int gd_info_raw(int64_t inst, int64_t* res);",
              "int gd_info_raw(int64_t inst, int64_t* res);\n"
              "int gd_run_mom_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo,\n"
              "                   uint32_t tb, int nbx, int nz, int pdl, const float* ms, const float* mal, const float* mbe,\n"
              "                   const uint16_t* mpw, float* mom, int mflags, cudaStream_t st);\n"
              "void gd_run_mom(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M,\n"
              "                int64_t K, int64_t ldo, int64_t tb, int64_t nbx, int64_t nz, int64_t pdl, torch::Tensor ms,\n"
              "                torch::Tensor mal, torch::Tensor mbe, torch::Tensor mpw, torch::Tensor mom, int64_t mflags) {\n"
              "  const int e = gd_run_mom_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K,\n"
              "      (int)ldo, (uint32_t)tb, (int)nbx, (int)nz, (int)pdl, ms.data_ptr<float>(),\n"
              "      mal.numel() ? mal.data_ptr<float>() : nullptr, mbe.numel() ? mbe.data_ptr<float>() : nullptr,\n"
              "      mpw.numel() ? reinterpret_cast<const uint16_t*>(mpw.data_ptr()) : nullptr, mom.data_ptr<float>(), (int)mflags,\n"
              "      at::cuda::getCurrentCUDAStream());\n"
              "  TORCH_CHECK(e == 0, \"gd_run_mom: inst \", inst, \" CUDA error \", e);\n"
              "}", "cpp decl")
    return cuda, cpp


SBBM_CU = r'''
// SBBM: SBB1's arithmetic; the four per-token sums come from the producer's partial records (P per token), no cluster.
template <bool kRouter, bool kInputScale>
__global__ void __launch_bounds__(SB_NT) sbbm_kernel(const SbbParams p, const float* __restrict__ mom, int P) {
  __shared__ float warp_red[SB_NT / 32][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.x / SB_CS, rank = blockIdx.x % SB_CS;
  const int e0 = (rank * SB_NT + tid) * 8;
  float pw[8], qw[8], al[8], bt[8], st[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { pw[i] = 1.f; qw[i] = 1.f; al[i] = 1.f; bt[i] = 1.f; }
  if (p.post_w) sb_ld8bf(p.post_w + e0, pw);
  if (p.pre_w) sb_ld8bf(p.pre_w + e0, qw);
  if (p.alpha) {
    if (p.gate_channel_wise) { sb_ld8f(p.alpha + e0, al); sb_ld8f(p.beta + e0, bt); }
    else {
#pragma unroll
      for (int i = 0; i < 8; ++i) { al[i] = p.alpha[0]; bt[i] = p.beta[0]; }
    }
  }
  sb_ld8f(p.stream + (size_t)row * SB_H + e0, st);
  asm volatile("griddepcontrol.wait;" ::: "memory");
  float br[8];
  sb_ld8bf(p.branch + (size_t)row * SB_H + e0, br);
  float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
  const float4* mr = reinterpret_cast<const float4*>(mom) + (size_t)row * P;
  for (int q = tid; q < P; q += SB_NT) {
    const float4 v = __ldcg(mr + q);
    m.x += v.x; m.y += v.y; m.z += v.z; m.w += v.w;
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    m.x += __shfl_xor_sync(0xffffffffu, m.x, o);
    m.y += __shfl_xor_sync(0xffffffffu, m.y, o);
    m.z += __shfl_xor_sync(0xffffffffu, m.z, o);
    m.w += __shfl_xor_sync(0xffffffffu, m.w, o);
  }
  if (lane == 0) { warp_red[warp][0] = m.x; warp_red[warp][1] = m.y; warp_red[warp][2] = m.z; warp_red[warp][3] = m.w; }
  __syncthreads();
  float tb = 0.f, tr = 0.f, tc = 0.f, ts = 0.f;
#pragma unroll
  for (int w = 0; w < SB_NT / 32; ++w) { tb += warp_red[w][0]; tr += warp_red[w][1]; tc += warp_red[w][2]; ts += warp_red[w][3]; }
  const float post_rstd = rsqrtf(tb * (1.f / (float)SB_H) + p.post_eps);
  const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, ts, 2.f * tc), tr);
  const float pre_rstd = rsqrtf(fmaxf(pre_sum, 0.f) * (1.f / (float)SB_H) + p.pre_eps);
  float ns[8], nz[8];
  __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float v = br[i];
    if (p.post_w && kInputScale) v = sb_rbf(v * pw[i]);
    float n = v * post_rstd;
    if (p.post_w && !kInputScale) n *= pw[i];
    n = sb_rbf(n);
    const float u = fmaf(bt[i], n, al[i] * st[i]);
    ns[i] = u;
    float z = u * pre_rstd;
    if (p.pre_w) z *= qw[i];
    nz[i] = z;
    ob[i] = __float2bfloat16_rn(z);
  }
  *reinterpret_cast<uint4*>(p.out + (size_t)row * SB_H + e0) = *reinterpret_cast<const uint4*>(ob);
  sb_st8f(p.new_stream + (size_t)row * SB_H + e0, ns);
  if (kRouter) sb_st8f(p.router + (size_t)row * SB_H + e0, nz);
}

template <bool A, bool B>
static void sbbm_go(const SbbParams& p, const float* mom, int P, int T, cudaStream_t s) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(T * SB_CS); cfg.blockDim = dim3(SB_NT); cfg.dynamicSmemBytes = 0; cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, sbbm_kernel<A, B>, p, mom, P);
}
void sbbm_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, torch::Tensor router,
              torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, double post_eps,
              double pre_eps, int64_t input_scale, torch::Tensor mom, int64_t P) {
  const int T = branch.size(0);
  TORCH_CHECK(branch.size(1) == SB_H && stream.size(1) == SB_H && branch.is_contiguous() && stream.is_contiguous(), "sbbm: layout");
  TORCH_CHECK(mom.numel() >= (int64_t)T * P * 4 && mom.scalar_type() == at::kFloat, "sbbm: moments");
  SbbParams p = {};
  p.branch = (const __nv_bfloat16*)branch.data_ptr(); p.stream = stream.data_ptr<float>();
  p.out = (__nv_bfloat16*)out.data_ptr(); p.new_stream = new_stream.data_ptr<float>();
  p.router = router.numel() ? router.data_ptr<float>() : nullptr;
  p.post_w = post_w.numel() ? (const __nv_bfloat16*)post_w.data_ptr() : nullptr;
  p.pre_w = pre_w.numel() ? (const __nv_bfloat16*)pre_w.data_ptr() : nullptr;
  p.alpha = alpha.numel() ? alpha.data_ptr<float>() : nullptr;
  p.beta = beta.numel() ? beta.data_ptr<float>() : nullptr;
  p.gate_channel_wise = alpha.numel() == SB_H ? 1 : 0;
  p.post_eps = (float)post_eps; p.pre_eps = (float)pre_eps;
  auto s = c10::cuda::getCurrentCUDAStream();
  const bool r = p.router != nullptr, isc = (input_scale & 1) != 0;
  if (r && isc) sbbm_go<true, true>(p, mom.data_ptr<float>(), (int)P, T, s);
  else if (r) sbbm_go<true, false>(p, mom.data_ptr<float>(), (int)P, T, s);
  else if (isc) sbbm_go<false, true>(p, mom.data_ptr<float>(), (int)P, T, s);
  else sbbm_go<false, false>(p, mom.data_ptr<float>(), (int)P, T, s);
}
'''
SBBM_DECL = ("void sbbm_run(torch::Tensor branch, torch::Tensor stream, torch::Tensor out, torch::Tensor new_stream, "
             "torch::Tensor router, torch::Tensor post_w, torch::Tensor pre_w, torch::Tensor alpha, torch::Tensor beta, "
             "double post_eps, double pre_eps, int64_t input_scale, torch::Tensor mom, int64_t P);")


def build(layer):
    from torch.utils.cpp_extension import load_inline
    gns = VL.layer_extract(layer, ("GD_", "_GD", "_gd"), ("K2A_BUCKETS", "_k2a_bucket_of", "SU1_K3_TRIG_AT"))
    sns = VL.layer_extract(layer, ("SBB1_",))
    cuda, cpp = gdm_patch(gns["GD_CUDA_SRC"], gns["GD_CPP_SRC"])
    keys = []
    for T, rt in O_ROUTES.items():
        k = gns["_gd_key"](8192, 1024, rt)
        assert k is not None, rt
        for out in (0, 2):
            kk = (out,) + tuple(k[1:])
            if kk not in keys:
                keys.append(kk)
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
    cuda = cuda.replace("@INST@", inst).replace("@INFO@", info)
    t0 = time.time()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    gmod = load_inline(name=f"vdsm_gd_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                       functions=["gd_run", "gd_info", "gd_unpack", "gd_run_mom"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"], verbose=False)
    scu = sns["SBB1_CU"] + sns["SBB1_HOST"] + SBBM_CU
    sdecl = sns["SBB1_DECL"] + " " + SBBM_DECL
    tag2 = hashlib.sha1((sdecl + scu).encode()).hexdigest()[:10]
    smod = load_inline(name=f"vdsm_sbb_{tag2}", cpp_sources=sdecl, cuda_sources=scu, functions=["sbb1_run", "sbbm_run"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"[mom] built GD ({len(keys)} instances) + SBB modules in {time.time() - t0:.1f}s", flush=True)
    return gns, gmod, keys, smod


def main():
    ts_ = [int(v) for v in VL.arg("ts", "1,8,16,32").split(",")]
    reps = int(VL.arg("reps", "21"))
    rounds = int(VL.arg("rounds", "6"))
    layer = VL.arg("layer", str(ROOT / "kernels" / "lab" / "k3_slim_5_ab.py"))
    out_path = Path(VL.arg("out", str(ROOT / "box_out" / "vdsm1" / "lab.json")))
    gns, gmod, keys, smod = build(layer)
    for i, k in enumerate(keys):
        print(f"[mom] gd inst {i} {k}: regs/local/occ/smem {list(gmod.gd_info(i))}", flush=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)
    Wo = (torch.randn((H, 1024), generator=gen, device=dev) / math.sqrt(1024)).to(torch.bfloat16)
    gate = 0.3 * torch.randn((H,), generator=gen, device=dev)
    alpha = (torch.sigmoid(gate) * 0.5 + 0.7).float().contiguous()       # channel-wise residual coefficients (shape class)
    beta = (torch.sigmoid(-gate) * 0.6 + 0.2).float().contiguous()
    pre_w = (1.0 + 0.1 * torch.randn((H,), generator=gen, device=dev)).to(torch.bfloat16).contiguous()
    e = torch.empty(0, device=dev)
    post_eps, pre_eps = 1e-6, 1e-6
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    results = {"rows": []}
    packs = {}
    for T in ts_:
        rt = O_ROUTES[T]
        mt, wr, wk, ku, st_, s, xp = rt
        k0 = gns["_gd_key"](8192, 1024, rt)
        i0, i2 = keys.index((0,) + tuple(k0[1:])), keys.index((2,) + tuple(k0[1:]))
        pk_key = (wr, wk, ku, s)
        if pk_key not in packs:
            base = gns["_gd_window_base"](Wo)
            packs[pk_key] = gns["_gd_pack"](Wo, wr, wk, ku, s, base)
        pk = packs[pk_key]
        P = pk["nbx"]
        x = (torch.randn((T, 1024), generator=gen, device=dev)).to(torch.bfloat16)
        stream = torch.randn((T, H), generator=gen, device=dev)
        o0 = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        o2 = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
        mom = torch.zeros((T * P * 4,), dtype=torch.float32, device=dev)
        rd = T <= 8
        outs = {nm: (torch.empty((T, H), dtype=torch.bfloat16, device=dev), torch.empty((T, H), dtype=torch.float32, device=dev),
                     torch.empty((T, H), dtype=torch.float32, device=dev) if rd else e) for nm in ("s1", "sm", "c0", "c2")}
        mflags = 4 | 8   # gated, channel-wise; no post weight (the reference's attention boundary)

        def gd0(x=x, o=o0, pdl=2):
            gmod.gd_run(i0, pk["w"], pk["esc"], x, o, T, 1024, H, pk["tb"], pk["nbx"], 1, pdl)

        def gd2(x=x, o=o2, pdl=2):
            gmod.gd_run_mom(i2, pk["w"], pk["esc"], x, o, T, 1024, H, pk["tb"], pk["nbx"], 1, pdl, stream, alpha, beta, e, mom,
                            mflags)

        def sb1(o=o0, nm="s1"):
            a, b, c = outs[nm]
            smod.sbb1_run(o, stream, a, b, c, e, pre_w, alpha, beta, post_eps, pre_eps, 0)

        def sbm(o=o2, nm="sm"):
            a, b, c = outs[nm]
            smod.sbbm_run(o, stream, a, b, c, e, pre_w, alpha, beta, post_eps, pre_eps, 0, mom, P)

        arms = {"gd0": gd0, "gd2": gd2, "sbb1": sb1, "sbbm": sbm,
                "chain0": lambda: (gd0(), sb1(o0, "c0")), "chain2": lambda: (gd2(), sbm(o2, "c2"))}
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        same_o = bool(torch.equal(o0, o2))
        # reference sums (FP64) of the four moments from o0 for a direct check of the partial records
        v = o0.double()
        r = alpha.double() * stream.double()
        sb = beta.double() * v
        ref_m = torch.stack([(v * v).sum(1), (r * r).sum(1), (r * sb).sum(1), (sb * sb).sum(1)], 1)
        ref_a = torch.stack([(v * v).sum(1), (r * r).sum(1), (r * sb).abs().sum(1), (sb * sb).sum(1)], 1)
        got_m = mom.view(T, P, 4).double().sum(1)
        mrel = float(((got_m - ref_m).abs() / ref_a.clamp(min=1e-30)).max())

        def rel(a, b):
            return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))
        chk = {"o_bitwise": same_o, "moments_rel_max": mrel,
               "out_rel": rel(outs["sm"][0], outs["s1"][0]), "out_mismatch_frac": float((outs["sm"][0] != outs["s1"][0]).float().mean()),
               "stream_rel": rel(outs["sm"][1], outs["s1"][1]),
               "router_rel": rel(outs["sm"][2], outs["s1"][2]) if rd else 0.0,
               "chain_out_rel": rel(outs["c2"][0], outs["c0"][0]), "chain_stream_rel": rel(outs["c2"][1], outs["c0"][1])}
        print(f"[mom] T{T}: P {P}; {json.dumps({k2: (round(v2, 9) if isinstance(v2, float) else v2) for k2, v2 in chk.items()})}",
              flush=True)
        names = list(arms)
        graphs = [VL.graph_of(arms[nm]) for nm in names]
        warm = lambda: torch.sum(x, dim=0)  # noqa: E731  (GD o's input is the predecessor's fresh output)
        meds = {nm: [] for nm in names}
        for rd_ in range(rounds):
            tt = VL.time_many(graphs, flush, sink, reps, warm)
            for nm, t_ in zip(names, tt):
                meds[nm].append(t_)
        row = {"T": T, "P": P, "chk": chk, "us": {nm: statistics.median(meds[nm]) for nm in names}, "rounds": meds}
        u = row["us"]
        row["gd_epilogue_cost_us"] = u["gd2"] - u["gd0"]
        row["sbb_saving_us"] = u["sbb1"] - u["sbbm"]
        row["chain_saving_us"] = u["chain0"] - u["chain2"]
        results["rows"].append(row)
        print(f"[mom] T{T}: raw us gd0 {u['gd0']:.2f} gd2 {u['gd2']:.2f} (epilogue +{row['gd_epilogue_cost_us']:.2f}) | sbb1 "
              f"{u['sbb1']:.2f} sbbm {u['sbbm']:.2f} (saves {row['sbb_saving_us']:.2f}) | chain0 {u['chain0']:.2f} chain2 "
              f"{u['chain2']:.2f} (saves {row['chain_saving_us']:.2f})", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
    print("[mom] done", flush=True)


if __name__ == "__main__":
    main()

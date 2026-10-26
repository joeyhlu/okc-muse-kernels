"""k2a_v45_patches.py: K2A v4.5 = post-expert norm + weighted sum fused into K2 at small T (vid 0 = v3.3 NC1 for T <= 8,
vid 3 = v3.6 HK2 for T 9..32), K3 not launched there; exact text patches against kernels/muse_decode_layer/k3_v28.py (the
K2A source embedded there) that also apply to kernels/muse_moe_w4a16/k2a_v37.py's KERNEL_SRC_V3 (the lab uses them).

Mechanism (no new buffers): P.h is all-zero during K2 (K1's finalizers zero it; K2 never uses it), so its first 2 * M3 + 2
floats serve as int counters: tc[t] = W2 stage contributions to token t's routes (target 512 = 32 row blocks x 16 stages per
local route), cl[t] = claim flag, [2 M3] tokens done, [2 M3 + 1] CTAs done. After its gemm, each K2 CTA syncs, fences
(release, as the shipped split-item hand-off) and adds the stage counts of every (item piece, route) of its range. Then
every CTA claims ready tokens (parallel scan of tc against the per-token target from meta, CAS on cl; bounded wait with
err[0] / err[3], no trap) and runs K3's per-token loop (4 column quarters, K3's arithmetic and order, so the output equals
K3's given y / ss) until all M tokens are done; the last CTA out zeroes the counters (h is zero again for the next K1).
vid 2 / vid 5 (and vid 1, 4) keep K3: launch3 skips K3 only for NC 1 non-HOTCAP, launch36 always (only vid 3 is HK2).

    python -I tools/k2a_v45_patches.py [k3_v28.py] [out.py]   # asserts every `old` occurs exactly once, applies all to a copy
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v28.py"

FP_HELPERS = r"""// v4.5: post-expert norm + weighted sum in K2 (small T). Int counters in P.h (zero during K2): tc[t], cl[4 t + q], [5 M3] units
// done, [5 M3 + 1] CTAs done. fp_count: this CTA's W2 stage contributions of segment [x0, x1) (all threads, after a sync).
__device__ __forceinline__ void fp_count(const Params& P, const Route& S, const int16_t* s_list, const int* sb, int TP,
                                         int x0, int x1) {
  int* tc = reinterpret_cast<int*>(P.h);
  int e = 0;
  for (int x = x0; x < x1;) {
    while (sb[e + 1] <= x) ++e;
    const int np = S.np[e];
    const int l = x - sb[e];
    const int item = l >> 4, s0 = l & 15;
    const int s1 = min(16, s0 + (x1 - x));
    const int j = item % np;
    const int ntok = min(TP, S.cnt[e] - j * TP);
    const int c0 = S.off[e] + j * TP;
    for (int i = threadIdx.x; i < ntok; i += blockDim.x) atomicAdd(tc + (((int)s_list[c0 + i]) >> 3), s1 - s0);
    x += s1 - s0;
  }
}
// every K2 CTA after its contributions: claim ready tokens and compute their output rows (K3's loop), until all are done
__device__ __forceinline__ void fp_post(const Params& P) {
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  __shared__ int s_tgt[M3];
  __shared__ int s_pick, s_stat;
  int* tc = reinterpret_cast<int*>(P.h);
  int* cl = tc + M3;        // 4 * M3 unit claims
  int* ctl = tc + 5 * M3;   // [0] units done, [1] CTAs done
  const int tid = threadIdx.x, M = P.M;
  for (int t = tid; t < M; t += blockDim.x) {
    int nl = 0;
#pragma unroll
    for (int s = 0; s < 8; ++s) nl += (__ldcg(P.meta + MH + t * 8 + s) >= 0) ? 1 : 0;
    s_tgt[t] = 512 * nl;
  }
  __syncthreads();
  const long long t0 = clock64();
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  const int U = 4 * M;   // claim units = k3_kernel's items (token, column quarter)
  while (true) {
    if (tid == 0) { s_pick = 0x7fffffff; s_stat = (__ldcg(ctl) >= U) ? 1 : 0; }
    __syncthreads();
    if (s_stat == 0) {
      for (int u = tid; u < U; u += blockDim.x) {
        const int k = (u - (int)blockIdx.x % U + U) % U;   // rotated order: CTAs start on different units
        if (__ldcg(cl + u) == 0 && __ldcg(tc + (u >> 2)) >= s_tgt[u >> 2]) atomicMin(&s_pick, k);
      }
    }
    __syncthreads();
    if (tid == 0 && s_stat == 0) {
      if (s_pick == 0x7fffffff) {
        s_stat = (clock64() - t0 > WAIT_BUDGET) ? 3 : 2;
        if (s_stat == 3) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }
      } else {
        const int u = (s_pick + (int)blockIdx.x % U) % U;
        if (atomicCAS(cl + u, 0, 1) == 0) { __threadfence(); s_pick = u; s_stat = 4; } else { s_stat = 2; }
      }
    }
    __syncthreads();
    const int stat = s_stat, unit = s_pick;
    __syncthreads();   // everyone has read s_stat / s_pick before thread 0 rewrites them
    if (stat == 1 || stat == 3) break;
    if (stat == 2) { __nanosleep(1000); continue; }   // nothing ready: back off ~1 us (keeps L2 polling light)
    const int tok = unit >> 2;
    for (int q4 = unit & 3; q4 <= (unit & 3); ++q4) {   // k3_kernel's item (tok, q4)
      if (tid < 8) {
        const int c = __ldcg(P.meta + MH + tok * 8 + tid);
        const float wt = __ldg(P.wts + tok * 8 + tid);
        s_fc[tid] = c;
        s_fr[tid] = (c >= 0) ? rsqrtf(__ldcg(P.ss + c) * (1.f / 4096.f) + 1e-8f) * wt : 0.f;
      }
      __syncthreads();
      if (tid < 256) {
        const int col = q4 * 1024 + 4 * tid;
        float4 v[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const int c = s_fc[s];
          v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
        }
        const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const float r = s_fr[s];
          a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
          a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
        }
        uint2 o;
        o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
        *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
      }
      __syncthreads();
    }
    if (tid == 0) atomicAdd(ctl, 1);
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(ctl + 1, 1) == (int)gridDim.x - 1) {   // last CTA out: h is zero again for the next K1
      for (int t = 0; t < M; ++t) atomicExch(tc + t, 0);
      for (int u = 0; u < 4 * M; ++u) atomicExch(cl + u, 0);
      atomicExch(ctl, 0);
      atomicExch(ctl + 1, 0);
    }
  }
}

"""

V0_CALL = "    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);\n"
V3_CALL = "  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);\n"
K3_LAUNCH = ("  if (mask & 4) {\n    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;\n"
             "    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));\n  }\n")

# (what, old, new): every `old` occurs exactly once in kernels/muse_decode_layer/k3_v28.py
PATCHES = [
    ("FP: helpers (before hot_item)", "// v3.3 warm-expert item (all of its stages s0..s1)",
     FP_HELPERS + "// v3.3 warm-expert item (all of its stages s0..s1)"),
    ("FP: vid 0 (NC 1) K2 counts + post", V0_CALL,
     V0_CALL + "    if constexpr (NC == 1) {   // v4.5: post-norm + weighted sum in K2 (K3 not launched for NC 1)\n"
     "      __syncthreads();\n      __threadfence();\n      int fb_, fe_;\n"
     "      my_range2(S.sb[32], blockIdx.x, gridDim.x, fb_, fe_);\n"
     "      fp_count(P, S, s_list, S.sb, 8 * NC, fb_, fe_);\n      fp_post(P);\n    }\n"),
    ("FP: vid 3 (HK2) K2 counts + post", V3_CALL,
     V3_CALL + "  __syncthreads();   // v4.5: post-norm + weighted sum in K2 (K3 not launched for HK2)\n  __threadfence();\n"
     "  for (int sg = 0; sg < 4; ++sg)\n"
     "    if (s_pr[3 * sg + 1] > s_pr[3 * sg])\n"
     "      fp_count(P, S, s_list, s_pr[3 * sg + 2] ? S.sbh : S.sbc, s_pr[3 * sg + 2] ? 8 * NCH : 8, s_pr[3 * sg], s_pr[3 * sg + 1]);\n"
     "  fp_post(P);\n"),
    ("FP: launch3 skips K3 for NC 1 non-HOTCAP",
     "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, p));\n  }\n" + K3_LAUNCH,
     "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, p));\n  }\n"
     + K3_LAUNCH.replace("  if (mask & 4) {", "  if ((mask & 4) && (HOTCAP || NC != 1)) {   // v4.5: NC 1 does it in K2")),
    ("FP: launch36 skips K3",
     "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));\n  }\n" + K3_LAUNCH,
     "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));\n  }\n"
     + K3_LAUNCH.replace("  if (mask & 4) {", "  if (false && (mask & 4)) {   // v4.5: HK2 does it in K2")),
]
KERNEL_PATCHES = PATCHES


def apply(text, patches=PATCHES):
    for what, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"k2a_v45 patch {what!r}: `old` occurs {n} times (want 1)")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v28_k2a45_check.py")
    text = src.read_text(encoding="utf-8")
    for what, old, _ in PATCHES:
        print(f"  {text.count(old)}x  {what}")
    patched = apply(text)
    for need in ("V3_SELF_TEST_CASES", "K2A33_SELF_TEST", "self_test"):
        if need in text:
            print(f"  self-test table {need!r}: {'kept' if patched.count(need) == text.count(need) else 'CHANGED'}")
    out.write_text(patched, encoding="utf-8")
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

"""gen_k3_ntl.py: kernels/lab/k3_ntl_<n>.py = a GC layer file (default k3_gc_16) + NTL ("no tail"): the routing tail
(v8::tail_kernel2, 4.1-5.1 us + 0.4-0.7 us idle on the serial chain) is not launched at T <= 16; its two halves move to
the two ends of the chain without any single-CTA serial step and without any counter wait:

* top-8 / weights / local ids: every VMS CTA replays tail_kernel2's routing warp (FP32 partial slices summed in slice order
  + 0 for the absent lb, sigmoid via expf, + bias, orderable-key / ballot top-8 with the lowest index on ties, weights
  s / (sum + deps) * rscale, local id from local_start) for all T tokens itself, after its griddepcontrol.wait (which
  already guarantees that the producer grid completed and flushed), one token per compute warp (t = warp, warp + 8).
  CTA 0 also writes the global ids (the layer returns them) and, for the self-tests, tw / lid.
* pre-expert norm, variant B (exact rounding point): the pooler epilogue (GD T = 1 pooler, VF = 2 fused pooler + router at
  T = 2..8, GC cat at T = 9..16) adds each token's sum of bf16(p)^2 over its rows into an FP32 [T] buffer (smem atomics,
  then one red.add per CTA per token; p itself is written unchanged). Each VMS CTA computes s = rsqrtf(sum / 4096 + eps)
  per token and builds xn = bf16((p * s) * w) (tail_kernel2's formula and BF16 rounding point) for the x rows it loads.
  The only difference to tail2's xn is the FP32 summation order of sum(p^2) (assumption A24). VMS CTA 0 zeroes the
  [T] buffer after the end grid barrier (every CTA has read it before arriving), i.e. reset for the next call.
* T = 17..32 (VML) and T >= 128: unchanged (tail2 kept).

NTL_MODE[bucket] (read at call time): 0 = base (tail2 + VMS), 2 = live. prepare self-tests per bucket: producer sum(p^2)
vs torch, ids / weights / local ids bitwise vs tail_kernel2 on the same producer outputs, MoE output vs tail2 + VMS,
buffer reset; a failing bucket stays on tail2 (FALLBACK print). Compile-time flags only (VMS_NTL builds a second VMS
module; the base VMS module preprocesses back to its own source).

Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_ntl.py [--base=k3_gc_16] [--n=1]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_16')}.py"
OUT = ROOT / f"kernels/lab/k3_ntl_{N}.py"
TAG = f"k3-ntl-{N}"
NL = chr(10)

# ---------------------------------------------------------------------------------------------------------- GD (producers)
GD_PATCHES = [
    ("uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n};",
     "uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
     "  float* ntl_ss = nullptr;   // NTL: per-token sum of bf16(out)^2, red.add (pooler hosts only)\n};",
     "GD Args: ntl_ss"),
    ("  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n  constexpr int KUP = KU - CAT;",
     "  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n"
     "  constexpr bool GNT_HOST = S == 1 && NT == 256 && OUT == 0 && (CAT > 0 || VF == 2 || (MT == 1 && WR == 1 && WK == 8 && KU == 16));\n"
     "  constexpr int KUP = KU - CAT;",
     "GD kernel: GNT_HOST (the pooler instances)"),
    ("  __shared__ __align__(8) unsigned long long mbar;\n",
     "  __shared__ __align__(8) unsigned long long mbar;\n"
     "  __shared__ float gnt_red[GNT_HOST ? MT * 8 : 1];\n"
     "  if constexpr (GNT_HOST) {\n    if ((int)threadIdx.x < MT * 8) gnt_red[threadIdx.x] = 0.f;\n  }\n",
     "GD kernel: gnt_red smem, zeroed at entry (ordered by the epilogue's __syncthreads)"),
    ("    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n    if constexpr (CAT > 0) {",
     "    if constexpr (GNT_HOST) {\n"
     "      if (A.ntl_ss != nullptr) {   // NTL: sum over this CTA's rows of bf16(p)^2 per token, one red.add per token\n"
     "        for (int i = (int)threadIdx.x; i < NO; i += NT) {\n"
     "          const float v = cta_value(i);\n"
     "          store(i, v);\n"
     "          const int l = i & 31, c = (i >> 5) & 3, mt = (i >> 7) % MT;\n"
     "          const float q = __bfloat162float(__float2bfloat16_rn(v));\n"
     "          atomicAdd(&gnt_red[mt * 8 + 2 * (l & 3) + (c & 1)], q * q);\n"
     "        }\n"
     "        __syncthreads();\n"
     "        if ((int)threadIdx.x < MT * 8 && tok0 + (int)threadIdx.x < A.M)\n"
     "          asm volatile(\"red.global.add.f32 [%0], %1;\\n\" :: \"l\"(A.ntl_ss + tok0 + (int)threadIdx.x), \"f\"(gnt_red[threadIdx.x])\n"
     "                       : \"memory\");\n"
     "      } else {\n"
     "        for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
     "      }\n"
     "    } else {\n"
     "      for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
     "    }\n"
     "    if constexpr (CAT > 0) {",
     "GD S = 1 epilogue: pooler sum of squares"),
    ("}  // namespace gd\n\nusing namespace gd;\n",
     "}  // namespace gd\n\nusing namespace gd;\n"
     "static float* g_ntl_ss = nullptr;   // NTL: armed by gd_ntl_arm() right before ONE pooler launch, consumed by it\n"
     "void gd_ntl_arm_raw(void* ss) { g_ntl_ss = reinterpret_cast<float*>(ss); }\n",
     "GD host: g_ntl_ss"),
    ("  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n  switch (inst) {",
     "  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n"
     "  A.ntl_ss = g_ntl_ss; g_ntl_ss = nullptr;\n  switch (inst) {",
     "gd_run_raw consumes the arm"),
    ("  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n  const int nz = 1;",
     "  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n"
     "  A.ntl_ss = g_ntl_ss; g_ntl_ss = nullptr;\n  const int nz = 1;",
     "gdc_run_raw consumes the arm"),
    ("  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;\n",
     "  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;\n  A.ntl_ss = g_ntl_ss; g_ntl_ss = nullptr;\n",
     "gdr_run_raw consumes the arm"),
]

GD_CPP_ADD = r'''void gd_ntl_arm_raw(void* ss);
void gd_ntl_arm(torch::Tensor ss) {
  TORCH_CHECK(ss.scalar_type() == at::kFloat && ss.is_cuda() && ss.is_contiguous() && ss.numel() >= 32, "gd_ntl_arm: f32 [>= 32]");
  gd_ntl_arm_raw(ss.data_ptr());
}
'''

# ------------------------------------------------------------------------------------------------------------ VMS (consumer)
VMS_ROUTE_PRE = r'''#if VMS_NTL
  const float4* nbp = reinterpret_cast<const float4*>(p.rbias) + lane * 2;
  const float4 nb0 = __ldg(nbp), nb1 = __ldg(nbp + 1);
#endif
'''

VMS_ROUTE_FN = r'''#if VMS_NTL
#if VMS_NTL_NI
__device__ __noinline__
#else
__device__ __forceinline__
#endif
void nt_route(const float* plg, long long sst, long long lld, const int* lstart, const float* ssum, int m, float mult,
              float rscale, float deps, float xeps, int* tid_out, float* tw_out, int* lid_out, Smem& S, const int warp,
              const int lane, const int tid, const int c, const float4 nb0, const float4 nb1) {
  if (warp < NW) {   // tail_kernel2's routing warp for tokens t = warp, warp + 8 (same arithmetic, same order)
    const int nls = __ldcg(lstart);
    const float ssv = tid < m ? __ldcg(ssum + tid) : 0.f;   // issued with the partials, not after the routing
    for (int t = warp; t < m; t += NW) {
      float4 qa[8][2];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const float4* pq = reinterpret_cast<const float4*>(plg + q * sst + t * lld) + lane * 2;
        qa[q][0] = __ldcg(pq);
        qa[q][1] = __ldcg(pq + 1);
      }
      float4 a0 = qa[0][0], a1 = qa[0][1];
#pragma unroll
      for (int q = 1; q < 8; ++q) {
        a0.x += qa[q][0].x; a0.y += qa[q][0].y; a0.z += qa[q][0].z; a0.w += qa[q][0].w;
        a1.x += qa[q][1].x; a1.y += qa[q][1].y; a1.z += qa[q][1].z; a1.w += qa[q][1].w;
      }
      a0.x += 0.f; a0.y += 0.f; a0.z += 0.f; a0.w += 0.f; a1.x += 0.f; a1.y += 0.f; a1.z += 0.f; a1.w += 0.f;
      const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float bs[8] = {nb0.x, nb0.y, nb0.z, nb0.w, nb1.x, nb1.y, nb1.z, nb1.w};
      float s[8], b[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s[j] = 1.f / (1.f + expf(-lg[j] * mult));
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
        const float inv = rscale / (sum + deps);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const float wv = sel_s[k] * inv;
          const int rel = ((sel_id[k] - nls) % 256 + 256) % 256, li = rel < 32 ? rel : -1;
          S.rtw[t * 8 + k] = wv;
          S.rid[t * 8 + k] = li;
          if (c == 0) { tid_out[t * 8 + k] = sel_id[k]; tw_out[t * 8 + k] = wv; lid_out[t * 8 + k] = li; }
        }
      }
    }
    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);
  }
}
#endif

'''

VMS_ROUTE_FR = r"""#if VMS_NTL && VMS_NTL_FR
// FR (R-ntl-1, local marks): the routing replay sits on the critical path before the first weight issue (~6.4k cycles
// at T=1 on the 3070: loads ~2.8k, sigmoid ~1k, top-k ~2.1k; T=16 = 2 serial rounds per warp). FR: (1) every thread
// of the CTA builds s = sigmoid(sum of the 8 partials) for 4 experts at once (loads up front; smem scratch in ring slot
// 0, free until the first issue); (2) the top-k warp pre-sorts its 8 (b, j) per lane (19-CAS network, ties -> lower j =
// the scan's choice), so each of the 8 rounds is key -> redux -> ballot -> shift (no 8-wide scan; the id / s shuffles
// leave the chain); (3) a warp with 2 tokens (T > 8) runs both selections interleaved. Same arithmetic, same order,
// same tie rules as tail_kernel2: bitwise ids / tw / lid.
__device__ __forceinline__ bool nt_better(float b1, int j1, float b2, int j2) { return b1 > b2 || (b1 == b2 && j1 < j2); }
template <int NU>
__device__ __forceinline__ void nt_topk(const float* scr, const int t0, const int nls, const float rscale, const float deps,
                                        int* tid_out, float* tw_out, int* lid_out, Smem& S, const int lane, const int c,
                                        const float4 nb0, const float4 nb1) {
  const float bs[8] = {nb0.x, nb0.y, nb0.z, nb0.w, nb1.x, nb1.y, nb1.z, nb1.w};
  constexpr int NWK[19][2] = {{0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}, {0, 1}, {2, 3},
                              {4, 5}, {6, 7}, {2, 4}, {3, 5}, {1, 4}, {3, 6}, {1, 2}, {3, 4}, {5, 6}};
  float hb[NU][8], hs[NU][8];
  int hj[NU][8];
#pragma unroll
  for (int u = 0; u < NU; ++u) {
    const float4* sp = reinterpret_cast<const float4*>(scr + (t0 + u * NW) * 256) + lane * 2;
    const float4 s0 = sp[0], s1 = sp[1];
    const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) { hs[u][j] = sv[j]; hb[u][j] = sv[j] + bs[j]; hj[u][j] = j; }
#pragma unroll
    for (int x = 0; x < 19; ++x) {
      const int i = NWK[x][0], k = NWK[x][1];
      const bool sw = nt_better(hb[u][k], hj[u][k], hb[u][i], hj[u][i]);
      const float tb = hb[u][i], ts = hs[u][i];
      const int tj = hj[u][i];
      hb[u][i] = sw ? hb[u][k] : tb; hs[u][i] = sw ? hs[u][k] : ts; hj[u][i] = sw ? hj[u][k] : tj;
      hb[u][k] = sw ? tb : hb[u][k]; hs[u][k] = sw ? ts : hs[u][k]; hj[u][k] = sw ? tj : hj[u][k];
    }
  }
  int ow[NU][8], rj[NU][8];
  float rsv[NU][8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
#pragma unroll
    for (int u = 0; u < NU; ++u) {
      const unsigned uu = __float_as_uint(hb[u][0] + 0.f);
      const unsigned key = (uu & 0x80000000u) ? ~uu : (uu | 0x80000000u);
      const unsigned mx = __reduce_max_sync(0xffffffffu, key);
      const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
      ow[u][k] = owner; rj[u][k] = hj[u][0]; rsv[u][k] = hs[u][0];
      const bool me = lane == owner;
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        hb[u][j] = me ? hb[u][j + 1] : hb[u][j]; hs[u][j] = me ? hs[u][j + 1] : hs[u][j]; hj[u][j] = me ? hj[u][j + 1] : hj[u][j];
      }
      hb[u][7] = me ? -INFINITY : hb[u][7];
    }
  }
#pragma unroll
  for (int u = 0; u < NU; ++u) {
    const int t = t0 + u * NW;
    int sel_id[8];
    float sel_s[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      sel_id[k] = __shfl_sync(0xffffffffu, lane * 8 + rj[u][k], ow[u][k]);
      sel_s[k] = __shfl_sync(0xffffffffu, rsv[u][k], ow[u][k]);
    }
    if (lane == 0) {
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) sum += sel_s[i];
      const float inv = rscale / (sum + deps);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float wv = sel_s[k] * inv;
        const int rel = ((sel_id[k] - nls) % 256 + 256) % 256, li = rel < 32 ? rel : -1;
        S.rtw[t * 8 + k] = wv;
        S.rid[t * 8 + k] = li;
        if (c == 0) { tid_out[t * 8 + k] = sel_id[k]; tw_out[t * 8 + k] = wv; lid_out[t * 8 + k] = li; }
      }
    }
  }
}
__device__ __forceinline__ void nt_route_fr(const float* plg, long long sst, long long lld, const int* lstart, const float* ssum,
                                            int m, float mult, float rscale, float deps, float xeps, int* tid_out,
                                            float* tw_out, int* lid_out, Smem& S, float* scr, const int warp, const int lane,
                                            const int tid, const int c, const float4 nb0, const float4 nb1) {
  int nls = 0;
  float ssv = 0.f;
  if (warp < NW) { nls = __ldcg(lstart); ssv = tid < m ? __ldcg(ssum + tid) : 0.f; }
  const int nit = m * 64;   // items (token, 4 experts): <= 1024 at T = 16 -> <= 2 passes of 2 x NTL
  for (int b0 = 0; b0 < nit; b0 += 2 * NTL) {
    float4 v[2][8];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int it = b0 + k * NTL + tid;
      if (it < nit) {
        const float* row = plg + (it >> 6) * lld;
#pragma unroll
        for (int q = 0; q < 8; ++q) v[k][q] = __ldcg(reinterpret_cast<const float4*>(row + q * sst) + (it & 63));
      }
    }
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int it = b0 + k * NTL + tid;
      if (it < nit) {
        float4 a = v[k][0];
#pragma unroll
        for (int q = 1; q < 8; ++q) { a.x += v[k][q].x; a.y += v[k][q].y; a.z += v[k][q].z; a.w += v[k][q].w; }
        a.x += 0.f; a.y += 0.f; a.z += 0.f; a.w += 0.f;
        float4 o;
        o.x = 1.f / (1.f + expf(-a.x * mult));
        o.y = 1.f / (1.f + expf(-a.y * mult));
        o.z = 1.f / (1.f + expf(-a.z * mult));
        o.w = 1.f / (1.f + expf(-a.w * mult));
        reinterpret_cast<float4*>(scr)[it] = o;
      }
    }
  }
  __syncthreads();
  if (warp < NW) {
    if (warp < m) {
      if (warp + NW < m) nt_topk<2>(scr, warp, nls, rscale, deps, tid_out, tw_out, lid_out, S, lane, c, nb0, nb1);
      else nt_topk<1>(scr, warp, nls, rscale, deps, tid_out, tw_out, lid_out, S, lane, c, nb0, nb1);
    }
    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // scratch (generic) before the ring's first TMA write
#endif
}
#endif

"""

VMS_ROUTE_POST = r'''#if VMS_NTL && VMS_NTL_FR
  nt_route_fr(p.lg, p.sst, p.lld, p.lstart, p.ssum, p.m, p.mult, p.rscale, p.deps, p.xeps, p.tid_out, p.tw_out,
              p.lid_out, S, reinterpret_cast<float*>(ring), warp, lane, tid, c, nb0, nb1);
  __syncthreads();
#elif VMS_NTL
  nt_route(p.lg, p.sst, p.lld, p.lstart, p.ssum, p.m, p.mult, p.rscale, p.deps, p.xeps, p.tid_out, p.tw_out,
           p.lid_out, S, warp, lane, tid, c, nb0, nb1);
  __syncthreads();
#endif
'''

VMS_XN = r'''#if VMS_NTL && VMS_NTL_XN == 1
          if (!PB) {   // xn = bf16((p * s) * w), tail_kernel2's formula, for the rows this warp gathers
            const float rs = S.rs[S.ent[st + g] >> 3];
            const uint4* ws = reinterpret_cast<const uint4*>(p.xw) + warp * 64 + 2 * t;
#pragma unroll
            for (int cc = 0; cc < 8; ++cc) {
              xr[2 * cc] = nt_xn(xr[2 * cc], __ldg(ws + cc * 8), rs);
              xr[2 * cc + 1] = nt_xn(xr[2 * cc + 1], __ldg(ws + cc * 8 + 1), rs);
            }
          }
#endif
'''

VMS_XN_MARK = r'''#if VMS_NTL && VMS_NTL_XN == 2
          nt_fresh = !PB;
          if (!PB) nt_rs = S.rs[S.ent[st + g] >> 3];
#endif
'''

VMS_XN_LATE = r'''#if VMS_NTL && VMS_NTL_XN == 2
      if (nt_fresh) {   // xn = bf16((p * s) * w) once per job, after the row loads had the weight wait to land
        nt_fresh = false;
        const uint4* ws = reinterpret_cast<const uint4*>(p.xw) + warp * 64 + 2 * t;
#pragma unroll
        for (int cc = 0; cc < 8; ++cc) {
          xr[2 * cc] = nt_xn(xr[2 * cc], __ldg(ws + cc * 8), nt_rs);
          xr[2 * cc + 1] = nt_xn(xr[2 * cc + 1], __ldg(ws + cc * 8 + 1), nt_rs);
        }
      }
#endif
'''

VMS_HELPER = r'''#if VMS_NTL
__device__ __forceinline__ uint4 nt_xn(const uint4 x, const uint4 w, float rs) {
  const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&x);
  const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&w);
  __align__(16) __nv_bfloat16 o[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = __float2bfloat16((__bfloat162float(xb[i]) * rs) * __bfloat162float(wb[i]));
  return *reinterpret_cast<const uint4*>(o);
}
#define VMS_IDS(i) S.rid[i]
#define VMS_TW(i) S.rtw[i]
#else
#define VMS_IDS(i) __ldcg(p.ids + (i))
#define VMS_TW(i) __ldg(p.tw + (i))
#endif

'''

VMS_RUN_NTL = r'''#if VMS_NTL
void run_ntl(torch::Tensor x, torch::Tensor lg, torch::Tensor rbias, torch::Tensor lstart, torch::Tensor ssum, torch::Tensor xw,
             torch::Tensor tid_out, torch::Tensor tw_out, torch::Tensor lid_out, torch::Tensor wA, torch::Tensor wB,
             torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss,
             torch::Tensor ctr, torch::Tensor out, torch::Tensor marks, int64_t m, double eps, double xeps, double mult,
             double rscale, double deps, int64_t pdl) {
  TORCH_CHECK(m >= 1 && m <= 16 && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && x.size(0) == m &&
              x.size(1) == 4096 && reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && lg.scalar_type() == at::kFloat &&
              lg.dim() == 3 && lg.size(0) == 8 && lg.size(1) == m && lg.size(2) == 256 && lg.stride(2) == 1 &&
              lg.stride(1) % 4 == 0 && lg.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(lg.data_ptr()) % 16 == 0 &&
              rbias.scalar_type() == at::kFloat && rbias.numel() == 256 && reinterpret_cast<uintptr_t>(rbias.data_ptr()) % 16 == 0 &&
              lstart.scalar_type() == at::kInt && lstart.numel() >= 1 && ssum.scalar_type() == at::kFloat && ssum.numel() >= m &&
              xw.scalar_type() == at::kBFloat16 && xw.numel() == 4096 && reinterpret_cast<uintptr_t>(xw.data_ptr()) % 16 == 0 &&
              tid_out.scalar_type() == at::kInt && tid_out.numel() >= m * 8 && tw_out.scalar_type() == at::kFloat &&
              tw_out.numel() >= m * 8 && lid_out.scalar_type() == at::kInt && lid_out.numel() >= m * 8,
              "vms run_ntl: x [m, 4096] bf16, lg [8, m, 256] f32, bias / local_start / ssum / xw / outs");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vms::vms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vms::SMEM_BYTES));
    attr = true;
  }
  vms::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.ids = tid_out.data_ptr<int>(); p.tw = tw_out.data_ptr<float>();
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = 0;
  p.lg = lg.data_ptr<float>(); p.sst = lg.stride(0); p.lld = lg.stride(1); p.rbias = rbias.data_ptr<float>();
  p.lstart = lstart.data_ptr<int>(); p.ssum = ssum.data_ptr<float>(); p.xw = reinterpret_cast<const __nv_bfloat16*>(xw.data_ptr());
  p.tid_out = tid_out.data_ptr<int>(); p.tw_out = tw_out.data_ptr<float>(); p.lid_out = lid_out.data_ptr<int>();
  p.mult = (float)mult; p.rscale = (float)rscale; p.deps = (float)deps; p.xeps = (float)xeps;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vms::G); lc.blockDim = dim3(vms::NTL); lc.dynamicSmemBytes = vms::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vms::vms_kernel, p));
}
#endif

'''

VMS_PATCHES = [
    ("#ifndef VMS_G\n#define VMS_G 132\n#endif\n",
     "#ifndef VMS_G\n#define VMS_G 132\n#endif\n#ifndef VMS_NTL\n#define VMS_NTL 0\n#endif\n#ifndef VMS_NTL_NI\n#define VMS_NTL_NI 1\n#endif\n"
     "#ifndef VMS_NTL_XN\n#define VMS_NTL_XN 2\n#endif\n#ifndef VMS_NTL_FR\n#define VMS_NTL_FR " + ARGS.get("fr", "1") + "\n#endif\n", "VMS_NTL defaults (NI: noinline routing, XN: 0 none / 1 at job switch / 2 after the weight wait)"),
    ("  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n};",
     "  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;\n#if VMS_NTL\n"
     "  const float* lg; const float* rbias; const int* lstart; float* ssum; const __nv_bfloat16* xw;\n"
     "  int* tid_out; float* tw_out; int* lid_out; long long sst, lld; float mult, rscale, deps, xeps;\n#endif\n};",
     "Params: NTL fields"),
    ("  float4 red[NRB][NW][32];\n};",
     "  float4 red[NRB][NW][32];\n#if VMS_NTL\n  int rid[MAXP];\n  float rtw[MAXP];\n  float rs[16];\n#endif\n};",
     "Smem: rid / rtw / rs"),
    ("__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {",
     VMS_HELPER + VMS_ROUTE_FN + VMS_ROUTE_FR + "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {", "nt_xn + VMS_IDS / VMS_TW"),
    ("#if __CUDA_ARCH__ >= 900\n  asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");\n#endif\n  if (warp == 0) {\n"
     "    const int m8 = p.m * 8;",
     VMS_ROUTE_PRE + "#if __CUDA_ARCH__ >= 900\n  asm volatile(\"griddepcontrol.wait;\\n\" ::: \"memory\");\n#endif\n" + VMS_ROUTE_POST +
     "  if (warp == 0) {\n    const int m8 = p.m * 8;",
     "routing after the PDL wait"),
    ("idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255;",
     "idv[r] = i < m8 ? ((VMS_IDS(i) - p.lo) & 255) : 255;", "job table reads the CTA's own local ids"),
    ("            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);\n          }\n",
     "            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);\n          }\n" + VMS_XN + VMS_XN_MARK,
     "phase-A x rows: xn from p"),
    ("    int sl = 0; uint32_t par = 0;\n", "    int sl = 0; uint32_t par = 0;\n#if VMS_NTL && VMS_NTL_XN == 2\n    bool nt_fresh = false;\n    float nt_rs = 0.f;\n#endif\n", "nt_fresh / nt_rs"),
    ("      mbar_wait(&S.full[sl], par);\n      if (mk && tid == 0 && q == 0) mrow[2] = gt();\n",
     "      mbar_wait(&S.full[sl], par);\n" + VMS_XN_LATE + "      if (mk && tid == 0 && q == 0) mrow[2] = gt();\n", "late xn transform"),
    ("const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(", "const float cf = VMS_TW(tk * 8 + s) * rsqrtf(",
     "final pass: weights from smem"),
    ("    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;\n",
     "    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;\n#if VMS_NTL\n    if (tid < p.m) p.ssum[tid] = 0.f;\n#endif\n",
     "CTA 0 zeroes the [T] sum buffer after the end barrier"),
    ("std::vector<int64_t> attrs() {", VMS_RUN_NTL + "std::vector<int64_t> attrs() {", "host run_ntl"),
]

# ------------------------------------------------------------------------------------------------------------------- Python
PY_GLOBALS = r'''NTL_ENABLE = True
NTL_MODE = {1: 2, 8: 2, 16: 2, 32: 0}   # per GD bucket, read at call time: 0 tail2 + VMS (base), 2 live (no tail2), 1 inert (tail2 kept + VMS-NTL); diagnostics: 3 pooler sum only (tail2 + VMS), 4 VMS-NTL on unarmed sums (tail2 kept)
NTL_SELF_TEST_T = {1: (1,), 8: (8, 2, 5), 16: (16, 9, 13)}
_NTL = {"mod": None, "G": None, "ok": {}, "ss": None, "scr": None, "live": False, "armed": False, "pend": None, "seen": set()}
_NTL_VMS_CPP = r"""
void run_ntl(torch::Tensor x, torch::Tensor lg, torch::Tensor rbias, torch::Tensor lstart, torch::Tensor ssum, torch::Tensor xw,
             torch::Tensor tid_out, torch::Tensor tw_out, torch::Tensor lid_out, torch::Tensor wA, torch::Tensor wB,
             torch::Tensor gsA, torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss,
             torch::Tensor ctr, torch::Tensor out, torch::Tensor marks, int64_t m, double eps, double xeps, double mult,
             double rscale, double deps, int64_t pdl);
"""
'''

PY_FUNCS = r'''

def _ntl_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-DVMS_NTL=1", "-Xptxas=-warn-spills",
             "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + _NTL_VMS_CPP + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    _NTL["mod"] = load_inline(name=f"k3_vmsntl_{tag}", cpp_sources=VMS_CPP_SRC + _NTL_VMS_CPP, cuda_sources=VMS_CUDA_SRC,
                              functions=["run", "run_ntl", "attrs"], extra_cuda_cflags=flags)
    _NTL["G"] = G
    _K3_BUILD_S["ntl"] = time.time() - t1


def _ntl_vst(state):
    k = state.get("k2a33")
    st = k.get("st") if isinstance(k, dict) else None
    return st if (st is not None and st.get("vms_ok") and st.get("vms") is not None) else None


def _ntl_on(state, T):
    b = _gd_bucket(T)
    m = NTL_MODE.get(b, 0) if b is not None else 0
    ok = NTL_ENABLE and m in (1, 2, 3, 4) and _NTL["ok"].get(b) and VMS_ENABLE and T <= VMS_MAXT and _ntl_vst(state) is not None
    return m if ok else 0


def _ntl_arm(mod):
    mod.gd_ntl_arm(_NTL["ss"])
    _NTL["armed"] = True


def _ntl_vms(state, y, lg, ls, tid):
    vs, v8 = _ntl_vst(state)["vms"], state["v8"]
    m = y.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=y.device)
    _NTL["mod"].run_ntl(y, lg, v8["bias"], ls, _NTL["ss"], v8["nw"], tid, _NTL["scr"][0], _NTL["scr"][1], vs["wA"], vs["wB"],
                        vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"], out, vs["nomarks"], m,
                        K2A_POST_NORM_EPS, RMS_EPS, 0.1767766952966369, 1.0, 1e-15, 1)
    if m not in _NTL["seen"]:
        _NTL["seen"].add(m)
        print(f"[ntl] first live call T={m}", flush=True)
    return out


def _ntl_test(state, st, b, T, ls, seed):
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn((T, HIDDEN), generator=gen, device=dev)
    hb = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).to(torch.bfloat16).contiguous()
    h = hb.float().contiguous()
    lst = torch.tensor([ls], dtype=torch.int32, device=dev)
    part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
    _NTL["live"], _NTL["armed"], _NTL["pend"] = True, False, None
    if b == 1:
        if _gd_live(state, "pooler", T) is None or not _VF["r8"] or _SBB1["mod"] is None:
            return "GD pooler T=1 / lean router not live"
        y = _gd(state, "pooler", hb)
        _SBB1["mod"].vf_router32(h, state["router"], part, 0)
    elif b == 8:
        if _gd_live(state, "pooler", T) is None or not _VFR["ok"]:
            return "VF fused pooler not live"
        r = _vfr_gdr(state, hb, h, 2)
        if r is None:
            return "VF fused pooler did not run"
        y, part = r
    else:
        r = _gd_cat_live(state, T)
        if r is None:
            return "GC cat not live"
        y = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
        _ntl_arm(r["mod"])
        _gd_cat_run(r, hb, y, part, T, GD_CAT_PDL)
    _NTL["live"] = False
    if y is None or not _NTL["armed"]:
        return "producer not armed / did not run"
    _NTL["armed"] = False
    torch.cuda.synchronize()
    ss = _NTL["ss"].clone()
    ref_ss = y.float().pow(2).sum(-1)
    d = float(((ss[:T] - ref_ss).abs() / ref_ss.clamp_min(1e-30)).max())
    if not bool(torch.isfinite(ss).all()) or d > 1e-4 or bool((ss[T:] != 0).any()):
        return f"pooler sum(p^2) off (max rel {d:.2e}, tail nonzero {bool((ss[T:] != 0).any())})"
    xn, tw, tid, lid = _v8_tail(y, part, state, {"local_start": lst})
    vs = st["vms"]
    vs["ctr"][2] = 0
    ref = _vms_call(st, xn, lid, tw)
    tid2 = torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev)
    _NTL["scr"][0].fill_(-7.0)
    _NTL["scr"][1].fill_(-7)
    out = _ntl_vms(state, y, part, lst, tid2)
    torch.cuda.synchronize()
    tw2, lid2 = _NTL["scr"][0][:T * TOP_K].view(T, TOP_K), _NTL["scr"][1][:T * TOP_K].view(T, TOP_K)
    if not (torch.equal(tid2, tid) and torch.equal(lid2, lid) and torch.equal(tw2, tw)):
        return (f"routing differs (tid {int((tid2 != tid).sum())} lid {int((lid2 != lid).sum())} tw {int((tw2 != tw).sum())} "
                f"of {T * TOP_K})")
    if bool((_NTL["ss"] != 0).any()):
        return "sum buffer not reset by VMS CTA 0"
    errf = int(vs["ctr"][2])
    rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
    if errf or not bool(torch.isfinite(out.float()).all()) or rel > 2e-3:
        return f"MoE output vs tail2 + VMS rel {rel:.2e} err {errf}"
    return None


def _ntl_prepare(state):
    _NTL["ok"] = {}
    if not NTL_ENABLE:
        return
    try:
        _pt1_wait("ntl")
        if _NTL["mod"] is None:
            _ntl_build()
    except Exception as ex:
        print(f"[ntl] FALLBACK: VMS-NTL build raised {ex!r} -> tail2 everywhere", flush=True)
        return
    st = _ntl_vst(state)
    gm = _GD.get("mod")
    if (st is None or state.get("v8") is None or gm is None or not hasattr(gm, "gd_ntl_arm")
            or _NTL["G"] != _VMS.get("G")):
        print("[ntl] FALLBACK: VMS / v8 tail / GD module missing or SM count differs -> tail2 everywhere", flush=True)
        return
    dev = state["router"].device
    _NTL["ss"] = torch.zeros((32,), dtype=torch.float32, device=dev)
    _NTL["scr"] = (torch.zeros((16 * TOP_K,), dtype=torch.float32, device=dev), torch.zeros((16 * TOP_K,), dtype=torch.int32, device=dev))
    t1 = time.time()
    for b, Ts in NTL_SELF_TEST_T.items():
        msgs = []
        try:
            for T in Ts:
                for k, ls in enumerate((0, 224)):
                    for rep in range(2):
                        e = _ntl_test(state, st, b, T, ls, 9300 + 31 * T + 7 * k + rep)
                        if e is not None:
                            msgs.append(f"T={T} ls={ls} rep {rep}: {e}")
                            break
        except Exception as ex:
            msgs.append(f"EXC {ex!r}")
        finally:
            _NTL["live"], _NTL["armed"], _NTL["pend"] = False, False, None
            torch.cuda.synchronize()
            _NTL["ss"].zero_()
            st["vms"]["ctr"][2] = 0
        _NTL["ok"][b] = not msgs
        print(f"[ntl] self-test bucket {b}: {'ok (bitwise tid / lid / tw vs tail2, sum(p^2), reset, output)' if not msgs else 'FALLBACK ' + msgs[0]}",
              flush=True)
    print(f"[ntl] attrs {list(_NTL['mod'].attrs())} live buckets {[b for b, v in _NTL['ok'].items() if v and NTL_MODE.get(b) == 2]} "
          f"({time.time() - t1:.1f}s)", flush=True)
'''

PY_PATCHES = [
    ("\n_GD_CAT_SEEN = set()\n", "\n" + PY_GLOBALS + "_GD_CAT_SEEN = set()\n", "NTL globals"),
    ("\n\ndef _gd_cat_live(state, tokens):\n", PY_FUNCS + "\n\ndef _gd_cat_live(state, tokens):\n", "NTL python functions"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gd_ntl_arm"]',
     "GD module exports gd_ntl_arm"),
    ("        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)\n"
     "        r[\"mod\"].gd_run(r[\"inst\"], r[\"w\"], r[\"esc\"], x, out, T, r[\"K\"], ldo, r[\"tb\"], r[\"nbx\"], 1, pdl)\n",
     "        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)\n"
     "        if proj == \"pooler\" and _NTL[\"live\"] in (1, 2, 3):\n"
     "            _ntl_arm(r[\"mod\"])\n"
     "        r[\"mod\"].gd_run(r[\"inst\"], r[\"w\"], r[\"esc\"], x, out, T, r[\"K\"], ldo, r[\"tb\"], r[\"nbx\"], 1, pdl)\n",
     "_gd arms the T = 1 pooler"),
    ("    r[\"mod\"].gdr_run(vf, r[\"w\"], r[\"esc\"], x, y, T, r[\"K\"], r[\"N\"], r[\"tb\"], r[\"nbx\"], pdl,\n",
     "    if vf == 2 and _NTL[\"live\"] in (1, 2, 3):\n        _ntl_arm(r[\"mod\"])\n"
     "    r[\"mod\"].gdr_run(vf, r[\"w\"], r[\"esc\"], x, y, T, r[\"K\"], r[\"N\"], r[\"tb\"], r[\"nbx\"], pdl,\n",
     "_vfr_gdr arms the T = 2..8 fused pooler"),
    ("        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
     "        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        if _NTL[\"live\"] in (1, 2, 3):\n            _ntl_arm(r[\"mod\"])\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
     "_gd_cat arms the T = 9..16 cat"),
    ("def _v8_tail(y, logits, state, data):\n    v8 = state[\"v8\"]\n",
     "def _v8_tail(y, logits, state, data):\n"
     "    if _NTL[\"armed\"] or _NTL[\"live\"] == 4:\n"
     "        _NTL[\"armed\"] = False\n"
     "        T = y.shape[0]\n"
     "        if (logits is not None and logits.dim() == 3 and tuple(logits.shape) == (8, T, N_EXPERTS) and logits.stride(2) == 1\n"
     "                and logits.dtype == torch.float32 and y.dtype == torch.bfloat16 and y.is_contiguous() and y.shape[1] == D_EXPERT):\n"
     "            tid = torch.empty((T, TOP_K), dtype=torch.int32, device=y.device)\n"
     "            if _NTL[\"live\"] != 3:\n"
     "                _NTL[\"pend\"] = (y, logits, data[\"local_start\"], tid)\n"
     "            if _NTL[\"live\"] == 2:\n"
     "                return y, None, tid, None\n"
     "        else:\n"
     "            _NTL[\"ss\"].zero_()\n"
     "    v8 = state[\"v8\"]\n",
     "_v8_tail: NTL hand-off (no tail2 launch)"),
    ("def _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens):\n",
     "def _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens):\n"
     "    if _NTL[\"pend\"] is not None:\n"
     "        y, lg, ls, tid = _NTL[\"pend\"]\n"
     "        _NTL[\"pend\"] = None\n"
     "        return _ntl_vms(state, y, lg, ls, tid)\n",
     "_v8_experts: the VMS-NTL launch"),
    ("    mode, place = _v8_router_mode(tokens, state)\n",
     "    _NTL[\"live\"] = _ntl_on(state, tokens)\n    _NTL[\"armed\"] = False\n    _NTL[\"pend\"] = None\n"
     "    mode, place = _v8_router_mode(tokens, state)\n",
     "forward: NTL per-call state"),
    ("    if VMS_ENABLE:\n        jobs.append((\"vms\", _vms_build))\n",
     "    if VMS_ENABLE:\n        jobs.append((\"vms\", _vms_build))\n    if VMS_ENABLE and NTL_ENABLE:\n"
     "        jobs.append((\"ntl\", _ntl_build))\n",
     "PT1: parallel VMS-NTL build"),
    ("    torch.cuda.synchronize()\n    _pt1_join_all()\n",
     "    torch.cuda.synchronize()\n    _pt1_join_all()\n    _ntl_prepare(state)\n", "prepare: NTL self-tests"),
]


def _region(t, start):
    a0 = t.index(start)
    a1 = t.index(NL + "'''" + NL, a0) if start.endswith("r'''") else t.index(NL + '"""' + NL, a0)
    return a0, a1


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = (f"# nonce {TAG} ({BASE.stem} + NTL: no routing tail at T <= 16; per-CTA top-8 in VMS after its PDL wait, "
             "pooler sum(p^2) red.add + xn built in the VMS x load; NTL_MODE per bucket read at call time)")
    t = b.replace(bn, nonce)
    for start, patches in (("GD_CUDA_SRC = r'''", GD_PATCHES), ('VMS_CUDA_SRC = r"""', VMS_PATCHES)):
        a0, a1 = _region(t, start)
        c = t[a0:a1]
        for a, b2, what in patches:
            GM.check(c.count(a) == 1, f"anchor for {what}: found {c.count(a)}")
            c = c.replace(a, b2)
            rep.append(what)
        t = t[:a0] + c + t[a1:]
    cpp_anchor = "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n"
    GM.check(t.count(cpp_anchor) == 1, f"gd_unpack_cat binding: found {t.count(cpp_anchor)}")
    t = t.replace(cpp_anchor, GD_CPP_ADD + cpp_anchor)
    rep.append("CPP binding gd_ntl_arm")
    for a, b2, what in PY_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_ntl] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_ntl] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

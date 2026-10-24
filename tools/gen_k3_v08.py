"""gen_k3_v08.py: k3_v08.py = k3_v07.py + measured layer-level pieces, each behind a per-token-bucket route table
that falls back to k3_v07's own path (rules R2/R3). Budget behind each piece: notes/layer_budget.md.

    python3 -I tools/gen_k3_v08.py            # writes kernels/muse_decode_layer/k3_v08.py, asserts the invariants
    python3 -I tools/gen_k3_v08.py --check    # same checks, compares with the file on disk, writes nothing

Canonical sources (read, never edited):
  kernels/muse_decode_layer/k3_v07.py   everything not listed below (r1 glue, K1 slice, K2A v2.2 slice, _experts, ...)
  kernels/muse_bf16_proj/r1b_v2.py      R1B: the split-K GEMV (FP32 router direct at T <= 16, BF16 qkv_gate / pooler /
                                        unpooler slice at T == 1) and the router hi||lo "kcat" path; CPP/CUDA sources
                                        byte-identical, host pieces ast-identical after an R1B_/_r1b_ rename map
                                        (judge job f5e82357e66f4494 / r1b_v2 scored 1.1155 on muse_bf16_proj)

What changes vs k3_v07 (everything else is asserted ast-identical):
  * new V8 section (levers + R1B slice + glue) before the Rung-1 banner;
  * kernel_fn: qkv / pooler / unpooler via _v8_gemm (GEMV at T == 1), the router via _v8_logits with a per-bucket
    mode (stock | gemv | kcat) and placement (side = k3_v07 order, side_first, main = no side stream);
  * prepare: _v8_prepare() last (after K2A, so K2A's self-test start budget is unchanged).
"""

import ast
import importlib.util
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_LAYER = ROOT / "kernels/muse_decode_layer/k3_v07.py"
SRC_R1B = ROOT / "kernels/muse_bf16_proj/r1b_v2.py"
SRC_K1V1 = ROOT / "kernels/muse_attention_decode/k1_v1.py"
SRC_K1V4 = ROOT / "kernels/muse_attention_decode/k1_v4.py"
SRC_K2A3 = ROOT / "kernels/muse_moe_w4a16/k2a_v3.py"
USE_K2A3 = True    # stage 3': K2A v3 (judge clone 1.245x over v2.2 on muse_moe_w4a16) as the expert path
USE_K1_V4 = "--k1v1" not in sys.argv   # stage 2: K1 v4 (judge 1.414 vs k1_v2 1.347 on muse_attention_decode) in place of K1 v1
OUT = ROOT / "kernels/muse_decode_layer/k3_v08.py"
NONCE = "k3-v08-2"

_spec = importlib.util.spec_from_file_location("gen_k3", ROOT / "tools/gen_k3.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
Src, patch, check, dump, sha = G.Src, G.patch, G.check, G.dump, G.sha
_spec7 = importlib.util.spec_from_file_location("gen_k3_v07", ROOT / "tools/gen_k3_v07.py")
G7 = importlib.util.module_from_spec(_spec7)
_spec7.loader.exec_module(G7)

R1B_SHA1 = {"CPP_SRC": None, "CUDA_SRC": None}   # filled from r1b_v2.py and printed (pinning starts with v08)
R1B_RENAME = {
    "ROUTER_GEMV_MAX_M": "R1B_ROUTER_GEMV_MAX_M", "GEMV_ROUTES": "R1B_GEMV_ROUTES", "NWARP": "R1B_NWARP",
    "UNROLL": "R1B_UNROLL", "SMEM_MAX": "R1B_SMEM_MAX", "COUNTERS": "R1B_COUNTERS", "bucket": "_r1b_bucket",
    "vec_of": "_r1b_vec_of", "geometry": "_r1b_geometry", "split_router": "_r1b_split_router",
    "_build": "_r1b_build", "_router_split": "_r1b_router_split", "_gemv": "_r1b_gemv", "_gemv_route": "_r1b_gemv_route",
    "_aligned": "_r1b_aligned", "_test_input": "_r1b_test_input", "_errors": "_r1b_errors", "_self_test": "_r1b_self_test",
    "_self_test_router_split": "_r1b_self_test_router_split", "CPP_SRC": "R1B_CPP_SRC", "CUDA_SRC": "R1B_CUDA_SRC",
}
_R1B_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, R1B_RENAME), key=len, reverse=True))
                     + r")(?![A-Za-z0-9_])")
R1B_CONSTS = ("R1B_FIXUP", "ROUTER_GEMV_MAX_M", "GEMV_ROUTES", "NWARP", "UNROLL", "SMEM_MAX", "COUNTERS")
R1B_FNS = ("bucket", "vec_of", "geometry", "split_router")
R1B_SOURCES = ("CPP_SRC", "CUDA_SRC")
R1B_FNS2 = ("_build", "_router_split", "_gemv", "_gemv_route", "_aligned", "_test_input", "_errors", "_self_test",
            "_self_test_router_split")


def r1b_rename(text):
    return _R1B_RE.sub(lambda m: R1B_RENAME.get(m.group(1), m.group(1)), text)


def r1b_new_name(nm):
    return R1B_RENAME.get(nm, nm)


V8_HEAD = '''# ==============================================================================================
# V8 (tools/gen_k3_v08.py): layer-level pieces measured with tools/box/layer_budget.py + the judge clone; each one
# has a per-token-bucket route table (key = _k2a_bucket_of(T): 1, 8, 16, 32, 128, 256; None above 320) and falls back
# to k3_v07's own path for any bucket not routed to it, or when its build / self-test fails.
# ==============================================================================================
# Router (logits FP32): "stock" = k3_v07 (FP32 F.linear up to 8 tokens, two BF16 hi/lo GEMMs above), "gemv" = R1B's
# split-K GEMV on the FP32 weight (exact), "kcat" = one BF16 GEMM over [x | x] x [hi | lo]^T with FP32 accumulation,
# "f32" = V8's FP32-activation GEMV on the boundary's FP32 router input (T <= 8 only, SGLang's precision; its split-K
# partials are summed by the tail). Up to 8 tokens only "f32" or "stock" ever run (BF16 activations are refused there).
V8_ROUTER_ROUTE = @ROUTER_ROUTE@
# Where the router + routing run: "side" = k3_v07 (side stream, captured after the pooler), "side_first" = side
# stream captured before the pooler, "main" = main stream before the pooler (no side stream, no events).
V8_ROUTER_PLACE = @ROUTER_PLACE@
# qkv_gate / pooler / unpooler slice at T == 1 on R1B's GEMV (routes R1B_GEMV_ROUTES; judge: 1.21 / 1.20 / 1.36x).
V8_DENSE_GEMV_T1 = @DENSE_GEMV@
V8_R1B = True                # build + self-test R1B in prepare(); False = k3_v07 router and dense paths everywhere
# Post-pooler tail (one V8 kernel per token: routing top-8 from the router logits + pre-expert RMSNorm + local ids;
# replaces FusedMuseSparkRouting, the local-id add + index_select and F.rms_norm): None = k3_v07, "sep" = router
# per V8_ROUTER_ROUTE/PLACE and the pooler as before, then the tail; "cat" = ONE cuBLAS GEMM over
# [pooler; router_hi; router_lo] (N = 4608, FP32 out: the router's hi and lo products are each FP32-accumulated
# and summed in the tail, the pooler columns are rounded to BF16 as SGLang's BF16 pooler output) + the tail.
V8_TAIL_ROUTE = @TAIL_ROUTE@
V8_TAIL = True               # build + self-test the tail in prepare(); False = k3_v07 path everywhere

'''

V8_GLUE = '''

def _v8_prepare(weights):
    """R1B's prepare (r1b_v2.py), on the layer's weights: the router split (kcat) + its self-test, the GEMV build and
    the per-route FP64 self-test. Returns R1B's state, or None (k3_v07's router and dense paths everywhere)."""
    if not V8_R1B:
        return None
    t0 = time.time()
    dev = weights["router"].device
    st = {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    hi, lo = _r1b_split_router(st["router"])
    st["router_cat"] = torch.cat((hi, lo), dim=1).contiguous()      # [256, 16384]
    st["router_mode"] = "kcat"
    try:
        if not _r1b_self_test_router_split(st, weights):
            raise RuntimeError("tolerance")
    except Exception as ex:  # noqa: BLE001
        print(f"[v8] router kcat FAILED -> stock router: {ex!r}", flush=True)
        st["router_mode"] = None
    st["mod"] = None
    st["routes"] = {}
    st["counters"] = torch.zeros((R1B_COUNTERS,), dtype=torch.int32, device=dev)
    st["dummy"] = torch.zeros((4,), dtype=torch.float32, device=dev)
    try:
        st["mod"] = _r1b_build()
        st["routes"] = dict(R1B_GEMV_ROUTES)
    except Exception as ex:  # noqa: BLE001
        print("[v8] R1B GEMV build FAILED -> cuBLAS. Tail:\\n" + str(ex)[-8000:], flush=True)
        st["mod"] = None
    if st["mod"] is not None:
        try:
            st["routes"] = _r1b_self_test(st, weights)
        except Exception as ex:  # noqa: BLE001
            print(f"[v8] R1B self-test crashed -> cuBLAS: {ex!r}", flush=True)
            st["mod"] = None
            st["routes"] = {}
        st["counters"].zero_()
    torch.cuda.synchronize()
    print(f"[v8] R1B ready in {time.time() - t0:.1f}s: router_mode={st['router_mode']} gemv routes "
          f"{sorted(f'{a}/{b}' for a, b in st['routes'])}; router route {V8_ROUTER_ROUTE} place {V8_ROUTER_PLACE};"
          f" dense gemv T1 {V8_DENSE_GEMV_T1}", flush=True)
    return st


def _v8_gemm(x, layer, w, state):
    """A dense projection: R1B's GEMV at T == 1 when routed and self-tested, else k3_v07's _gemm."""
    st = state.get("r1b")
    if V8_DENSE_GEMV_T1 and st is not None and x.shape[0] == 1:
        n, k = w.shape
        route = _r1b_gemv_route(st, layer, 1, n, k)
        if route is not None and _r1b_aligned(x):
            return _r1b_gemv(st, x, w, n, route)
    return _gemm(x, w)


def _v8_router_mode(tokens, state):
    """(mode, place) for this token count (shape-only)."""
    st = state.get("r1b")
    rep = _k2a_bucket_of(tokens)
    mode = V8_ROUTER_ROUTE.get(rep, "stock") if st is not None else "stock"
    if mode == "f32":
        # FP32 router input x FP32 weight (stock's precision); only where the boundary writes the FP32 router input
        # (T <= 8, as SGLang) and the tail (which sums its partials) is live for this bucket.
        if tokens > ROUTER_FP32_MAX_TOKENS:
            mode = "gemv"
        elif state.get("v8") is None or V8_TAIL_ROUTE.get(rep) != "sep":
            mode = "stock"
    if tokens <= ROUTER_FP32_MAX_TOKENS and mode in ("gemv", "kcat"):
        mode = "stock"   # rules: up to 8 tokens the router reads FP32 activations, as SGLang does
    if mode == "gemv" and _r1b_gemv_route(st, "router", tokens, N_EXPERTS, HIDDEN) is None:
        mode = "kcat"
    if mode == "kcat" and not st["router_mode"]:
        mode = "stock"
    place = V8_ROUTER_PLACE.get(rep, "side") if mode != "stock" else "side"
    return mode, place


def _v8_logits(mode, ffn_input, router_input, state):
    """Router logits [T, 256] FP32."""
    st = state.get("r1b")
    if mode == "f32":
        part = torch.empty((8, router_input.shape[0], N_EXPERTS), dtype=torch.float32, device=router_input.device)
        state["v8"]["mod"].v8_router32(router_input, state["router"], part)
        return part
    if mode == "gemv" and _r1b_aligned(ffn_input):
        route = _r1b_gemv_route(st, "router", ffn_input.shape[0], N_EXPERTS, HIDDEN)
        return _r1b_gemv(st, ffn_input, st["router"], N_EXPERTS, route)
    if mode in ("gemv", "kcat") and st["router_mode"]:
        return _r1b_router_split(ffn_input, st, "kcat")
    if router_input is not None:
        return F.linear(router_input, state["router"])
    logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
    logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
    return logits


V8_TAIL_CPP_SRC = r"""
#include <torch/extension.h>
void v8_tail(torch::Tensor y, torch::Tensor lg, int64_t la_off, int64_t lb_off, int64_t l_ld, int64_t ns,
             int64_t sstride, torch::Tensor nw, torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn,
             torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps);
void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part);
"""

V8_TAIL_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <math.h>

namespace v8 {
constexpr int D = 4096;      // pooler / expert width
constexpr int NE = 256;      // experts
constexpr int TK = 8;        // top-k
constexpr int NT = 512;      // threads: 8 elements of the row each; threads 0..255 also own one expert

// One CTA per token (warp 0 also does the routing). y: [T, y_ld] (FP32 -> rounded to BF16 first, or BF16), columns 0..4095 = pooler output.
// Logits [T, l_ld] FP32 at la (+ lb if given: the router's hi and lo products). Outputs: xn = BF16 RMSNorm(y) * nw
// (F.rms_norm on the BF16 pooler output with the BF16 effective weight), tw/tid = FusedMuseSparkRouting's top-8
// (sigmoid(mult * logit) + bias selects, argmax ties -> lowest id, weights = picked sigmoid / (sum + deps) * rscale),
// lid = this GPU's local ids ((id - local_start) mod 256 if < 32, else -1).
template <bool Y32>
__global__ void __launch_bounds__(NT) tail_kernel(
    const void* __restrict__ y, long y_ld, const float* __restrict__ la, const float* __restrict__ lb, long l_ld,
    int ns, long sstride, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ bias, const int* __restrict__ lstart,
    __nv_bfloat16* __restrict__ xn, float* __restrict__ tw, int* __restrict__ tid, int* __restrict__ lid,
    float eps, float mult, float rscale, float deps) {
  const int t = blockIdx.x, tx = threadIdx.x, lane = tx & 31, warp = tx >> 5;
  __shared__ float red[NT / 32];
  // warp 0 owns the routing: lane l holds experts 8l .. 8l+7 (loads issued first, used after the norm)
  float s[8], b[8];
  if (warp == 0) {
    const float4* pa = reinterpret_cast<const float4*>(la + (long)t * l_ld) + lane * 2;
    float4 a0 = pa[0], a1 = pa[1];
    for (int q = 1; q < ns; ++q) {   // FP32 split-K partials of the router (v8_router32), summed in a fixed order
      const float4* pq = reinterpret_cast<const float4*>(la + q * sstride + (long)t * l_ld) + lane * 2;
      const float4 q0 = pq[0], q1 = pq[1];
      a0.x += q0.x; a0.y += q0.y; a0.z += q0.z; a0.w += q0.w; a1.x += q1.x; a1.y += q1.y; a1.z += q1.z; a1.w += q1.w;
    }
    if (lb != nullptr) {
      const float4* pb = reinterpret_cast<const float4*>(lb + (long)t * l_ld) + lane * 2;
      const float4 b0 = pb[0], b1 = pb[1];
      a0.x += b0.x; a0.y += b0.y; a0.z += b0.z; a0.w += b0.w; a1.x += b1.x; a1.y += b1.y; a1.z += b1.z; a1.w += b1.w;
    }
    const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float4* pbias = reinterpret_cast<const float4*>(bias) + lane * 2;
    const float4 c0 = pbias[0], c1 = pbias[1];
    const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j] = 1.f / (1.f + expf(-lg[j] * mult));
      b[j] = s[j] + bs[j];
    }
  }
  float v[8];
  if (Y32) {
    const float4* p = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(y) + (long)t * y_ld) + tx * 2;
    const float4 a = p[0], c = p[1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(__float2bfloat16(v[i]));
  } else {
    const uint4 q = *(reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(y) + (long)t * y_ld) + tx);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
  }
  const uint4 wq = *(reinterpret_cast<const uint4*>(nw) + tx);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < NT / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / (float)D + eps);
  {
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq);
    __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((v[i] * rstd) * __bfloat162float(w[i]));
    *(reinterpret_cast<uint4*>(xn + (long)t * D) + tx) = *reinterpret_cast<const uint4*>(o8);
  }
  if (warp != 0) return;
  // top-8 by biased score, ties -> lowest expert id (tl.argmax), shuffles only
  int sel_id[TK];
  float sel_s[TK];
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    float bv = b[0], sv = s[0];
    int bj = 0;
#pragma unroll
    for (int j = 1; j < 8; ++j)
      if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
    int bi = lane * 8 + bj;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      const float os = __shfl_xor_sync(0xffffffffu, sv, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; sv = os; }
    }
    sel_id[k] = bi;
    sel_s[k] = sv;
    if ((bi >> 3) == lane) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j == (bi & 7)) b[j] = -INFINITY;
    }
  }
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK; ++i) sum += sel_s[i];
    const float inv = rscale / (sum + deps);
    const int ls = lstart[0];
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      tw[t * TK + k] = sel_s[k] * inv;
      tid[t * TK + k] = sel_id[k];
      const int rel = ((sel_id[k] - ls) % NE + NE) % NE;
      lid[t * TK + k] = rel < 32 ? rel : -1;
    }
  }
}

// T <= 8 router, exactly SGLang's precision regime: FP32 activations (the boundary's FP32 router input) x FP32
// weight, FP32 accumulation. x [m <= 8, 8192], w [256, 8192] -> part [8 (K slices), m, 256]; the tail sums the
// 8 slices in order (deterministic, no atomics, no counters). grid (32, 8) x 256 threads: warp = one router row,
// CTA = 8 rows x one 1024-wide K slice; each lane keeps 8 float4 weight loads in flight (evict-first).
template <int MT>
__global__ void __launch_bounds__(256) router32_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                       float* __restrict__ part, int m) {
  constexpr int K = 8192, KS = 1024, KS4 = KS / 4;
  __shared__ float4 xs[MT * KS4];
  const int sl = blockIdx.y, row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  const int k0 = sl * KS;
  float4 wv[8];
  const float4* wp = reinterpret_cast<const float4*>(w + (long)row * K + k0);
#pragma unroll
  for (int i = 0; i < 8; ++i) wv[i] = __ldcs(wp + i * 32 + lane);
  for (int idx = threadIdx.x; idx < MT * KS4; idx += 256) {
    const int j = idx / KS4, c = idx - j * KS4;
    xs[idx] = (j < m) ? reinterpret_cast<const float4*>(x + (long)j * K + k0)[c] : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  __syncthreads();
  float acc[MT];
#pragma unroll
  for (int j = 0; j < MT; ++j) acc[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int c = i * 32 + lane;
#pragma unroll
    for (int j = 0; j < MT; ++j) {
      const float4 xv = xs[j * KS4 + c];
      acc[j] = fmaf(wv[i].x, xv.x, acc[j]);
      acc[j] = fmaf(wv[i].y, xv.y, acc[j]);
      acc[j] = fmaf(wv[i].z, xv.z, acc[j]);
      acc[j] = fmaf(wv[i].w, xv.w, acc[j]);
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
}  // namespace v8

void v8_router32(torch::Tensor x, torch::Tensor w, torch::Tensor part) {
  const int m = (int)x.size(0);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat,
              "v8_router32: dtypes");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && part.is_contiguous(), "v8_router32: layout");
  TORCH_CHECK(x.size(1) == 8192 && w.size(0) == 256 && w.size(1) == 8192 && m >= 1 && m <= 8
              && part.numel() == 8L * m * 256, "v8_router32: shapes");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "v8_router32: alignment");
  auto st = at::cuda::getCurrentCUDAStream();
  if (m == 1)
    v8::router32_kernel<1><<<dim3(32, 8), 256, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  else
    v8::router32_kernel<8><<<dim3(32, 8), 256, 0, st>>>(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void v8_tail(torch::Tensor y, torch::Tensor lg, int64_t la_off, int64_t lb_off, int64_t l_ld, int64_t ns,
             int64_t sstride, torch::Tensor nw, torch::Tensor bias, torch::Tensor lstart, torch::Tensor xn,
             torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, double eps, double mult, double rscale, double deps) {
  const int T = (int)y.size(0);
  TORCH_CHECK(y.stride(1) == 1 && lg.stride(lg.dim() - 1) == 1 && y.size(1) >= v8::D && ns >= 1 && sstride % 4 == 0,
              "v8_tail: layout");
  TORCH_CHECK(lg.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat, "v8_tail: dtypes");
  TORCH_CHECK(nw.scalar_type() == at::kBFloat16 && nw.numel() == v8::D, "v8_tail: norm weight");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(xn.data_ptr()) % 16 == 0, "v8_tail: alignment");
  if (T == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  const float* base = lg.data_ptr<float>();
  const float* la = base + la_off;
  const float* lb = lb_off >= 0 ? base + lb_off : nullptr;
  TORCH_CHECK(l_ld % 4 == 0 && la_off % 4 == 0 && (lb_off < 0 || lb_off % 4 == 0)
              && reinterpret_cast<uintptr_t>(base) % 16 == 0 && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0,
              "v8_tail: logits / bias alignment");
  const long y_ld = y.stride(0);
  auto* xp = reinterpret_cast<__nv_bfloat16*>(xn.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr());
  if (y.scalar_type() == at::kFloat)
    v8::tail_kernel<true><<<T, v8::NT, 0, st>>>(y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns, (long)sstride, wp, bias.data_ptr<float>(),
        lstart.data_ptr<int>(), xp, tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
        (float)eps, (float)mult, (float)rscale, (float)deps);
  else
    v8::tail_kernel<false><<<T, v8::NT, 0, st>>>(y.data_ptr(), y_ld, la, lb, (long)l_ld, (int)ns, (long)sstride, wp, bias.data_ptr<float>(),
        lstart.data_ptr<int>(), xp, tw.data_ptr<float>(), tid.data_ptr<int>(), lid.data_ptr<int>(),
        (float)eps, (float)mult, (float)rscale, (float)deps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _v8_tail_build():
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((V8_TAIL_CPP_SRC + V8_TAIL_CUDA_SRC).encode()).hexdigest()[:10]
    return load_inline(name=f"k3_v8_tail_{tag}", cpp_sources=V8_TAIL_CPP_SRC, cuda_sources=V8_TAIL_CUDA_SRC,
                       functions=["v8_tail", "v8_router32"], extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def _v8_tail(y, logits, state, data):
    """(expert_x BF16 [T, 4096], topk_weights FP32 [T, 8], topk_ids int32 [T, 8], local_ids int32 [T, 8]).
    logits None: y = [T, 4608] FP32 (pooler | router hi | router lo); else y = pooler output and logits [T, 256] FP32,
    or [8, T, 256] FP32 split-K partials (T <= 8 router on the FP32 router input), summed in order by the tail."""
    v8 = state["v8"]
    tokens = y.shape[0]
    dev = y.device
    xn = torch.empty((tokens, D_EXPERT), dtype=torch.bfloat16, device=dev)
    tw = torch.empty((tokens, TOP_K), dtype=torch.float32, device=dev)
    tid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    lid = torch.empty((tokens, TOP_K), dtype=torch.int32, device=dev)
    ns, sstride = 1, 0
    if logits is None:
        lg, la, lb, l_ld = y, D_EXPERT, D_EXPERT + N_EXPERTS, y.stride(0)
    elif logits.dim() == 3:   # [8, T, 256] FP32 partials of the T <= 8 router (v8_router32)
        lg, la, lb, l_ld = logits, 0, -1, logits.stride(1)
        ns, sstride = logits.shape[0], logits.stride(0)
    else:
        lg, la, lb, l_ld = logits, 0, -1, logits.stride(0)
    v8["mod"].v8_tail(y, lg, la, lb, l_ld, ns, sstride, v8["nw"], v8["bias"], data["local_start"], xn, tw, tid, lid,
                      RMS_EPS, 0.1767766952966369, 1.0, 1e-15)
    return xn, tw, tid, lid


def _v8_tail_prepare(weights, state):
    """Build the tail, the [pooler; hi; lo] weight, and self-test both against the k3_v07 ops they replace
    (F.rms_norm module, FusedMuseSparkRouting, the local-id table, FP64 router logits). Returns the tail state or None."""
    if not V8_TAIL:
        return None
    t0 = time.time()
    dev = weights["pooler"].device
    try:
        mod = _v8_tail_build()
    except Exception as ex:  # noqa: BLE001
        print("[v8] tail build FAILED -> k3_v07 path. Tail:\\\\n" + str(ex)[-8000:], flush=True)
        return None
    router = weights["router"].contiguous()
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    v8 = {"mod": mod, "nw": state["pre_expert_norm"]._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
          "bias": weights["router_bias"].float().contiguous(),
          "wcat": torch.cat((weights["pooler"], hi, lo), dim=0).contiguous()}   # [4608, 8192] BF16
    try:
        gen = torch.Generator(device=dev).manual_seed(4242)
        for T, ls in ((1, 0), (8, 64), (16, 224), (33, 32), (128, 96), (256, 160)):
            x = (torch.randn((T, HIDDEN), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            d = {"local_start": lstart}
            ref_logits = (x.double() @ router.double().T).float()
            ref_pool = _gemm(x, weights["pooler"])
            ref_xn = state["pre_expert_norm"](ref_pool)
            ref_tw, ref_tid = state["routing"](None, ref_logits.clone(), TOP_K, True)
            ref_lid = state["local_id_table"][(ref_tid.long() + N_EXPERTS * ls).reshape(-1)].view(T, TOP_K)
            for kind in ("cat", "sep"):
                if kind == "cat":
                    y = torch.mm(x, v8["wcat"].t(), out_dtype=torch.float32)
                    lg = (y[:, D_EXPERT:D_EXPERT + N_EXPERTS] + y[:, D_EXPERT + N_EXPERTS:]).contiguous()
                    lerr = float((lg - ref_logits).norm() / ref_logits.norm())
                    got = _v8_tail(y, None, {"v8": v8}, d)
                else:
                    lerr = 0.0
                    got = _v8_tail(ref_pool, ref_logits, {"v8": v8}, d)
                torch.cuda.synchronize()
                xn, tw, tid, lid = got
                xerr = float((xn.float() - ref_xn.float()).norm() / ref_xn.float().norm())
                ids_eq = bool(torch.equal(torch.sort(tid, 1).values, torch.sort(ref_tid.to(torch.int32), 1).values))
                o_g, o_r = torch.argsort(tid, 1), torch.argsort(ref_tid.to(torch.int32), 1)
                werr = float((tw.gather(1, o_g) - ref_tw.float().gather(1, o_r)).abs().max())
                lid_eq = bool(torch.equal(torch.sort(lid, 1).values, torch.sort(ref_lid.to(torch.int32), 1).values))
                print(f"[v8] tail self-test {kind} T={T} ls={ls}: logits rel {lerr:.2e} xn rel {xerr:.2e} ids_eq {ids_eq}"
                      f" w maxerr {werr:.1e} lid_eq {lid_eq}", flush=True)
                if not (lerr < 1e-4 and xerr < 1e-2 and ids_eq and werr < 1e-5 and lid_eq
                        and bool(torch.isfinite(xn.float()).all())):
                    raise RuntimeError(f"tail self-test mismatch ({kind}, T={T})")
        # T <= 8 router on FP32 activations (v8_router32) + the tail summing its partials
        for T, ls in ((1, 0), (5, 128), (8, 32)):
            h = torch.randn((T, HIDDEN), generator=gen, device=dev)
            h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
            lstart = torch.tensor([ls], dtype=torch.int32, device=dev)
            ref_logits = (h.double() @ router.double().T).float()
            part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
            mod.v8_router32(h, state["router"], part)
            pool = _gemm(h.to(torch.bfloat16), weights["pooler"])
            got = _v8_tail(pool, part, {"v8": v8}, {"local_start": lstart})
            ref_tw, ref_tid = state["routing"](None, ref_logits.clone(), TOP_K, True)
            torch.cuda.synchronize()
            lerr = float((part.sum(0) - ref_logits).norm() / ref_logits.norm())
            ids_eq = bool(torch.equal(torch.sort(got[2], 1).values, torch.sort(ref_tid.to(torch.int32), 1).values))
            o_g, o_r = torch.argsort(got[2], 1), torch.argsort(ref_tid.to(torch.int32), 1)
            werr = float((got[1].gather(1, o_g) - ref_tw.float().gather(1, o_r)).abs().max())
            print(f"[v8] router32 self-test T={T}: logits rel {lerr:.2e} ids_eq {ids_eq} w maxerr {werr:.1e}", flush=True)
            if not (lerr < 1e-5 and ids_eq and werr < 1e-5):
                raise RuntimeError(f"router32 self-test mismatch (T={T})")
    except Exception as ex:  # noqa: BLE001
        print(f"[v8] tail self-test FAILED -> k3_v07 path: {ex!r}", flush=True)
        return None
    print(f"[v8] tail ready in {time.time() - t0:.1f}s; route {V8_TAIL_ROUTE}", flush=True)
    return v8
'''

KFN_OLD_QKV = '''    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
'''
KFN_NEW_QKV = '''    qkv = _v8_gemm(data["attn_input"], "qkv_gate", dense["qkv_gate"], state)
'''
KFN_OLD_ROUTER = '''    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    if R1_LOCAL_ID_TABLE:
        # table[g + 256 * s] = expert_map[(g - s) % 256]: one add, one int32 index_select.
        idx = torch.add(topk_ids, data["local_start"], alpha=N_EXPERTS)
        local_ids = torch.index_select(state["local_id_table"], 0, idx.reshape(-1)).view(tokens, TOP_K)
    else:
        local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_x = state["pre_expert_norm"](expert_input)
'''
KFN_NEW_ROUTER = '''    mode, place = _v8_router_mode(tokens, state)
    tail = V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens)) if state.get("v8") is not None else None
    if tail == "cat" and tokens <= ROUTER_FP32_MAX_TOKENS:
        tail = "sep"   # rules: the "cat" GEMM reads BF16 activations for the router; not up to 8 tokens
    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T or mode not in ("stock", "f32") or tail == "cat"
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    if tail == "cat":
        # V8: pooler and router in ONE GEMM (FP32 out), then the tail (routing + pre-expert norm + local ids).
        y32 = torch.mm(ffn_input, state["v8"]["wcat"].t(), out_dtype=torch.float32)
        expert_x, topk_weights, topk_ids, local_ids = _v8_tail(y32, None, state, data)
    else:
        if place == "main":
            # V8: router (+ routing) on the main stream ahead of the pooler (no side stream, no events).
            logits = _v8_logits(mode, ffn_input, router_input, state)
            if tail is None:
                topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
            expert_input = _v8_gemm(ffn_input, "pooler", dense["pooler"], state)
        else:
            # The router (and routing) run on a side stream, overlapping the pooler on the main one.
            main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
            side_stream.wait_stream(main_stream)
            if place != "side_first":
                expert_input = _v8_gemm(ffn_input, "pooler", dense["pooler"], state)
            with torch.cuda.stream(side_stream):
                logits = _v8_logits(mode, ffn_input, router_input, state)
                if tail is None:
                    topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
            if place == "side_first":
                expert_input = _v8_gemm(ffn_input, "pooler", dense["pooler"], state)
            main_stream.wait_stream(side_stream)
            if not torch.cuda.is_current_stream_capturing():
                for tensor in ((logits,) if tail is not None else (topk_weights, topk_ids)):
                    tensor.record_stream(main_stream)
        if tail is not None:
            # V8: routing + pre-expert norm + local ids in one kernel after the join.
            expert_x, topk_weights, topk_ids, local_ids = _v8_tail(expert_input, logits, state, data)
        else:
            # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
            if R1_LOCAL_ID_TABLE:
                # table[g + 256 * s] = expert_map[(g - s) % 256]: one add, one int32 index_select.
                idx = torch.add(topk_ids, data["local_start"], alpha=N_EXPERTS)
                local_ids = torch.index_select(state["local_id_table"], 0, idx.reshape(-1)).view(tokens, TOP_K)
            else:
                local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
            # all-reduce: identity
            expert_x = state["pre_expert_norm"](expert_input)
'''
KFN_OLD_UNPOOL = '''    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        if R1_UNPOOL_INTO_ZEROS:
'''
KFN_NEW_UNPOOL = '''    if tokens == 1 and V8_DENSE_GEMV_T1 and state.get("r1b") is not None and _r1b_aligned(expert_output) \\
            and _r1b_gemv_route(state["r1b"], "unpooler", 1, UNPOOL_SLICE, D_EXPERT) is not None:
        # V8: R1B GEMV on this GPU's 1024-row slice, then the all-gather's zero pad (identity).
        unpooled = _v8_gemm(expert_output, "unpooler", dense["unpooler"][:UNPOOL_SLICE], state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    elif tokens <= SLICED_UNPOOL_MAX_TOKENS:
        if R1_UNPOOL_INTO_ZEROS:
'''
PREP_OLD = '''    if USE_K2A:
        state["k2a"] = _k2a_prepare(weights, t0)
'''
PREP_NEW = '''    if USE_K2A:
        state["k2a"] = _k2a_prepare(weights, t0)
    state["r1b"] = _v8_prepare(weights)
    state["v8"] = _v8_tail_prepare(weights, state)
'''

# Route tables (measured on the box; see kernels/muse_decode_layer/k3_v08.md). Initial values = r1b_v2's proj routes.
ROUTER_ROUTE = {1: "f32", 8: "f32", 16: "gemv", 32: "stock", 128: "kcat", 256: "kcat"}
# Box layer A/B 2026-10-24 (tools/box/layer_ab2.py, box_out/ab_v08a.log, ab_v08b.log; ratios vs k3_v07 path):
#   router: gemv side_first T=1 1.076/1.071, gemv side T=8 1.171/1.158, T=16 side_first 1.009, T>=32 within +-1%;
#   tail: sep T=1 1.12/1.10, T=8 1.21/1.20; cat T=16 1.054/1.037, T=32 1.067/1.036, T=128 1.063/1.043,
#   T=256 1.048/1.027 (cat at T<=8 ties sep). Layer load-geomean: router alone 1.043, + tail sep 1.066, + cat 1.085.
ROUTER_PLACE = {1: "side_first", 8: "side", 16: "side_first", 32: "side", 128: "side", 256: "main"}
TAIL_ROUTE = {1: "sep", 8: "sep", 16: "cat", 32: "cat", 128: "cat", 256: "cat"}
DENSE_GEMV = True


def r1b_section(r1b):
    for nm in R1B_SOURCES:
        v = r1b.value(nm)
        R1B_SHA1[nm] = sha(v)
        check('"""' not in v and r1b.seg(nm).startswith(f'{nm} = r"""'), f"{nm} literal form")
    parts = [V8_HEAD.replace("@ROUTER_ROUTE@", repr(ROUTER_ROUTE)).replace("@ROUTER_PLACE@", repr(ROUTER_PLACE))
             .replace("@DENSE_GEMV@", repr(DENSE_GEMV)).replace("@TAIL_ROUTE@", repr(TAIL_ROUTE)),
             "# R1B slice: kernels/muse_bf16_proj/r1b_v2.py (nonce r1b-v2-1), sources byte-identical, host code renamed.\n"]
    for nm in R1B_CONSTS:
        parts.append(r1b_rename(r1b.seg(nm)))
    parts.append("\n\n")
    for nm in R1B_FNS:
        parts.append(r1b_rename(r1b.seg(nm)) + "\n\n")
    for nm in R1B_SOURCES:
        parts.append("R1B_" + r1b.seg(nm) + "\n")
    parts.append("\n")
    for nm in R1B_FNS2:
        parts.append(r1b_rename(r1b.seg(nm)) + "\n\n")
    parts.append(V8_GLUE)
    return "".join(parts)




def k1v4_stage(text, lay):
    """K1 v1 -> K1 v4 inside the layer: the CUDA/C++ sources byte-identical to k1_v4.py; _k1_build and _k1_occupancy
    ast-identical to k1_v4.py after gen_k3's rename map; _k1_run / _k1_self_test / _k1_ready / _k1_prepare get exactly
    k1_v4's deltas (the stages argument, per-(L, stages) occupancy, FP16-Q reference); K1_STAGES_T table added.
    Asserts first that the K1 v1 slice in k3_v07 is what gen_k3 derives from k1_v1.py (sources + verbatim fns)."""
    v1, v4 = Src(SRC_K1V1), Src(SRC_K1V4)
    for nm in ("CPP_SRC", "CUDA_SRC"):
        check(lay.seg("K1_" + nm) == "K1_" + v1.seg(nm), f"k3_v07 K1_{nm} is not k1_v1's")
        check(v4.seg(nm).startswith(f'{nm} = r"""') and '"""' not in v4.value(nm), f"k1_v4 {nm} literal form")
        text = patch(text, lay.seg("K1_" + nm), "K1_" + v4.seg(nm), f"K1_{nm} -> v4")
    for nm in ("_build", "_occupancy"):
        check(dump(ast.parse(G.rename(v1.seg(nm))).body[0]) == dump(lay.nodes[G.RENAME[nm]]), f"k3_v07 {nm} != k1_v1's")
        text = patch(text, lay.seg(G.RENAME[nm]), G.rename(v4.seg(nm)), f"{nm} -> v4")
    check(v4.value("K1_STAGES_T") == {("sliding", 1): 4, ("sliding", 8): 4, ("full", 1): 4, ("full", 8): 4}, "K1_STAGES_T")
    check(v4.value("K1_ROUTE_MIN_BATCH") == lay.value("K1_ROUTE_MIN_BATCH") and v4.value("K1_CPS_T") == lay.value("K1_CPS_T"),
          "k1_v4 route / cps tables differ from k3_v07's")
    text = patch(text, lay.seg("K1_CPS_T"), lay.seg("K1_CPS_T") + "#   ring depth per class (k1_v4.py).\n" + v4.seg("K1_STAGES_T"),
                 "K1_STAGES_T")
    # _k1_run: k1_v4's _k1 deltas
    text = patch(text, "def _k1_run(qkv, data, cfg, fold=None, min_chunk=None, cps=None, nsg=None):\n",
                 "def _k1_run(qkv, data, cfg, fold=None, min_chunk=None, cps=None, nsg=None, stages=None):\n", "_k1_run sig")
    text = patch(text, "    uses). Overrides (self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg.\"\"\"\n",
                 "    uses). Overrides (self-test): fold (bool), min_chunk, cps (target CTAs/SM), nsg, stages.\"\"\"\n",
                 "_k1_run doc")
    text = patch(text, "        nsg = _k1_nsg(B)\n    cnt = _k1_counters(dev)",
                 "        nsg = _k1_nsg(B)\n    if stages is None:\n        stages = K1_STAGES_T.get(key, K1_STAGES)\n"
                 "    cnt = _k1_counters(dev)", "_k1_run stages")
    text = patch(text, "    cps_eff = max(1, min(int(cps), _k1_occupancy(L, dev)))\n",
                 "    cps_eff = max(1, min(int(cps), _k1_occupancy(L, int(stages), dev)))\n", "_k1_run occupancy")
    text = patch(text, "                    1 if fold else 0, int(nsg))\n    return out\n",
                 "                    1 if fold else 0, int(nsg), int(stages))\n    return out\n", "_k1_run stages arg")
    text = patch(text, "                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=(L == 8))\n",
                 "                ref = _k1_ref_attention(d, k_new, v_new, q_e4m3=False)   # v4: fp16 Q at both L\n",
                 "_k1_self_test reference")
    text = patch(text, '    hit = ((cfg["q_len"], str(dev)) in _K1_OCC and',
                 '    hit = ((cfg["q_len"], K1_STAGES_T.get((_k1_kind(cfg), cfg["q_len"]), K1_STAGES), str(dev)) in _K1_OCC and',
                 "_k1_ready key")
    text = patch(text, "        for L in (1, 8):\n            _k1_occupancy(L, dev)\n",
                 "        for (kind, L), st in K1_STAGES_T.items():\n            _k1_occupancy(L, int(st), dev)\n",
                 "_k1_prepare occupancy")
    text = patch(text, "K1=v1 route=", "K1=v4 route=", "switches print")
    text = patch(text, "k1 v1 occupancy L1/L8", "k1 v4 occupancy", "prepare summary")
    return text


K2A3_NAMES = ("M3", "SW3", "V3_VARIANTS", "V3_PLAN", "stage_repack", "KERNEL_SRC_V3", "CPP_SRC_V3", "build_ext_v3",
              "v3_prepare", "V3_HINT", "v3_call", "v3_reset", "V3_SELF_TEST_CASES")
K2A3_RENAME = {
    # v3's own names
    "M3": "K2A3_M3", "SW3": "K2A3_SW3", "V3_VARIANTS": "K2A3_VARIANTS", "V3_PLAN": "K2A3_PLAN",
    "stage_repack": "_k2a3_stage_repack", "KERNEL_SRC_V3": "K2A3_KERNEL_SRC", "CPP_SRC_V3": "K2A3_CPP_SRC",
    "build_ext_v3": "_k2a3_build_ext", "v3_prepare": "_k2a3_prepare_st", "V3_HINT": "K2A3_HINT", "v3_call": "_k2a3_call",
    "v3_reset": "_k2a3_reset", "V3_SELF_TEST_CASES": "K2A3_SELF_TEST_CASES",
    # pieces shared with v2.2 (asserted ast-identical between k2a_v3.py and k2a_v22.py): the layer's K2A_/_k2a_ names
    "E_LOC": "K2A_E_LOC", "D_HIDDEN": "K2A_D_HIDDEN", "D_EXPERT": "K2A_D_EXPERT", "W_SHIFT": "K2A_W_SHIFT",
    "META_HDR": "K2A_META_HDR", "prepack_expert": "_k2a_prepack_expert", "TOP_K": "K2A_TOP_K",
}
_K2A3_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(sorted(map(re.escape, K2A3_RENAME), key=len, reverse=True))
                      + r")(?![A-Za-z0-9_])")
K2A3_SHARED = ("E_LOC", "D_HIDDEN", "D_EXPERT", "TOP_K", "POST_NORM_EPS", "M_MAX", "META_HDR", "BITPOS", "W_SHIFT",
               "reencode_scales", "pack_words", "tile_row_perm", "prepack_expert", "bucket_of", "BUCKETS",
               "_dequant", "ref_fp32", "rel_err", "synth_inputs")

K2A3_GLUE = '''

def _v8_k2a3_prepare(weights, t0):
    """K2A v3 (k2a_v3.py: stage repack straight from the NVFP4 weights, build, occupancy) on the layer's experts,
    then its self-test per bucket through the layer entry point (remote ids -1, as the layer passes them), two calls
    per case (the second checks that h / the counters were restored). Returns {"st", "ok": {bucket rep: bool}} or None."""
    if not any(v == "v3" for v in V8_K2A3_ROUTE.values()):
        return None
    dev = weights["w13"].device
    w = {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
         "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
         "post_norm_w": weights["post_expert_norm"]}
    t1 = time.time()
    try:
        st = _k2a3_prepare_st(w, log=lambda s: print(s, flush=True))
        _K3_BUILD_S["k2a3"] = time.time() - t1
    except Exception as ex:  # noqa: BLE001
        print("[v8] K2A v3 repack/build FAILED: experts stay on K2A v2.2 / Marlin. Tail:\\n" + str(ex)[-8000:], flush=True)
        return None
    ok = {rep: True for _, _, rep in K2A_BUCKETS}
    for bs, seed, dist in K2A3_SELF_TEST_CASES:
        rep = _k2a_bucket_of(bs)
        try:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            for k in range(2):
                st["err"].zero_()
                got = _v8_k2a3_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st)
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(got, ref)
                errf = int(st["err"][0])
                good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                print(f"[k2a3] layer self-test bs={bs} {dist} v{K2A3_PLAN[rep]} call {k}: rel {tot:.3e} worst row "
                      f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                if not good:
                    ok[rep] = False
        except Exception as ex:  # noqa: BLE001
            print(f"[v8] K2A v3 self-test bs={bs} raised {ex!r}", flush=True)
            ok[rep] = False
        _k2a3_reset(st)
    print(f"[v8] K2A v3 buckets {ok} in {time.time() - t1:.1f}s; route {V8_K2A3_ROUTE}", flush=True)
    return {"st": st, "ok": ok}


def _v8_k2a3_call(x, local_ids, topk_weights, st):
    """K2A v3 on the layer: x [T, 4096] BF16 (pre-expert-normed), local ids (0..31, remote -1; lo = 0)."""
    x = x.contiguous()
    if x.data_ptr() % 16:   # bulk copies need 16 B-aligned rows (a pointer property, not a value)
        x = x.clone()
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    return _k2a3_call(st, data, K2A3_PLAN[_k2a_bucket_of(x.shape[0])])


def _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens):
    """THE expert entry point (sum_k w_k * RMSNorm(expert_k(x) * (1 + post_expert_norm)), BF16 [T, 4096]):
    K2A v3 where V8_K2A3_ROUTE says "v3" and its self-test passed for the bucket, else K2A v2.2 per K2A_ROUTE,
    else None (the caller runs r1's Marlin path). A later expert kernel drops in here."""
    rep = _k2a_bucket_of(tokens)
    k3s = state.get("k2a3")
    if (k3s is not None and rep is not None and V8_K2A3_ROUTE.get(rep) == "v3" and k3s["ok"].get(rep)
            and tokens <= K2A3_M3 and cfg["local_experts"] == K2A_E_LOC and cfg["top_k"] == K2A_TOP_K
            and cfg["n_experts"] == N_EXPERTS):
        try:
            return _v8_k2a3_call(expert_x, local_ids, topk_weights, k3s["st"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "k2a3", None, "K2A v3", ex)
    if state["k2a"] is not None and _k2a_supported(cfg, tokens):
        try:
            return _k2a_experts(expert_x, local_ids, topk_weights, state["k2a"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "k2a", None, "K2A", ex)
    return None


def _v8_unpool_t1(expert_output, state):
    """T == 1: R1B GEMV on this GPU's 1024-row slice written straight into the zeroed [1, 8192] output (the
    all-gather's zero pad), or None when the route is not live."""
    st = state.get("r1b")
    w = state["dense"]["unpooler"][:UNPOOL_SLICE]
    if not (V8_DENSE_GEMV_T1 and st is not None and _r1b_aligned(expert_output)):
        return None
    route = _r1b_gemv_route(st, "unpooler", 1, UNPOOL_SLICE, D_EXPERT)
    if route is None or route[3] != 1:
        return None
    out = torch.zeros((1, HIDDEN), dtype=expert_output.dtype, device=expert_output.device)
    mt, rpw, wk, split = route
    st["mod"].r1b_gemv(expert_output, w, out[:, :UNPOOL_SLICE], st["dummy"], st["counters"], 1, UNPOOL_SLICE,
                       D_EXPERT, mt, rpw, wk, split, 1 if R1B_FIXUP else 0)
    return out
'''


def k2a3_stage(text, lay):
    k22, k3 = Src(ROOT / "kernels/muse_moe_w4a16/k2a_v22.py"), Src(SRC_K2A3)
    for nm in K2A3_SHARED:
        check(dump(k22.nodes[nm]) == dump(k3.nodes[nm]), f"k2a_v3 {nm} differs from k2a_v22's (layer reuses v2.2's)")
    for nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
        check(k3.seg(nm).startswith(f'{nm} = r"""') and '"""' not in k3.value(nm), f"{nm} literal form")
    parts = ["# K2A v3 slice: kernels/muse_moe_w4a16/k2a_v3.py (nonce k2a-v3-1), sources byte-identical, host code renamed\n"
             "# (K2A3_/_k2a3_; names shared with v2.2 map to the layer's K2A_/_k2a_ ones).\n"
             "V8_K2A3_ROUTE = " + repr(K2A3_ROUTE) + "   # bucket rep -> \"v3\" | anything else = K2A v2.2 / Marlin per K2A_ROUTE\n"
             "V8_BUILD_V22_ALWAYS = False   # lab A/B only: build K2A v2.2 even when v3 covers every bucket\n"]
    for nm in K2A3_NAMES:
        if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
            parts.append(K2A3_RENAME[nm] + k3.seg(nm)[len(nm):] + "\n")
        else:
            parts.append(_K2A3_RE.sub(lambda m: K2A3_RENAME[m.group(1)], k3.seg(nm)) + "\n")
    parts.append(K2A3_GLUE)
    section = "".join(parts)
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, section + "\n\n" + banner, "K2A v3 section")
    text = patch(text, """    expert_output = None
    if state["k2a"] is not None and _k2a_supported(cfg, tokens):
        try:
            # K2a: route + W13/SiLU*up + W2 + post-expert norm x topk weight sum; ids are local (lo = 0).
            expert_output = _k2a_experts(expert_x, local_ids, topk_weights, state["k2a"])
        except Exception as ex:  # noqa: BLE001
            _k3_disable(state, "k2a", None, "K2A", ex)
            expert_output = None
    if expert_output is None:
""", """    # V8: one expert entry point (K2A v3 -> K2A v2.2 -> None = r1 Marlin).
    expert_output = _v8_experts(expert_x, local_ids, topk_weights, state, cfg, tokens)
    if expert_output is None:
""", "kernel_fn experts")
    text = patch(text, """    if tokens == 1 and V8_DENSE_GEMV_T1 and state.get("r1b") is not None and _r1b_aligned(expert_output) \\
            and _r1b_gemv_route(state["r1b"], "unpooler", 1, UNPOOL_SLICE, D_EXPERT) is not None:
        # V8: R1B GEMV on this GPU's 1024-row slice, then the all-gather's zero pad (identity).
        unpooled = _v8_gemm(expert_output, "unpooler", dense["unpooler"][:UNPOOL_SLICE], state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
    elif tokens <= SLICED_UNPOOL_MAX_TOKENS:
""", """    ffn_output = _v8_unpool_t1(expert_output, state) if tokens == 1 else None
    if ffn_output is not None:
        pass   # V8: R1B GEMV into the zeroed output (all-gather: identity)
    elif tokens <= SLICED_UNPOOL_MAX_TOKENS:
""", "kernel_fn unpooler")
    # prepare: v3 first; K2A v2.2 is built only when some bucket still routes to it
    text = patch(text, """    if USE_K2A:
        state["k2a"] = _k2a_prepare(weights, t0)
""", """    state["k2a3"] = None
    if USE_K2A:
        state["k2a3"] = _v8_k2a3_prepare(weights, t0)
        k3ok = state["k2a3"]["ok"] if state["k2a3"] is not None else {}
        need_v2 = [rep for rep, r in K2A_ROUTE.items() if r == "v2" and not (V8_K2A3_ROUTE.get(rep) == "v3" and k3ok.get(rep))]
        state["k2a"] = _k2a_prepare(weights, t0) if (need_v2 or V8_BUILD_V22_ALWAYS) else None
        print(f"[v8] experts: v3 buckets {sorted(r for r, o in k3ok.items() if o and V8_K2A3_ROUTE.get(r) == 'v3')},"
              f" v2.2 built for {need_v2}", flush=True)
""", "prepare experts")
    text = patch(text, """experts={'K2A' if state['k2a'] is not None else 'r1-Marlin'}\"""",
                 """experts={'K2A v3' if state.get('k2a3') is not None else ('K2A' if state['k2a'] is not None else 'r1-Marlin')}\"""",
                 "prepare summary experts")
    return text


K2A3_ROUTE = {1: "v3", 8: "v3", 16: "v3", 32: "v3", 128: "v3", 256: "v3"}


def generate():
    lay, r1b = Src(SRC_LAYER), Src(SRC_R1B)
    text = lay.text
    lines = text.split("\n")
    check(lines[1].startswith("# nonce k3-v07-1 "), "k3_v07 nonce")
    text = patch(text, lines[1] + "\n" + "\n".join(lines[2:5]) + "\n",
                 f"# nonce {NONCE} (k3_v07 + V8 layer pieces: R1B router/GEMV routes; see k3_v08.md)\n"
                 "# GENERATED by tools/gen_k3_v08.py from k3_v07.py (itself generated by gen_k3_v07.py) and r1b_v2.py.\n"
                 "# Do not hand-edit: change a canonical source or a generator and regenerate.\n", "header")
    banner = ("# ==============================================================================================\n"
              "# Rung 1 (kernels/muse_decode_layer/r1_glue.py, nonce r1-1): unchanged, and the fallback of both pieces.\n")
    text = patch(text, banner, r1b_section(r1b) + "\n\n" + banner, "V8 section")
    text = patch(text, KFN_OLD_QKV, KFN_NEW_QKV, "kernel_fn qkv")
    text = patch(text, KFN_OLD_ROUTER, KFN_NEW_ROUTER, "kernel_fn router")
    text = patch(text, KFN_OLD_UNPOOL, KFN_NEW_UNPOOL, "kernel_fn unpooler")
    text = patch(text, PREP_OLD, PREP_NEW, "prepare")
    if USE_K1_V4:
        text = k1v4_stage(text, lay)
    if USE_K2A3:
        text = k2a3_stage(text, lay)
    return text, lay, r1b


def verify(text, lay, r1b):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text)
    try:
        py_compile.compile(str(tmp), doraise=True)
        out = Src(tmp)
    finally:
        tmp.unlink()
    rep = []
    check(text.splitlines()[1].startswith(f"# nonce {NONCE} ") and text.count(NONCE) == 1, "nonce")
    for nm in R1B_SOURCES:
        check(out.value("R1B_" + nm) == r1b.value(nm), f"R1B_{nm} differs")
        check(out.seg("R1B_" + nm) == "R1B_" + r1b.seg(nm), f"R1B_{nm} raw text differs")
        rep.append(f"R1B_{nm}: byte-identical to r1b_v2.py (sha1 {R1B_SHA1[nm][:12]})")
    for nm in R1B_CONSTS + R1B_FNS + R1B_FNS2:
        ref = ast.parse(r1b_rename(r1b.seg(nm))).body[0]
        check(dump(ref) == dump(out.nodes[r1b_new_name(nm)]), f"{nm} differs after renames")
    rep.append(f"r1b_v2 host pieces ast-identical after renames: {len(R1B_CONSTS + R1B_FNS + R1B_FNS2)}")
    if USE_K2A3:
        k3 = Src(SRC_K2A3)
        for nm in K2A3_NAMES:
            if nm in ("KERNEL_SRC_V3", "CPP_SRC_V3"):
                check(out.value(K2A3_RENAME[nm]) == k3.value(nm), f"{nm} not byte-identical")
            else:
                ref = ast.parse(_K2A3_RE.sub(lambda m: K2A3_RENAME[m.group(1)], k3.seg(nm))).body[0]
                check(dump(ref) == dump(out.nodes[K2A3_RENAME[nm]]), f"k2a_v3 {nm} differs after renames")
        rep.append(f"k2a_v3 pieces: sources byte-identical (sha1 {sha(k3.value('KERNEL_SRC_V3'))[:12]}), "
                   f"{len(K2A3_NAMES) - 2} host pieces ast-identical after renames; shared pieces == k2a_v22's")
    allowed = {"kernel_fn", "prepare"} | ({"K1_CPP_SRC", "K1_CUDA_SRC", "_k1_build", "_k1_occupancy", "_k1_run",
                                             "_k1_self_test", "_k1_ready", "_k1_prepare"} if USE_K1_V4 else set())
    changed = [n for n, node in lay.nodes.items() if n not in out.nodes or dump(out.nodes[n]) != dump(node)]
    check(set(changed) <= allowed, f"unexpected changes vs k3_v07: {sorted(set(changed) - allowed)}")
    rep.append(f"vs k3_v07: {len(lay.nodes) - len(changed)} top-level nodes ast-identical; changed {sorted(changed)}")
    # static scan (same rules as gen_k3_v07)
    for node in ast.walk(out.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                check(m.split(".")[0] not in G.BANNED_NAMES, f"banned import {m}")
        if isinstance(node, ast.Name):
            check(node.id not in G.BANNED_NAMES, f"banned name {node.id} (line {node.lineno})")
        if isinstance(node, ast.Attribute):
            check(node.attr not in ("environ", "system", "popen"), f"banned attribute {node.attr}")
    hot = ("kernel_fn", "_v8_gemm", "_v8_router_mode", "_v8_logits", "_r1b_gemv", "_r1b_gemv_route", "_r1b_router_split",
           "_v8_tail", "_v8_experts", "_v8_k2a3_call", "_v8_unpool_t1", "_k2a3_call")
    for fn in hot:
        for sub in ast.walk(out.nodes[fn]):
            check(not (isinstance(sub, ast.Attribute) and sub.attr in G.SYNC_ATTRS), f"{fn} has a host sync")
    bad = G7.undefined_globals(text, str(OUT))
    check(not bad, f"undefined globals: {sorted(bad)}")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"static: banned names none; no host syncs in {', '.join(hot)}; no undefined globals; "
               f"{len(text.encode())} bytes")
    return rep


def main(argv):
    text, lay, r1b = generate()
    rep = verify(text, lay, r1b)
    for r in rep:
        print(f"[gen_k3_v08] OK  {r}")
    if "--k1v1" in argv:   # A/B helper only: the same file with K1 v1 kept, written elsewhere, never shipped
        alt = ROOT / "kernels/muse_decode_layer/v08lab/k3_v08_k1v1.py"
        alt.parent.mkdir(exist_ok=True)
        alt.write_text(text.replace(f"# nonce {NONCE} ", f"# nonce {NONCE}-k1v1-lab "))
        print(f"[gen_k3_v08] wrote {alt.relative_to(ROOT)} (lab A/B only)")
        return 0
    if "--check" in argv:
        check(OUT.exists() and OUT.read_text() == text, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_v08] --check: up to date (sha1 {sha(text)[:12]})")
        return 0
    OUT.write_text(text)
    print(f"[gen_k3_v08] wrote {OUT.relative_to(ROOT)} (sha1 {sha(text)[:12]}, {len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

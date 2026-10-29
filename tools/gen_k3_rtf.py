"""gen_k3_rtf.py: kernels/lab/k3_rtf_<n>.py = a GC layer file (default k3_gc_16) + RTF, the routing tail (v8::tail_kernel2)
folded into its producer's epilogue by a last-arriver CTA.

Every CTA of the producer grids holds a slice of EVERY token (GEMV row tiles / router K slices), so a token's last arriver is
the grid's last arriver: after its stores each CTA fences and takes one ticket (atomicAdd on its own counter slot); the CTA
whose ticket completes the grid (no spinning, no barrier: every other CTA exits at once) runs the tail of all M tokens and
resets the slot (replay-safe in the CUDA graph, no host work):
* T = 2..8: host = the VF = 2 fused pooler + router-rows GD launch (both halves: top-8 + routing weights + xn).
* T = 9..32: host = the GC cat launch (both halves).
* T = 1: the pooler (GD, main stream) and the FP32 router (lean router32, side stream) are separate kernels; each takes a
  half: the pooler's last arriver writes xn (pre-expert norm), a copy of the lean router kernel (same arithmetic) whose last
  arriver writes the top-8 / weights / local ids. The existing side-stream join orders both before the MoE kernel.
* T >= 128: the cat is cuBLAS (no epilogue): unchanged.
Arithmetic = tail_kernel2's (slice sums in the same order incl. the + 0 of the absent lb, sigmoid + bias, key / ballot top-8
ties, 16 x 32 x 8 norm layout played by 8 warps x 2): prepare self-tests demand bitwise tid / lid / tw / xn vs _v8_tail on the
same producer outputs (and bitwise router partials vs vf_router32 at T = 1); a failing bucket stays on the tail (FALLBACK print).
RTF_MODE[bucket] is read at call time: 0 off, 1 inert (rider writes scratch, tail2 kept: step-0 cost probe), 2 live (tail2
dropped for that bucket; the MoE consumes the rider's outputs).

Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_rtf.py [--base=k3_gc_16] [--n=1]
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
OUT = ROOT / f"kernels/lab/k3_rtf_{N}.py"
TAG = f"k3-rtf-{N}"
NL = chr(10)

DEVICE = r'''
// RTF: the routing tail of all M tokens, run by the grid's last-arriving CTA (arithmetic = v8::tail_kernel2)
__device__ __forceinline__ void rtf_route(const float (&lg)[8], const float4 c0, const float4 c1, float mult, float rscale,
                                          float deps, int ls, float* tw, int* tid, int* lid, int t, int lane, int* sid,
                                          float* ssc) {
  constexpr int TK = 8, NE = 256;
  const float bs[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
  float s[8], b[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    s[j] = 1.f / (1.f + expf(-lg[j] * mult));
    b[j] = s[j] + bs[j];
  }
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    float bv = b[0], sv = s[0];
    int bj = 0;
#pragma unroll
    for (int j = 1; j < 8; ++j)
      if (b[j] > bv) { bv = b[j]; sv = s[j]; bj = j; }
    const unsigned u = __float_as_uint(bv + 0.f);
    const unsigned key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    const unsigned mx = __reduce_max_sync(0xffffffffu, key);
    const int owner = __ffs(__ballot_sync(0xffffffffu, key == mx)) - 1;
    const int id = __shfl_sync(0xffffffffu, lane * 8 + bj, owner);
    const float sc = __shfl_sync(0xffffffffu, sv, owner);
    if (lane == 0) { sid[k] = id; ssc[k] = sc; }
    if (lane == owner) {
#pragma unroll
      for (int j = 0; j < 8; ++j)
        if (j == bj) b[j] = -INFINITY;
    }
  }
  __syncwarp();
  if (lane == 0) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < TK; ++i) sum += ssc[i];
    const float inv = rscale / (sum + deps);
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      tw[t * TK + k] = ssc[k] * inv;
      tid[t * TK + k] = sid[k];
      const int rel = ((sid[k] - ls) % NE + NE) % NE;
      lid[t * TK + k] = rel < 32 ? rel : -1;
    }
  }
}

__device__ __forceinline__ void rtf_add4(float4& a, const float4 x) { a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w; }

template <int WH, int STG>   // WH: compiled halves (1 top-8 / weights / local ids, 2 xn); STG: stage through gd_vf_dsm (VF = 2)
__device__ __forceinline__ void rtf_tail(const Args& A) {
  constexpr int D = 4096, NB = WH == 3 ? 2 : 1;   // register path: tokens per norm round (T = 1 pooler host <= 104 registers)
  __shared__ int s_last;
  __shared__ float s_red[8][16];
  __shared__ int s_id[8][8];
  __shared__ float s_sc[8][8];
  const int tx = (int)threadIdx.x, lane = tx & 31, warp = tx >> 5;
  if (!(A.rtf_what & 8)) __threadfence();
  __syncthreads();
  if (tx == 0) {
    if (A.rtf_what & 8) __threadfence();
    const unsigned nb = gridDim.x * gridDim.y * gridDim.z;
    const unsigned old = atomicAdd(reinterpret_cast<unsigned*>(A.rtf_ctr), 1u);
    const int last = old == nb - 1u;
    if (last) { __threadfence(); atomicExch(A.rtf_ctr, 0); }
    s_last = last;
  }
  __syncthreads();
  if (!s_last || (A.rtf_what & 4)) return;
  const int M = A.M;
  const __nv_bfloat16* y = reinterpret_cast<const __nv_bfloat16*>(A.out);
  if ((WH & 1) && (A.rtf_what & 1)) {   // the tail's logits warp, one token per warp
    const float4* pb = reinterpret_cast<const float4*>(A.rtf_bias) + lane * 2;
    if constexpr (STG) {   // all M x 8 x 256 partials -> smem (cp.async, no registers), then each warp reads its token
      float* sm = reinterpret_cast<float*>(gd_vf_dsm);
      for (int i = tx; i < M * 512; i += 256) {
        const int t = i >> 9, q = (i >> 6) & 7, c = i & 63;
        vf_cp16(sm + (t * 8 + q) * 256 + c * 4, A.rtf_part + (long long)q * A.rtf_sst + (long long)t * A.rtf_ld + c * 4);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      const float4 c0 = pb[0], c1 = pb[1];
      const int ls = A.rtf_ls[0];
      asm volatile("cp.async.wait_all;\n" ::: "memory");
      __syncthreads();
      for (int t = warp; t < M; t += 8) {
        const float* st = sm + t * 2048 + lane * 8;
        float lg[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) lg[j] = st[j];
#pragma unroll
        for (int q = 1; q < 8; ++q)
#pragma unroll
          for (int j = 0; j < 8; ++j) lg[j] += st[q * 256 + j];
#pragma unroll
        for (int j = 0; j < 8; ++j) lg[j] += 0.f;   // lb = 0
        rtf_route(lg, c0, c1, A.rtf_mult, A.rtf_rscale, A.rtf_deps, ls, A.rtf_tw, A.rtf_tid, A.rtf_lid, t, lane, s_id[warp],
                  s_sc[warp]);
      }
      __syncthreads();   // the y stage below reuses sm
    } else {
      const float4 c0 = pb[0], c1 = pb[1];
      const int ls = A.rtf_ls[0];
      for (int t = warp; t < M; t += 8) {
        const float* base = A.rtf_part + (long long)t * A.rtf_ld;
        float4 a0, a1;
#pragma unroll
        for (int hh = 0; hh < 4; ++hh) {   // slice pairs, summed in the tail's order (slice 0 first)
          float4 q0[2], q1[2];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const float4* pq = reinterpret_cast<const float4*>(base + (long long)(hh * 2 + q) * A.rtf_sst) + lane * 2;
            q0[q] = __ldcg(pq);
            q1[q] = __ldcg(pq + 1);
          }
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            if (hh == 0 && q == 0) { a0 = q0[0]; a1 = q1[0]; }
            else { rtf_add4(a0, q0[q]); rtf_add4(a1, q1[q]); }
          }
        }
        a0.x += 0.f; a0.y += 0.f; a0.z += 0.f; a0.w += 0.f; a1.x += 0.f; a1.y += 0.f; a1.z += 0.f; a1.w += 0.f;   // lb = 0
        const float lg[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        rtf_route(lg, c0, c1, A.rtf_mult, A.rtf_rscale, A.rtf_deps, ls, A.rtf_tw, A.rtf_tid, A.rtf_lid, t, lane, s_id[warp],
                  s_sc[warp]);
      }
    }
  }
  if ((WH & 2) && (A.rtf_what & 2)) {   // the tail's 16 norm warps (512 x 8 elements): warp w plays tail warps w and w + 8
    if constexpr (STG) {   // all M rows of y -> smem, then per-token sums, one barrier, rstd, xn
      __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(gd_vf_dsm);
      for (int i = tx; i < M * 512; i += 256) {
        const int t = i >> 9, c = i & 511;
        vf_cp16(sb + t * D + c * 8, y + (size_t)t * A.ldo + c * 8);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
      uint4 wq[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) wq[h] = *(reinterpret_cast<const uint4*>(A.rtf_nw) + (warp + 8 * h) * 32 + lane);
      asm volatile("cp.async.wait_all;\n" ::: "memory");
      __syncthreads();
      for (int t = 0; t < M; ++t)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 q = *(reinterpret_cast<const uint4*>(sb + t * D) + (warp + 8 * h) * 32 + lane);
          const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q);
          float v[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(hb[i]);
          float ss = 0.f;
#pragma unroll
          for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
          if (lane == 0) s_red[t][warp + 8 * h] = ss;
        }
      __syncthreads();
      for (int t = 0; t < M; ++t) {
        float tot = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) tot += s_red[t][i];
        const float rstd = rsqrtf(tot / (float)D + A.rtf_eps);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint4 q = *(reinterpret_cast<const uint4*>(sb + t * D) + (warp + 8 * h) * 32 + lane);
          const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q);
          const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq[h]);
          __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((__bfloat162float(hb[i]) * rstd) * __bfloat162float(w[i]));
          *(reinterpret_cast<uint4*>(A.rtf_xn + (size_t)t * D) + (warp + 8 * h) * 32 + lane) = *reinterpret_cast<const uint4*>(o8);
        }
      }
    } else {
      uint4 wq[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) wq[h] = *(reinterpret_cast<const uint4*>(A.rtf_nw) + (warp + 8 * h) * 32 + lane);
      for (int t0 = 0; t0 < M; t0 += NB) {
        uint4 q[NB][2];
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            q[j][h] = t0 + j < M ? __ldcg(reinterpret_cast<const uint4*>(y + (size_t)(t0 + j) * A.ldo) + (warp + 8 * h) * 32 + lane)
                                 : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (int j = 0; j < NB; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q[j][h]);
            float v[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(hb[i]);
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) s_red[j][warp + 8 * h] = ss;
          }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NB; ++j) {
          const int t = t0 + j;
          if (t < M) {
            float tot = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) tot += s_red[j][i];
            const float rstd = rsqrtf(tot / (float)D + A.rtf_eps);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&q[j][h]);
              const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&wq[h]);
              __align__(16) __nv_bfloat16 o8[8];
#pragma unroll
              for (int i = 0; i < 8; ++i) o8[i] = __float2bfloat16((__bfloat162float(hb[i]) * rstd) * __bfloat162float(w[i]));
              *(reinterpret_cast<uint4*>(A.rtf_xn + (size_t)t * D) + (warp + 8 * h) * 32 + lane) = *reinterpret_cast<const uint4*>(o8);
            }
          }
        }
        __syncthreads();
      }
    }
  }
}

struct RtfR {
  int* ctr; const float* bias; const int* ls; float* tw; int* tid; int* lid; float mult, rscale, deps; int flags;
};

// RTF at T = 1: vf_router32_kernel<1> (same arithmetic, same grid (16, 8) x 256, same bounds) + its last arriver's top-8
__global__ void __launch_bounds__(256, 6) rtf_router_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                                            float* __restrict__ part, int m, const RtfR r) {
  constexpr int K = 8192, KS = 1024, MT = 1;
  __shared__ int s_last;
  __shared__ __align__(16) float s_lg[8][256];
  __shared__ int s_id[8];
  __shared__ float s_sc[8];
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
  if (!(r.flags & 8)) __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    if (r.flags & 8) __threadfence();
    const unsigned nb = gridDim.x * gridDim.y;
    const unsigned old = atomicAdd(reinterpret_cast<unsigned*>(r.ctr), 1u);
    const int last = old == nb - 1u;
    if (last) { __threadfence(); atomicExch(r.ctr, 0); }
    s_last = last;
  }
  __syncthreads();
  if (!s_last || (r.flags & 4)) return;
  {   // warp q stages K slice q's 256 partials
    const float4* pq = reinterpret_cast<const float4*>(part + warp * 256) + lane * 2;
    const float4 x0 = __ldcg(pq), x1 = __ldcg(pq + 1);
    reinterpret_cast<float4*>(&s_lg[warp][0])[lane * 2] = x0;
    reinterpret_cast<float4*>(&s_lg[warp][0])[lane * 2 + 1] = x1;
  }
  __syncthreads();
  if (warp == 0) {
    float lg[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) lg[j] = s_lg[0][lane * 8 + j];
#pragma unroll
    for (int q = 1; q < 8; ++q)
#pragma unroll
      for (int j = 0; j < 8; ++j) lg[j] += s_lg[q][lane * 8 + j];
#pragma unroll
    for (int j = 0; j < 8; ++j) lg[j] += 0.f;   // lb = 0
    const float4* pb = reinterpret_cast<const float4*>(r.bias) + lane * 2;
    rtf_route(lg, pb[0], pb[1], r.mult, r.rscale, r.deps, r.ls[0], r.tw, r.tid, r.lid, 0, lane, s_id, s_sc);
  }
}

'''

HOST_ARM = r'''
struct RtfArm {
  int on; int what; int* ctr; __nv_bfloat16* xn; float* tw; int* tid; int* lid; const float* bias; const __nv_bfloat16* nw;
  const int* ls; float eps, mult, rscale, deps;
};
static RtfArm g_rtf = {};   // RTF: armed by gd_rtf_arm() right before ONE host launch, consumed (cleared) by it

static void rtf_apply(Args& A, const float* part, long long sst, int ld) {
  if (!g_rtf.on) return;
  g_rtf.on = 0;
  const int halves = g_rtf.what & (part != nullptr ? 3 : 2);
  const int what = halves | (g_rtf.what & 12);
  if (!halves && !(what & 4)) return;
  A.rtf_ctr = g_rtf.ctr; A.rtf_what = what; A.rtf_part = part; A.rtf_sst = sst; A.rtf_ld = ld;
  A.rtf_bias = g_rtf.bias; A.rtf_nw = g_rtf.nw; A.rtf_ls = g_rtf.ls; A.rtf_xn = g_rtf.xn; A.rtf_tw = g_rtf.tw;
  A.rtf_tid = g_rtf.tid; A.rtf_lid = g_rtf.lid; A.rtf_eps = g_rtf.eps; A.rtf_mult = g_rtf.mult; A.rtf_rscale = g_rtf.rscale;
  A.rtf_deps = g_rtf.deps;
}

void gd_rtf_arm_raw(void* ctr, void* xn, void* tw, void* tid, void* lid, const void* bias, const void* nw, const void* ls, int what,
                    float eps, float mult, float rscale, float deps) {
  g_rtf.on = what != 0; g_rtf.what = what; g_rtf.ctr = reinterpret_cast<int*>(ctr);
  g_rtf.xn = reinterpret_cast<__nv_bfloat16*>(xn); g_rtf.tw = reinterpret_cast<float*>(tw); g_rtf.tid = reinterpret_cast<int*>(tid);
  g_rtf.lid = reinterpret_cast<int*>(lid); g_rtf.bias = reinterpret_cast<const float*>(bias);
  g_rtf.nw = reinterpret_cast<const __nv_bfloat16*>(nw); g_rtf.ls = reinterpret_cast<const int*>(ls);
  g_rtf.eps = eps; g_rtf.mult = mult; g_rtf.rscale = rscale; g_rtf.deps = deps;
}

int gd_rtf_router_raw(const float* x, const float* w, float* part, void* ctr, void* tw, void* tid, void* lid, const void* bias,
                      const void* ls, float mult, float rscale, float deps, int flags, cudaStream_t st) {
  RtfR r;
  r.ctr = reinterpret_cast<int*>(ctr); r.bias = reinterpret_cast<const float*>(bias); r.ls = reinterpret_cast<const int*>(ls);
  r.tw = reinterpret_cast<float*>(tw); r.tid = reinterpret_cast<int*>(tid); r.lid = reinterpret_cast<int*>(lid);
  r.mult = mult; r.rscale = rscale; r.deps = deps; r.flags = flags;
  rtf_router_kernel<<<dim3(16, 8), 256, 0, st>>>(x, w, part, 1, r);
  return (int)cudaGetLastError();
}
'''

CPP_ADD = r'''void gd_rtf_arm_raw(void* ctr, void* xn, void* tw, void* tid, void* lid, const void* bias, const void* nw, const void* ls, int what,
                    float eps, float mult, float rscale, float deps);
int gd_rtf_router_raw(const float* x, const float* w, float* part, void* ctr, void* tw, void* tid, void* lid, const void* bias,
                      const void* ls, float mult, float rscale, float deps, int flags, cudaStream_t st);
static void rtf_chk_outs(torch::Tensor& ctr, torch::Tensor& tw, torch::Tensor& tid, torch::Tensor& lid, torch::Tensor& bias,
                         torch::Tensor& ls) {
  TORCH_CHECK(ctr.scalar_type() == at::kInt && ctr.numel() >= 1 && tw.scalar_type() == at::kFloat && tid.scalar_type() == at::kInt &&
              lid.scalar_type() == at::kInt && tw.is_contiguous() && tid.is_contiguous() && lid.is_contiguous() &&
              tw.numel() == tid.numel() && tw.numel() == lid.numel() && bias.scalar_type() == at::kFloat && bias.numel() == 256 &&
              bias.is_contiguous() && reinterpret_cast<uintptr_t>(bias.data_ptr()) % 16 == 0 && ls.scalar_type() == at::kInt &&
              ls.numel() >= 1, "rtf: ctr / tw / tid / lid / bias / local_start");
}
void gd_rtf_arm(torch::Tensor ctr, torch::Tensor xn, torch::Tensor tw, torch::Tensor tid, torch::Tensor lid, torch::Tensor bias,
                torch::Tensor nw, torch::Tensor ls, int64_t what, double eps, double mult, double rscale, double deps) {
  if (what != 0) {
    rtf_chk_outs(ctr, tw, tid, lid, bias, ls);
    TORCH_CHECK(xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.dim() == 2 && xn.size(1) == 4096 &&
                xn.size(0) * 8 == tw.numel() && reinterpret_cast<uintptr_t>(xn.data_ptr()) % 16 == 0 &&
                nw.scalar_type() == at::kBFloat16 && nw.numel() == 4096 && reinterpret_cast<uintptr_t>(nw.data_ptr()) % 16 == 0,
                "gd_rtf_arm: xn [M, 4096] bf16 / nw");
  }
  gd_rtf_arm_raw(ctr.data_ptr(), xn.data_ptr(), tw.data_ptr(), tid.data_ptr(), lid.data_ptr(), bias.data_ptr(), nw.data_ptr(),
                 ls.data_ptr(), (int)what, (float)eps, (float)mult, (float)rscale, (float)deps);
}
void gd_rtf_router(torch::Tensor x, torch::Tensor w, torch::Tensor part, torch::Tensor ctr, torch::Tensor tw, torch::Tensor tid,
                   torch::Tensor lid, torch::Tensor bias, torch::Tensor ls, double mult, double rscale, double deps, int64_t flags) {
  rtf_chk_outs(ctr, tw, tid, lid, bias, ls);
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat && part.scalar_type() == at::kFloat &&
              x.is_contiguous() && w.is_contiguous() && part.is_contiguous() && x.dim() == 2 && x.size(0) == 1 &&
              x.size(1) == 8192 && w.size(0) == 256 && w.size(1) == 8192 && part.numel() == 8 * 256 && tw.numel() == 8 &&
              reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(part.data_ptr()) % 16 == 0, "gd_rtf_router: x f32 [1, 8192], w [256, 8192], part [8, 1, 256]");
  const int e = gd_rtf_router_raw(x.data_ptr<float>(), w.data_ptr<float>(), part.data_ptr<float>(), ctr.data_ptr(), tw.data_ptr(),
                                  tid.data_ptr(), lid.data_ptr(), bias.data_ptr(), ls.data_ptr(), (float)mult, (float)rscale,
                                  (float)deps, (int)flags, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gd_rtf_router: CUDA error ", e);
}
'''

CUDA_PATCHES = [
    ("uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n};",
     "uint32_t tbr = 0;   // GC: router partials [8][M][ldo2] f32, bases\n"
     "  int* rtf_ctr = nullptr; int rtf_what = 0; const float* rtf_part = nullptr; long long rtf_sst = 0; int rtf_ld = 0;   // RTF\n"
     "  const float* rtf_bias = nullptr; const __nv_bfloat16* rtf_nw = nullptr; const int* rtf_ls = nullptr;\n"
     "  __nv_bfloat16* rtf_xn = nullptr; float* rtf_tw = nullptr; int* rtf_tid = nullptr; int* rtf_lid = nullptr;\n"
     "  float rtf_eps = 0.f, rtf_mult = 0.f, rtf_rscale = 0.f, rtf_deps = 0.f;\n};",
     "Args: RTF fields"),
    ("#define GC_XP(S_)", DEVICE + "#define GC_XP(S_)", "RTF device code (rtf_tail, rtf_router_kernel)"),
    ("  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n  constexpr int KUP = KU - CAT;",
     "  constexpr int NW = WR * WK, NT = NW * 32, NP = 3;\n"
     "  constexpr bool RTF_HOST = S == 1 && NT == 256 && OUT == 0 && (CAT > 0 || VF == 2 || (MT == 1 && WR == 1 && WK == 8 && KU == 16));\n"
     "  constexpr int KUP = KU - CAT;",
     "kernel: RTF_HOST"),
    ("    if constexpr (VF >= 1) {\n      if (A.marks) {\n        __syncthreads();\n"
     "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n  } else {",
     "    if constexpr (RTF_HOST) {\n      if (A.rtf_ctr != nullptr) rtf_tail<(CAT > 0 || VF == 2) ? 3 : 2, VF == 2 ? 1 : 0>(A);\n    }\n"
     "    if constexpr (VF >= 1) {\n      if (A.marks) {\n        __syncthreads();\n"
     "        if (threadIdx.x == 0) A.marks[vf_c * 4 + 3] = vf_gt();\n      }\n    }\n  } else {",
     "kernel: last-arriver call in the S = 1 epilogue"),
    ("}  // namespace gd\n\nusing namespace gd;\n", "}  // namespace gd\n\nusing namespace gd;\n" + HOST_ARM, "host: g_rtf / rtf_apply"),
    ("  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n  switch (inst) {",
     "  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n  rtf_apply(A, nullptr, 0, 0);\n  switch (inst) {",
     "gd_run_raw applies the arm (norm half only)"),
    ("  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n  const int nz = 1;",
     "  A.out2 = out2; A.ldo2 = ldo2; A.sst2 = (long long)M * ldo2; A.tbr = tbr;\n"
     "  rtf_apply(A, reinterpret_cast<const float*>(out2), (long long)M * ldo2, ldo2);\n  const int nz = 1;",
     "gdc_run_raw applies the arm"),
    ("  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;\n",
     "  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;\n"
     "  if (vf == 2) rtf_apply(A, part, (long long)M * 256, 256); else rtf_apply(A, nullptr, 0, 0);\n",
     "gdr_run_raw applies the arm"),
]

PY_GLOBALS = r'''RTF_MODE = {1: 0, 8: 0, 16: 0, 32: 0}   # per GD bucket: 0 off, 1 inert rider (scratch; tail2 kept), 2 live (tail2 dropped)
RTF_SELF_TEST_T = {1: (1,), 8: (8, 5), 16: (16, 9), 32: (32, 17)}
RTF_FLAGS = 0   # diagnostics (inert arms only): 4 ticket only (no tail work), 8 fence by thread 0 after the barrier
_RTF = {"ok": {}, "p": {}, "req": None, "ls": None, "scr": None, "ctr": None, "seen": set(), "live_ok": False}
'''

PY_FUNCS = r'''

def _rtf_on(T):
    b = _gd_bucket(T)
    m = RTF_MODE.get(b, 0) if b is not None else 0
    return m if (m in (1, 2) and _RTF["ok"].get(b) and _RTF["ls"] is not None) else 0


def _rtf_outs(state, T, m):
    if m == 1:
        return tuple(s[:T] for s in _RTF["scr"])
    dev = state["router"].device
    return (torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev), torch.empty((T, TOP_K), dtype=torch.float32, device=dev),
            torch.empty((T, TOP_K), dtype=torch.int32, device=dev), torch.empty((T, TOP_K), dtype=torch.int32, device=dev))


def _rtf_arm(mod, state, slot, outs, what, ls):
    v8 = state["v8"]
    mod.gd_rtf_arm(_RTF["ctr"][slot:slot + 1], outs[0], outs[1], outs[2], outs[3], v8["bias"], v8["nw"], ls,
                   what | (RTF_FLAGS if _RTF["live_ok"] else 0),
                   RMS_EPS, 0.1767766952966369, 1.0, 1e-15)
    if (slot, what) not in _RTF["seen"]:
        _RTF["seen"].add((slot, what))
        print(f"[rtf] first arm slot {slot} what {what} T={outs[1].shape[0]}", flush=True)


def _rtf_req1(state, T):
    m = _rtf_on(T) if T == 1 else 0
    if not m or _gd_live(state, "pooler", T) is None:
        _RTF["req"] = None
        return
    outs = _rtf_outs(state, T, m)
    _RTF["req"] = {"pool": (outs, m), "rout": (outs, m)}


def _rtf_router(state, x, part):
    rq = _RTF["req"].pop("rout", None) if _RTF["req"] else None
    if rq is None or not _VF["r8"] or _RTF["ls"] is None:
        return False
    outs, m = rq
    v8 = state["v8"]
    _GD["mod"].gd_rtf_router(x, state["router"], part, _RTF["ctr"][1:2], outs[1], outs[2], outs[3], v8["bias"], _RTF["ls"],
                             0.1767766952966369, 1.0, 1e-15, RTF_FLAGS if _RTF["live_ok"] else 0)
    if m == 2:
        _RTF["p"]["rt"] = (part.data_ptr(), outs[1], outs[2], outs[3])
    return True


def _rtf_cmp(got, ref):
    xn, tw, tid, lid = got
    if not (torch.equal(tid, ref[2]) and torch.equal(lid, ref[3]) and torch.equal(tw, ref[1])):
        return f"routing differs (tid/lid mismatches {int((tid != ref[2]).sum()) + int((lid != ref[3]).sum())}, tw equal {bool(torch.equal(tw, ref[1]))})"
    if not torch.equal(xn, ref[0]):
        ulp = int((xn.view(torch.int16).to(torch.int32) - ref[0].view(torch.int16).to(torch.int32)).abs().max())
        return f"xn differs by {ulp} bf16 ulp"
    return None


def _rtf_test(state, b, T, ls, seed):
    dev = state["router"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn((T, HIDDEN), generator=gen, device=dev)
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
    hb = h.to(torch.bfloat16).contiguous()
    lst = torch.tensor([ls], dtype=torch.int32, device=dev)
    d = {"local_start": lst}
    outs = _rtf_outs(state, T, 2)
    part = torch.empty((8, T, N_EXPERTS), dtype=torch.float32, device=dev)
    _RTF["p"].clear()
    _RTF["req"] = None
    if b == 1:
        r = _gd_live(state, "pooler", T)
        if r is None or not _VF["r8"] or _SBB1["mod"] is None:
            return "GD pooler T=1 / lean router not live"
        _rtf_arm(r["mod"], state, 0, outs, 2, lst)
        y = _gd(state, "pooler", hb)
        _RTF["ls"], _RTF["req"] = lst, {"rout": (outs, 1)}
        ran = _rtf_router(state, h, part)
        _RTF["req"] = None
        if y is None or not ran:
            return "pooler / router did not run"
        ref_part = torch.empty_like(part)
        _SBB1["mod"].vf_router32(h, state["router"], ref_part, 0)
        torch.cuda.synchronize()
        if not torch.equal(part, ref_part):
            return "router partials differ from vf_router32"
    elif b == 8:
        r = _gd_live(state, "pooler", T)
        if r is None or not _VFR["ok"]:
            return "VF fused pooler not live"
        _rtf_arm(r["mod"], state, 2, outs, 3, lst)
        y, part = _vfr_gdr(state, hb, h, 2)
    else:
        r = _gd_cat_live(state, T)
        if r is None:
            return "GC cat not live"
        y = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
        _rtf_arm(r["mod"], state, 3 if b == 16 else 4, outs, 3, lst)
        _gd_cat_run(r, hb, y, part, T, GD_CAT_PDL)
    _RTF["p"].clear()
    ref = _v8_tail(y, part, state, d)
    torch.cuda.synchronize()
    c = _RTF["ctr"].tolist()
    if any(c):
        return f"counters {c} not reset"
    return _rtf_cmp(outs, ref)


def _rtf_prepare(state):
    _RTF["ok"] = {}
    mod = _GD.get("mod")
    if state.get("v8") is None or mod is None or not hasattr(mod, "gd_rtf_arm"):
        print("[rtf] FALLBACK: GD module / v8 tail missing -> RTF off", flush=True)
        return
    dev = state["router"].device
    _RTF["ctr"] = torch.zeros((8,), dtype=torch.int32, device=dev)
    _RTF["scr"] = (torch.empty((32, D_EXPERT), dtype=torch.bfloat16, device=dev), torch.empty((32, TOP_K), dtype=torch.float32, device=dev),
                   torch.empty((32, TOP_K), dtype=torch.int32, device=dev), torch.empty((32, TOP_K), dtype=torch.int32, device=dev))
    for b, Ts in RTF_SELF_TEST_T.items():
        msgs = []
        try:
            for T in Ts:
                for k, ls in enumerate((0, 224)):
                    for rep in range(2):
                        e = _rtf_test(state, b, T, ls, 9100 + 31 * T + 7 * k)
                        if e is not None:
                            msgs.append(f"T={T} ls={ls} rep {rep}: {e}")
                            break
        except Exception as ex:
            msgs.append(f"EXC {ex!r}")
        finally:
            _RTF["p"].clear()
            _RTF["req"] = None
            _RTF["ls"] = None
            _RTF["ctr"].zero_()
        _RTF["ok"][b] = not msgs
        print(f"[rtf] self-test bucket {b}: {'ok (bitwise tid / lid / tw / xn vs the tail)' if not msgs else 'FALLBACK ' + msgs[0]}",
              flush=True)
'''

PY_PATCHES = [
    ("\n_GD_CAT_SEEN = set()\n", "\n" + PY_GLOBALS + "_GD_CAT_SEEN = set()\n", "RTF globals"),
    ("\n\ndef _gd_cat_live(state, tokens):\n", PY_FUNCS + "\n\ndef _gd_cat_live(state, tokens):\n", "RTF python functions"),
    ('functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]',
     'functions=["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat", "gd_rtf_arm", "gd_rtf_router"]',
     "GD module exports gd_rtf_arm / gd_rtf_router"),
    # _gd: the T = 1 pooler request (norm half)
    ("        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)\n"
     "        r[\"mod\"].gd_run(r[\"inst\"], r[\"w\"], r[\"esc\"], x, out, T, r[\"K\"], ldo, r[\"tb\"], r[\"nbx\"], 1, pdl)\n",
     "        pdl = (1 if GD_PDL.get(proj) else 0) | (2 if GD_TRIG.get(proj) else 0)\n"
     "        rq = _RTF[\"req\"].pop(\"pool\", None) if (proj == \"pooler\" and _RTF[\"req\"]) else None\n"
     "        if rq is not None:\n"
     "            _rtf_arm(r[\"mod\"], state, 0, rq[0], 2, _RTF[\"ls\"])\n"
     "        r[\"mod\"].gd_run(r[\"inst\"], r[\"w\"], r[\"esc\"], x, out, T, r[\"K\"], ldo, r[\"tb\"], r[\"nbx\"], 1, pdl)\n"
     "        if rq is not None and rq[1] == 2:\n"
     "            _RTF[\"p\"][\"xn\"] = (out.data_ptr(), rq[0][0])\n",
     "_gd arms the T = 1 pooler rider"),
    # _v8_logits: the T = 1 lean router with its top-8 last arriver
    ("        if kind != \"r32\" and _VF[\"r8\"] and _SBB1[\"mod\"] is not None:\n"
     "            _SBB1[\"mod\"].vf_router32(router_input, state[\"router\"], part, {\"lean4\": 1, \"lean5\": 2}.get(kind, 0))\n",
     "        if kind == \"lean\" and router_input.shape[0] == 1 and _rtf_router(state, router_input, part):\n"
     "            pass\n"
     "        elif kind != \"r32\" and _VF[\"r8\"] and _SBB1[\"mod\"] is not None:\n"
     "            _SBB1[\"mod\"].vf_router32(router_input, state[\"router\"], part, {\"lean4\": 1, \"lean5\": 2}.get(kind, 0))\n",
     "_v8_logits: T = 1 router rider"),
    # _v8_tail: hand the rider's outputs over when both halves were produced for exactly these (y, logits)
    ("def _v8_tail(y, logits, state, data):\n    v8 = state[\"v8\"]\n",
     "def _v8_tail(y, logits, state, data):\n"
     "    pr = _RTF[\"p\"]\n"
     "    if pr:\n"
     "        a, c = pr.pop(\"xn\", None), pr.pop(\"rt\", None)\n"
     "        if a is not None and c is not None and logits is not None and a[0] == y.data_ptr() and c[0] == logits.data_ptr():\n"
     "            return a[1], c[1], c[2], c[3]\n"
     "    v8 = state[\"v8\"]\n",
     "_v8_tail: RTF hand-off"),
    # _vfr_gdr: T = 2..8 host
    ("    mk = _VFR[\"marks\"] if VF_GDM else _VFR[\"nomk\"]\n"
     "    r[\"mod\"].gdr_run(vf, r[\"w\"], r[\"esc\"], x, y, T, r[\"K\"], r[\"N\"], r[\"tb\"], r[\"nbx\"], pdl,\n"
     "                     z if vf == 2 else _VFR[\"nomk\"], state[\"router\"], part, mk)\n",
     "    mk = _VFR[\"marks\"] if VF_GDM else _VFR[\"nomk\"]\n"
     "    rm = _rtf_on(T) if vf == 2 else 0\n"
     "    routs = _rtf_outs(state, T, rm) if rm else None\n"
     "    if routs is not None:\n"
     "        _rtf_arm(r[\"mod\"], state, 2, routs, 3, _RTF[\"ls\"])\n"
     "    r[\"mod\"].gdr_run(vf, r[\"w\"], r[\"esc\"], x, y, T, r[\"K\"], r[\"N\"], r[\"tb\"], r[\"nbx\"], pdl,\n"
     "                     z if vf == 2 else _VFR[\"nomk\"], state[\"router\"], part, mk)\n"
     "    if rm == 2:\n"
     "        _RTF[\"p\"][\"xn\"] = (y.data_ptr(), routs[0])\n"
     "        _RTF[\"p\"][\"rt\"] = (part.data_ptr(), routs[1], routs[2], routs[3])\n",
     "_vfr_gdr: T = 2..8 rider"),
    # _gd_cat: T = 9..32 host
    ("        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n",
     "        part = torch.empty((8, tokens, N_EXPERTS), dtype=torch.float32, device=x.device)\n"
     "        rm = _rtf_on(tokens)\n"
     "        routs = _rtf_outs(state, tokens, rm) if rm else None\n"
     "        if routs is not None:\n"
     "            _rtf_arm(r[\"mod\"], state, 3 if _gd_bucket(tokens) == 16 else 4, routs, 3, _RTF[\"ls\"])\n"
     "        _gd_cat_run(r, x, y, part, tokens, GD_CAT_PDL)\n"
     "        if rm == 2:\n"
     "            _RTF[\"p\"][\"xn\"] = (y.data_ptr(), routs[0])\n"
     "            _RTF[\"p\"][\"rt\"] = (part.data_ptr(), routs[1], routs[2], routs[3])\n",
     "_gd_cat: T = 9..32 rider"),
    # forward: per-call reset + the T = 1 request
    ("    mode, place = _v8_router_mode(tokens, state)\n",
     "    _RTF[\"p\"].clear()\n    _RTF[\"req\"] = None\n    _RTF[\"ls\"] = data[\"local_start\"] if _RTF[\"ok\"] else None\n    _RTF[\"live_ok\"] = True\n"
     "    mode, place = _v8_router_mode(tokens, state)\n",
     "forward: RTF per-call state"),
    ("            if place != \"side_first\":\n"
     "                expert_input = _v13_pooler(ffn_input, dense, state)\n"
     "            with torch.cuda.stream(side_stream):\n",
     "            if place != \"side_first\":\n"
     "                _rtf_req1(state, tokens)\n"
     "                expert_input = _v13_pooler(ffn_input, dense, state)\n"
     "                if _RTF[\"req\"]:\n"
     "                    _RTF[\"req\"].pop(\"pool\", None)\n"
     "            with torch.cuda.stream(side_stream):\n",
     "forward: T = 1 pooler + router request"),
    ("                    tensor.record_stream(main_stream)\n",
     "                    tensor.record_stream(main_stream)\n"
     "            _RTF[\"req\"] = None\n",
     "forward: T = 1 request cleared after the join"),
    ("    torch.cuda.synchronize()\n    _pt1_join_all()\n",
     "    torch.cuda.synchronize()\n    _pt1_join_all()\n    _rtf_prepare(state)\n",
     "prepare: RTF self-tests"),
]


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = (f"# nonce {TAG} ({BASE.stem} + RTF: routing tail folded into the producer epilogue by a last-arriver CTA at T <= 32, "
             "RTF_MODE per bucket read at call time, the separate tail as fallback)")
    t = b.replace(bn, nonce)
    a0 = t.index("GD_CUDA_SRC = r'''")
    a1 = t.index(NL + "'''" + NL, a0)
    c = t[a0:a1]
    for a, b2, what in CUDA_PATCHES:
        GM.check(c.count(a) == 1, f"anchor for {what}: found {c.count(a)}")
        c = c.replace(a, b2)
        rep.append(what)
    t = t[:a0] + c + t[a1:]
    cpp_anchor = "void gd_unpack_cat(torch::Tensor w, torch::Tensor esc, torch::Tensor out, int64_t K, int64_t G, int64_t wk, int64_t kup,\n"
    GM.check(t.count(cpp_anchor) == 1, f"gd_unpack_cat binding: found {t.count(cpp_anchor)}")
    t = t.replace(cpp_anchor, CPP_ADD + cpp_anchor)
    rep.append("CPP bindings gd_rtf_arm / gd_rtf_router")
    for a, b2, what in PY_PATCHES:
        GM.check(t.count(a) == 1, f"anchor for {what}: found {t.count(a)}")
        t = t.replace(a, b2)
        rep.append(what)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_rtf] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_rtf] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate the CR board file kernels/muse_moe_w4a16/cr_2.py: the clean-room MoE with the plan computed ON DEVICE
(every CTA derives the routing table, its equal-cost stage range and its token lists from topk_ids), so kernel_fn is three
graph-capturable launches: K1 (W13 + SwiGLU, per-token X row copies) -> PDL -> K2 (W2 + post-norm partials) -> PDL -> K3.
Device helpers (decode math, barriers, consume_piece<NG>) are taken verbatim from kernels/lab/cr_k1_lab.py's CR_CUDA.

  python kernels/lab/cr_gen_board.py [out=kernels/muse_moe_w4a16/cr_2.py]
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAB = ROOT / "kernels/lab/cr_k1_lab.py"

KERNEL = r'''
constexpr int MAXG = 264;
constexpr int NTL = 6;                       // distinct experts per CTA range with token lists (K1)
constexpr int KAPPA = 22;                    // stage cost ticks: max(100, ng * KAPPA) (cold stage = 100)
constexpr int MMAX = 320;                    // tokens per call (list build: one pass of NTH threads)
constexpr int PASSB = 32 * 64 * XRB;         // xp2 bytes of one full (64-token) pass

struct Args2 {
  const __nv_bfloat16* hid; const int* ids; int M, lo;
  const uint8_t* ws; const float* gsc; const uint8_t* xin;
  uint8_t* xp2; float* ss; __nv_bfloat16* ybuf; const float* nw;
  float* scratch; int* cnt; int* err; int* meta;
  int G, phase, nb;
};
struct SPlan {
  int ne, R, nu;
  int ex[32], cn[32], np[32], ub[33], rb[32], xb[32];
  long long cb[33];
  int cuts[MAXG + 1];
};
__device__ __forceinline__ int stage_cost(int ng) { return max(100, ng * KAPPA); }
__device__ __forceinline__ void udec(const SPlan& P, int u, int& j, int& b, int& p, int& ntok, int& ng) {
  j = 0;
  while (j + 1 < P.ne && P.ub[j + 1] <= u) ++j;
  const int r = u - P.ub[j];
  b = r / P.np[j]; p = r % P.np[j];
  ntok = min(64, P.cn[j] - 64 * p); ng = (ntok + 7) >> 3;
}

__global__ void __launch_bounds__(NTH, MINB) cr2_kernel(const Args2 A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ SPlan P;
  __shared__ int s_raw[32];
  __shared__ int s_pu[PMAX], s_ps0[PMAX], s_ps1[PMAX], s_pslot[PMAX], s_pq[PMAX];
  __shared__ int s_tj[NTL], s_wc[NTH / 32];
  __shared__ short s_tok[NTL][MMAX];
  __shared__ int s_np, s_nq, s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, G = A.G, nb = A.nb;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);
  if (tid == 0) {
    if (A.phase == 0) pdl_launch();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, NCW); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
  if (tid < 32) s_raw[tid] = 0;
  __syncthreads();
  // ---------------- on-device plan (identical in every CTA) ----------------
  for (int i = tid; i < A.M * 8; i += NTH) {
    const int e = A.ids[i] - A.lo;
    if ((unsigned)e < 32u) atomicAdd(&s_raw[e], 1);
  }
  __syncthreads();
  if (warp == 0) {
    const int cn = s_raw[lane];
    int rank = 0, nz = 0;
    for (int jj = 0; jj < 32; ++jj) {
      const int cj = __shfl_sync(0xffffffffu, cn, jj);
      nz += cj > 0;
      rank += (cj > cn) || (cj == cn && jj < lane);
    }
    if (cn > 0) { P.ex[rank] = lane; P.cn[rank] = cn; }
    __syncwarp();
    const int cj = lane < nz ? P.cn[lane] : 0;
    const int npj = (cj + 63) >> 6;
    long long bc = 0;
    int xbj = 0;
    for (int p = 0; p < npj; ++p) {
      const int ng = (min(64, cj - 64 * p) + 7) >> 3;
      bc += 32 * stage_cost(ng);
      xbj += 32 * 8 * ng * XRB;
    }
    long long ec = bc * nb;
    int uj = nb * npj, rj = cj, xj = xbj;
    long long ecs = ec;
    int ujs = uj, rjs = rj, xjs = xj;
    for (int o = 1; o < 32; o <<= 1) {
      const long long e2 = __shfl_up_sync(0xffffffffu, ecs, o);
      const int u2 = __shfl_up_sync(0xffffffffu, ujs, o), r2 = __shfl_up_sync(0xffffffffu, rjs, o),
                x2 = __shfl_up_sync(0xffffffffu, xjs, o);
      if (lane >= o) { ecs += e2; ujs += u2; rjs += r2; xjs += x2; }
    }
    P.np[lane] = npj; P.ub[lane] = ujs - uj; P.cb[lane] = ecs - ec; P.rb[lane] = rjs - rj; P.xb[lane] = xjs - xj;
    if (lane == 31) { P.ub[32] = ujs; P.cb[32] = ecs; P.R = rjs; P.nu = ujs; P.ne = nz; }
  }
  __syncthreads();
  for (int i = tid; i <= G; i += NTH) {
    int cut;
    if (i == 0) cut = 0;
    else if (i == G) cut = 32 * P.nu;
    else {
      const long long tau = P.cb[32] * i / G;
      int j = 0;
      while (j + 1 < P.ne && P.cb[j + 1] <= tau) ++j;
      const long long bcj = (P.cb[j + 1] - P.cb[j]) / nb;
      long long rem = tau - P.cb[j];
      const int b = (int)min((long long)(nb - 1), rem / bcj);
      rem -= (long long)b * bcj;
      cut = 32 * P.nu;
      for (int p = 0; p < P.np[j]; ++p) {
        const int ng = (min(64, P.cn[j] - 64 * p) + 7) >> 3;
        const int sc = stage_cost(ng);
        if (rem < 32ll * sc || p == P.np[j] - 1) {
          const int s = (int)min(32ll, (rem + sc / 2) / sc);
          cut = 32 * (P.ub[j] + b * P.np[j] + p) + s;
          break;
        }
        rem -= 32ll * sc;
      }
    }
    P.cuts[i] = cut;
  }
  __syncthreads();
  if (tid == 0) {
    // piece list: partial pieces first (slot 0 = the piece holding cuts[c], slot 1 = the one holding cuts[c+1]-1)
    const int a = P.cuts[c], bnd = P.cuts[c + 1];
    int np = 0;
    if (bnd > a) {
      const int ua = a >> 5, ub = (bnd - 1) >> 5;
      if (ua == ub) {
        const int s0 = a & 31, s1 = bnd - 32 * ua;
        s_pu[0] = ua; s_ps0[0] = s0; s_ps1[0] = s1; s_pslot[0] = (s0 == 0 && s1 == 32) ? -1 : 0; np = 1;
      } else {
        const bool hp = (a & 31) != 0, tp = (bnd - 32 * ub) != 32;
        if (hp) { s_pu[np] = ua; s_ps0[np] = a & 31; s_ps1[np] = 32; s_pslot[np] = 0; ++np; }
        if (tp) { s_pu[np] = ub; s_ps0[np] = 0; s_ps1[np] = bnd - 32 * ub; s_pslot[np] = 1; ++np; }
        for (int u = hp ? ua + 1 : ua; u <= (tp ? ub - 1 : ub); ++u) {
          if (np >= PMAX) { set_err(A.err, 9); break; }
          s_pu[np] = u; s_ps0[np] = 0; s_ps1[np] = 32; s_pslot[np] = -1; ++np;
        }
      }
    }
    int nq = 0;
    for (int q = 0; q < np; ++q) {
      int j, b, p, nt, ng;
      udec(P, s_pu[q], j, b, p, nt, ng);
      int k = 0;
      while (k < nq && s_tj[k] != j) ++k;
      if (k == nq) { if (nq < NTL) s_tj[nq++] = j; else { set_err(A.err, 10); k = 0; } }
      s_pq[q] = k;
    }
    s_np = np; s_nq = nq;
  }
  __syncthreads();
  const int np = s_np;
  if (A.phase == 0) {
    // token lists of this CTA's experts (ascending token order = route order)
    for (int q = 0; q < s_nq; ++q) {
      const int E = P.ex[s_tj[q]];
      bool f = false;
      if (tid < A.M) {
#pragma unroll
        for (int k = 0; k < 8; ++k) f |= (A.ids[tid * 8 + k] - A.lo) == E;
      }
      const unsigned bal = __ballot_sync(0xffffffffu, f);
      if (lane == 0) s_wc[warp] = __popc(bal);
      __syncthreads();
      if (f) {
        int pos = __popc(bal & ((1u << lane) - 1u));
        for (int w2 = 0; w2 < warp; ++w2) pos += s_wc[w2];
        s_tok[q][pos] = (short)tid;
      }
      __syncthreads();
    }
    if (c == 0 && warp == 0) {
      int r = -1;
      for (int j = 0; j < P.ne; ++j) if (P.ex[j] == lane) r = P.rb[j];
      A.meta[lane] = r;
    }
    for (int i = c * NTH + tid; i < P.R; i += G * NTH) A.ss[i] = 0.f;
  }
  // phase 1: only the producer waits for K1 (after prefetching W2 weights); consumers touch global memory only after
  // consuming slots whose X (K1's inter) was copied after that wait.

  if (warp >= NCW) {
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(CR_PREG) : "memory");
#endif
    if (warp != NCW) return;
    // ---------------- producer ----------------
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0, pre = 0;
    bool dead = false;
    if (A.phase == 1) {
#if __CUDA_ARCH__ >= 900
      // W2 weights do not depend on K1: the first NST slots' weight copies go out before waiting for K1's grid
      for (int p = 0; p < np && pre < NST; ++p) {
        int j, b, pp, ntok, ng;
        udec(P, s_pu[p], j, b, pp, ntok, ng);
        const int e = P.ex[j];
        const bool multi = P.np[j] > 1;
        const uint32_t xbytes = (uint32_t)(8 * ng * XRB);
        for (int s = s_ps0[p]; s < s_ps1[p] && pre < NST; ++s, ++pre) {
          const uint8_t* wsrc = A.ws + ((size_t)((e * (nb / HB) + b / HB) * 32 + s)) * 36864 + (b % HB) * CB;
          const uint32_t dst = ring_a + pre * SLOT;
          if (lane == 0) {
            mbar_expect_tx(full_a + 8 * pre, WB + xbytes);
            if (multi) bulk_g2s(dst, wsrc, WB, full_a + 8 * pre);
            else bulk_g2s_h(dst, wsrc, WB, full_a + 8 * pre, pol);
          }
        }
      }
#endif
      pdl_wait();                                    // K1's packed inter (xp2) and zeroed ss are visible after this
    }
    for (int p = 0; p < np; ++p) {
      int j, b, pp, ntok, ng;
      udec(P, s_pu[p], j, b, pp, ntok, ng);
      const int e = P.ex[j];
      const bool multi = P.np[j] > 1;
      const int xoff = P.xb[j] + pp * PASSB;
      const short* tl = s_tok[s_pq[p]] + 64 * pp;
      const uint32_t xbytes = A.phase == 0 ? (uint32_t)(256 * ntok) : (uint32_t)(8 * ng * XRB);
      for (int s = s_ps0[p]; s < s_ps1[p]; ++s, ++it) {
        const int slot = it % NST;
        if (it >= NST && !dead && !mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2)) dead = true;
        const uint8_t* wsrc = A.ws + ((size_t)((e * (nb / HB) + b / HB) * 32 + s)) * 36864 + (b % HB) * CB;
        const uint32_t dst = ring_a + slot * SLOT;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          if (it >= pre) {
            mbar_expect_tx(full_a + 8 * slot, WB + xbytes);
            if (multi) bulk_g2s(dst, wsrc, WB, full_a + 8 * slot);
            else bulk_g2s_h(dst, wsrc, WB, full_a + 8 * slot, pol);
          }
          if (A.phase == 1) bulk_g2s(dst + WB, A.xin + (size_t)xoff + (size_t)s * xbytes, xbytes, full_a + 8 * slot);
        }
        if (A.phase == 0) {
          __syncwarp();
          for (int r = lane; r < ntok; r += 32)
            bulk_g2s(dst + WB + r * XRB, A.hid + (size_t)tl[r] * 4096 + s * 128, 256, full_a + 8 * slot);
        }
#else
        warp_copy(lane, dst, wsrc, WB);
        if (A.phase == 1) warp_copy(lane, dst + WB, A.xin + (size_t)xoff + (size_t)s * xbytes, xbytes);
        else for (int r = 0; r < ntok; ++r) warp_copy(lane, dst + WB + r * XRB, A.hid + (size_t)tl[r] * 4096 + s * 128, 256);
        cpa_arrive(full_a + 8 * slot);
#endif
      }
    }
    return;
  }

  // ---------------- consumers ----------------
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CR_CREG) : "memory");
#endif
  const int g = lane >> 2, t = lane & 3;
  int it = 0;
  bool dead = false;
  float acc[2][NGMAX][4];
  for (int p = 0; p < np; ++p) {
    const int u = s_pu[p];
    int j, b, pp, ntok, ng;
    udec(P, u, j, b, pp, ntok, ng);
    const int e = P.ex[j], rbase = P.rb[j] + 64 * pp, xoff = P.xb[j] + pp * PASSB;
#pragma unroll
    for (int ti = 0; ti < 2; ++ti)
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][gi][i] = 0.f;
    {
      const int s0 = s_ps0[p], s1 = s_ps1[p];
#define CR_CP(N) consume_piece<N>(smem, full_a, empty_a, A.err, s0, s1, it, dead, false, warp, lane, nullptr, tid, acc)
      switch (ng) {
        case 1: CR_CP(1); break;
        case 2: CR_CP(2); break;
        case 3: CR_CP(3); break;
        case 4: CR_CP(4); break;
        case 5: CR_CP(5); break;
        case 6: CR_CP(6); break;
        case 7: CR_CP(7); break;
        default: CR_CP(8); break;
      }
#undef CR_CP
    }
    const int pslot = s_pslot[p];
    const float gs = A.gsc[e];
    bool fin = pslot < 0;
    if (!fin) {
      float* my = A.scratch + ((size_t)(c * 2 + pslot)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
      for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int gi = 0; gi < NGMAX; ++gi)
          if (gi < ng) {
#pragma unroll
            for (int i = 0; i < 4; ++i) my[((ti * NGMAX + gi) * 4 + i) * 32 + lane] = acc[ti][gi][i];
          }
      bar_sync(1, 32 * NCW);
      if (tid == 0) {
        __threadfence();
        const int n = s_ps1[p] - s_ps0[p];
        const int old = atomicAdd(A.cnt + u, n);
        s_last = (old + n == 32);
        if (old + n > 32) set_err(A.err, 5);
        __threadfence();
      }
      bar_sync(1, 32 * NCW);
      fin = s_last != 0;
      if (fin) {
        const int lo = 32 * u, hi = 32 * u + 32;
        int c0 = 0, c1 = G;
        while (c0 < c1) { const int m = (c0 + c1) >> 1; if (P.cuts[m + 1] > lo) c1 = m; else c0 = m + 1; }
        for (int cc = c0; cc < G && P.cuts[cc] < hi; ++cc) {
          if (cc == c || P.cuts[cc + 1] <= P.cuts[cc]) continue;
          const int os = ((P.cuts[cc] >> 5) == u) ? 0 : 1;
          const float* o = A.scratch + ((size_t)(cc * 2 + os)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int gi = 0; gi < NGMAX; ++gi)
              if (gi < ng) {
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[ti][gi][i] += __ldcg(o + ((ti * NGMAX + gi) * 4 + i) * 32 + lane);
              }
        }
        if (tid == 0) A.cnt[u] = 0;
      }
    }
    if (fin && A.phase == 0) {
      __nv_bfloat16* x2 = reinterpret_cast<__nv_bfloat16*>(A.xp2 + xoff);
      const int sstr = 8 * ng * (XRB / 2);
      const int d0 = 16 * NCW * b + 16 * warp;
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int tok = 8 * gi + 2 * t + (i & 1);
            const int dim = d0 + g + 8 * (i >> 1);
            if (tok < ntok)
              x2[(size_t)(dim >> 7) * sstr + tok * (XRB / 2) + (dim & 127)] =
                  __float2bfloat16_rn(silu_mul(acc[0][gi][i] * gs, acc[1][gi][i] * gs));
          }
        }
    }
    if (fin && A.phase == 1) {
      const int o0 = 32 * NCW * b + 32 * warp;
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
          float q0 = 0.f, q1 = 0.f;
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = 8 * gi + 2 * t + (i & 1);
              const int row = o0 + 16 * ti + g + 8 * (i >> 1);
              const float yb = bf16r(bf16r(acc[ti][gi][i] * gs) * A.nw[row]);
              if (i & 1) q1 += yb * yb; else q0 += yb * yb;
              if (tok < ntok) A.ybuf[(size_t)(rbase + tok) * 4096 + row] = __float2bfloat16_rn(yb);
            }
#pragma unroll
          for (int m = 4; m < 32; m <<= 1) {
            q0 += __shfl_xor_sync(0xffffffffu, q0, m);
            q1 += __shfl_xor_sync(0xffffffffu, q1, m);
          }
          const int tk = 8 * gi + 2 * t;
          if (g == 0) {
            if (tk < ntok) atomicAdd(A.ss + rbase + tk, q0);
            if (tk + 1 < ntok) atomicAdd(A.ss + rbase + tk + 1, q1);
          }
        }
    }
  }
}

// K3: out[t] = sum over t's local routes of w * rsqrt(mean(yb^2) + eps) * yb; route = meta[E] + rank of t among E's tokens
__global__ void __launch_bounds__(256) cr2_fin(const __nv_bfloat16* ybuf, const float* ss, const int* ids, const float* wts,
                                               const int* meta, __nv_bfloat16* out, int M, int lo) {
  pdl_wait();
  __shared__ int s_r[8];
  __shared__ float s_w[8];
  const int tt = blockIdx.x, tid = threadIdx.x, d = blockIdx.y * 1024 + tid * 4;
  for (int k = 0; k < 8; ++k) {
    const int E = ids[tt * 8 + k] - lo;
    const bool loc = (unsigned)E < 32u;
    int n = 0;
    for (int base = 0; base < tt; base += 256) {
      const int t2 = base + tid;
      bool f = false;
      if (loc && t2 < tt) {
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) f |= (ids[t2 * 8 + k2] - lo) == E;
      }
      n += __syncthreads_count(f);
    }
    if (tid == 0) { s_r[k] = loc ? meta[E] + n : -1; s_w[k] = wts[tt * 8 + k]; }
  }
  __syncthreads();
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < 8; ++k) {
    const int r = s_r[k];
    if (r < 0) continue;
    const float sc = s_w[k] * rsqrtf(ss[r] * (1.f / 4096.f) + 1e-8f);
    const uint2 v = *reinterpret_cast<const uint2*>(ybuf + (size_t)r * 4096 + d);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&v.x), b2 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
    o[0] += sc * __low2float(a); o[1] += sc * __high2float(a); o[2] += sc * __low2float(b2); o[3] += sc * __high2float(b2);
  }
  __nv_bfloat162 r0 = __floats2bfloat162_rn(o[0], o[1]), r1 = __floats2bfloat162_rn(o[2], o[3]);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&r0); w.y = *reinterpret_cast<uint32_t*>(&r1);
  *reinterpret_cast<uint2*>(out + (size_t)tt * 4096 + d) = w;
}
}  // namespace crk

template <typename K, typename... Args>
static void cr_launch(K kern, dim3 grid, dim3 block, size_t smem, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

void cr2_call(torch::Tensor hid, torch::Tensor ids, torch::Tensor wts, torch::Tensor out, torch::Tensor ws13,
              torch::Tensor gsc13, torch::Tensor ws2, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor xp2,
              torch::Tensor ss, torch::Tensor ybuf, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err,
              torch::Tensor meta, int64_t G, int64_t lo, int64_t pdl) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
    attr = true;
  }
  const int M = (int)hid.size(0);
  crk::Args2 A = {};
  A.hid = reinterpret_cast<const __nv_bfloat16*>(hid.data_ptr<at::BFloat16>()); A.ids = ids.data_ptr<int>();
  A.M = M; A.lo = (int)lo;
  A.ws = ws13.data_ptr<uint8_t>(); A.gsc = gsc13.data_ptr<float>(); A.xin = nullptr; A.xp2 = xp2.data_ptr<uint8_t>();
  A.ss = ss.data_ptr<float>(); A.ybuf = nullptr; A.nw = nullptr; A.scratch = scratch.data_ptr<float>();
  A.cnt = cnt.data_ptr<int>(); A.err = err.data_ptr<int>(); A.meta = meta.data_ptr<int>();
  A.G = (int)G; A.phase = 0; A.nb = 16;
  cr_launch(crk::cr2_kernel, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, false, A);
  crk::Args2 B = A;
  B.ws = ws2.data_ptr<uint8_t>(); B.gsc = gsc2.data_ptr<float>(); B.xin = xp2.data_ptr<uint8_t>(); B.xp2 = nullptr;
  B.ybuf = reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr<at::BFloat16>()); B.nw = nw.data_ptr<float>();
  B.cnt = cnt.data_ptr<int>() + 4096; B.phase = 1; B.nb = 8;
  cr_launch(crk::cr2_kernel, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, pdl != 0, B);
  cr_launch(crk::cr2_fin, dim3((unsigned)M, 4), dim3(256), 0, pdl != 0, (const __nv_bfloat16*)B.ybuf,
            (const float*)A.ss, (const int*)A.ids, (const float*)wts.data_ptr<float>(), (const int*)A.meta,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), M, (int)lo);
}

std::vector<int64_t> cr2_attrs() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, crk::cr2_kernel));
  C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, crk::cr2_kernel, crk::NTH, crk::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)crk::SMEM, (int64_t)occ};
}
'''

CPP = r'''
#include <torch/extension.h>
#include <vector>
void cr2_call(torch::Tensor hid, torch::Tensor ids, torch::Tensor wts, torch::Tensor out, torch::Tensor ws13,
              torch::Tensor gsc13, torch::Tensor ws2, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor xp2,
              torch::Tensor ss, torch::Tensor ybuf, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err,
              torch::Tensor meta, int64_t G, int64_t lo, int64_t pdl);
std::vector<int64_t> cr2_attrs();
'''

PY = r'''

CR2_CPP = {cpp!r}
CR2_ARCH = "sm_90a"
CR2_M_MAX = 320
CR2_SELF_TEST = True


def _cr2_build():
    from torch.utils.cpp_extension import load_inline
    gen = (["-gencode=arch=compute_90a,code=sm_90a", "-DCR_NST=4"] if CR2_ARCH == "sm_90a"
           else ["-gencode=arch=compute_86,code=sm_86", "-DCR_NST=1"])
    tag = hashlib.sha1((CR2_CPP + CR2_CUDA + CR2_ARCH).encode()).hexdigest()[:10]
    return load_inline(name=f"cr2_{{tag}}", cpp_sources=CR2_CPP, cuda_sources=CR2_CUDA, functions=["cr2_call", "cr2_attrs"],
                       extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-DCR_NGMAX=8", "-DCR_NCW=16"] + gen)


def _cr2_ref(d, w):
    x = d["hidden_states"].float()
    ids = d["topk_ids"].long()
    lo = int(d["config"]["local_start"])
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < lo + 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ cr_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ cr_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y.to(torch.bfloat16).float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * d["topk_weights"][tok, slot].float().unsqueeze(1))
    return out


def _cr2_case(bs, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {{"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {{"bs": bs, "local_start": 0, "local_count": 32}}}}


def prepare(weights):
    dev = weights["w13"].device
    t0 = time.time()
    mod = _cr2_build()
    a = list(mod.cr2_attrs())
    print(f"[cr2] built in {{time.time() - t0:.0f}}s; attrs (regs, local, smem, occ) {{a}}", flush=True)
    if CR2_ARCH == "sm_90a" and (a[0] != 96 or a[3] < 1):
        raise RuntimeError(f"cr2: launch registers {{a[0]}} != 96 or occupancy {{a[3]}} (setmaxnreg 24 / 112 budget)")
    ws13, gsc13 = cr_prepack(weights["w13"][:32], weights["w13_sf"][:32], weights["w13_gs"][:32])
    ws2, gsc2 = cr_prepack_w2(weights["w2"][:32], weights["w2_sf"][:32], weights["w2_gs"][:32])
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    st = {{"mod": mod, "ws13": ws13, "gsc13": gsc13, "ws2": ws2, "gsc2": gsc2, "G": G,
          "nw": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
          "xp2": torch.zeros(((CR2_M_MAX + 32) * 32 * 8 * 272,), dtype=torch.uint8, device=dev),
          "ss": torch.zeros((8 * CR2_M_MAX,), dtype=torch.float32, device=dev),
          "ybuf": torch.zeros((8 * CR2_M_MAX, 4096), dtype=torch.bfloat16, device=dev),
          "scratch": torch.zeros((G * 2 * CR_PART,), dtype=torch.float32, device=dev),
          "cnt": torch.zeros((8192,), dtype=torch.int32, device=dev),
          "err": torch.zeros((4,), dtype=torch.int32, device=dev),
          "meta": torch.zeros((64,), dtype=torch.int32, device=dev)}}
    if CR2_SELF_TEST:
        for bs, seed in ((1, 31), (8, 32), (32, 33), (128, 34), (256, 35)):
            d = _cr2_case(bs, 8800 + seed, dev)
            ref = _cr2_ref(d, weights)
            for call in range(2):
                out = kernel_fn(d, st)
                torch.cuda.synchronize()
                diff = out.float() - ref
                rel = float(diff.norm() / ref.norm().clamp(min=1e-30))
                rn = ref.norm(dim=1)
                worst = float((diff.norm(dim=1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
                errv = [int(v) for v in st["err"].tolist()]
                cl = int(st["cnt"].abs().sum())
                print(f"[cr2] self-test bs={{bs}} call {{call}}: rel {{rel:.3e}} worst row {{worst:.3e}} err {{errv}} "
                      f"counters {{cl}}", flush=True)
                if not (rel <= 1e-2 and worst <= 5e-2 and errv[0] == 0 and cl == 0):
                    raise RuntimeError(f"cr2 self-test failed at bs={{bs}}: rel {{rel}} worst {{worst}} err {{errv}}")
    return st


def kernel_fn(data, state):
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].float().contiguous()
    M = x.shape[0]
    if M > CR2_M_MAX:
        raise RuntimeError("cr2: M > CR2_M_MAX")
    out = torch.empty((M, 4096), dtype=torch.bfloat16, device=x.device)
    st = state
    st["mod"].cr2_call(x, ids, wts, out, st["ws13"], st["gsc13"], st["ws2"], st["gsc2"], st["nw"], st["xp2"], st["ss"],
                       st["ybuf"], st["scratch"], st["cnt"], st["err"], st["meta"], st["G"],
                       int(data["config"]["local_start"]), 1)
    return out
'''

HEADER = '''# okc: problem=muse_moe_w4a16
"""CR (clean-room MoE, notes/study/moe_cleanroom.md): NVFP4 W4A16 experts as three PDL-chained launches with the plan built
on device. K1: 1 CTA/SM x 16 mma.sync consumer warps + a bulk-copy producer, 512-row x 128-K slots, all tokens of an
expert per weight pass (<= 64), equal-cost contiguous stage ranges (partial pieces first, last-arriver finish), SwiGLU
in registers, inter written in K2's X layout. K2: the same consumer on W2, y -> bf16 -> x (1 + w) -> bf16, per-token
sum of squares. K3: weighted RMS-normalized sum. Numerics: E2M1 x E4M3 exact in BF16, FP32 accumulate, SGLang's
bf16 roundings. Generated by kernels/lab/cr_gen_board.py."""
import hashlib
import math
import time

import torch
'''


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kernels/muse_moe_w4a16/cr_2.py"
    src = LAB.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cu = None
    funcs = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CR_CUDA":
            cu = n.value.value
        if isinstance(n, (ast.FunctionDef, ast.Assign)):
            seg = ast.get_source_segment(src, n)
            name = n.name if isinstance(n, ast.FunctionDef) else getattr(n.targets[0], "id", "")
            funcs[name] = seg
    pre = cu[:cu.index("struct Args {")]
    helpers = cu[cu.index("__device__ __forceinline__ void deq4"):cu.index("__global__ void __launch_bounds__(NTH, MINB) cr_k1")]
    cuda = pre + helpers + KERNEL
    py_lab = "\n\n".join(funcs[k] for k in ("CR_WB", "CR_XRB", "CR_PART", "CR_BITPOS3", "_CR_GRID", "cr_dequant",
                                             "cr_reencode_scales", "cr_pack_words", "cr_prepack_rows", "cr_prepack",
                                             "cr_prepack_w2"))
    text = HEADER + "\nCR2_CUDA = r\"\"\"" + cuda + "\"\"\"\n\n" + py_lab + PY.format(cpp=CPP)
    ast.parse(text)
    for bad in ("importlib", "sys.modules"):
        assert bad not in text, bad
    assert len(text.encode()) < 1_000_000
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({len(text.encode())} bytes)")


if __name__ == "__main__":
    main()

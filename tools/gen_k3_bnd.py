"""gen_k3_bnd.py: BND = layer boundaries folded into the GD epilogues at T <= 32 (notes/study/xkernel_review.md #1, #2).

  python -I tools/gen_k3_bnd.py --n=<n> [--base=kernels/muse_decode_layer/k3_gc_16.py] [--unp=2,2,2,2] [--o=2,2,2]
        [--marks] [--dry]          -> kernels/lab/k3_bnd_<n>.py (k3_bndlat_<n>.py with --marks: GD / sbb1 marks of lat_patches)

#1 final norm (ffn boundary) into the GD unpooler (buckets 1 / 8 / 16 / 32), #2 attn boundary (+ FP32 router input at T <= 8)
into the GD o projection (buckets 1 / 8 / 16; o at T = 32 is not GD). Compile-time gd_kernel<..., BND>: BND = phase | sliced << 2.
Each CTA rounds its GEMV outputs to bf16 (= today's branch), computes SGLang's per-8-column fmaf chains (post / residual sums),
writes them to scratch, one grid barrier (one-wave grid -> co-resident; bounded wait, no trap), then reduces in SGLang's order
(xor trees per 256-column block, block pairs per rank, 16 ranks in order) -> bitwise with today's sbb / SGLang boundary.
Phase 1: every CTA reduces every token; phase 2: CTA t reduces token t, then a flag counter. The sliced unpooler (T <= 8, 1024
GEMV columns over 64 CTAs) also owns the 28 residual-only 256-column blocks (CTA b < 28 -> block 4 + b).
Modes per bucket at call time: BND_UNP / BND_O {bucket: 0 off, 1 rider (fused outputs to scratch, today's boundary kept), 2 fused,
3 rider stopped after the partial words}. Round 2 (R-bnd-0b): no grid barrier: partials are (epoch << 32 | f32) words polled
directly (epoch = per-launch ticket taken before the GEMV), stream / weights cp.async'd before the GEMV main loop.
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def check(c, msg):
    if not c:
        raise SystemExit(f"[gen_k3_bnd] FAIL: {msg}")


def sub(t, old, new, count=1, what=""):
    n = t.count(old)
    check(n == count, f"anchor {what or old[:70]!r}: found {n}, expected {count}")
    return t.replace(old, new)


DEV = r'''__device__ __forceinline__ uint32_t smem_u32'''
DEV_NEW = r'''__device__ __forceinline__ void bnd_ld2(const unsigned long long* p, unsigned long long& a, unsigned long long& b) {
  asm volatile("ld.relaxed.gpu.global.v2.b64 {%0,%1}, [%2];\n" : "=l"(a), "=l"(b) : "l"(p) : "memory");
}
__device__ __forceinline__ void bnd_st2(unsigned long long* p, unsigned long long a, unsigned long long b) {
  asm volatile("st.relaxed.gpu.global.v2.b64 [%0], {%1,%2};\n" :: "l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ unsigned long long bnd_pk(unsigned ep, float v) {
  return ((unsigned long long)ep << 32) | (unsigned long long)__float_as_uint(v);
}
__device__ __forceinline__ bool bnd_ok(unsigned long long w, unsigned ep) { return (unsigned)(w >> 32) == ep; }
__device__ __forceinline__ float bnd_val(unsigned long long w) { return __uint_as_float((unsigned)w); }
__device__ __forceinline__ void bnd_cp4(void* sdst, const void* gsrc) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void bnd_tree(float4& a) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    a.x += __shfl_xor_sync(0xffffffffu, a.x, o);
    a.y += __shfl_xor_sync(0xffffffffu, a.y, o);
    a.z += __shfl_xor_sync(0xffffffffu, a.z, o);
    a.w += __shfl_xor_sync(0xffffffffu, a.w, o);
  }
}
__device__ __forceinline__ float bnd_rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ uint32_t smem_u32'''

ARGS_OLD = "  void* out2 = nullptr; int ldo2 = 0; long long sst2 = 0; uint32_t tbr = 0;"
ARGS_NEW = ARGS_OLD + r'''
  const float* bst = nullptr; void* bout = nullptr; float* bns = nullptr; float* brt = nullptr;   // BND: stream in, outputs
  const uint16_t* bpw = nullptr; const uint16_t* bqw = nullptr; const float* bal = nullptr; const float* bbt = nullptr;
  unsigned long long* bpart = nullptr; unsigned long long* brs = nullptr; unsigned* bctr = nullptr; int* berr = nullptr;
  float bpe = 0.f, bqe = 0.f; int bgcw = 0, bisc = 0, bnob = 0, bstop = 0;   // bpart / brs: (epoch << 32 | f32) words'''

TPL_OLD = "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0>\n__global__"
TPL_NEW = "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0, int CAT = 0, int BND = 0>\n__global__"

SMEM_OLD = "  __shared__ __align__(8) unsigned long long mbar;\n"
SMEM_NEW = SMEM_OLD + r'''  constexpr int BPH = BND & 3, BSL = (BND >> 2) & 1;
  constexpr int BNC = WR * 16, BNT = MT * 8, BNX = BNC + (BSL ? 256 : 0), BNG = BNX / 8;
  static_assert(BND == 0 || ((BPH == 1 || BPH == 2) && S == 1 && CAT == 0 && VF == 0 && OUT == 0), "BND: S 1 bf16 GD only");
  __shared__ float bnd_v[BND ? BNT : 1][BND ? BNX : 1];
  __shared__ __align__(16) float bnd_r[BND ? BNT : 1][BND ? BNX : 1];
  __shared__ __align__(16) float bnd_w[BND ? 4 : 1][BND ? BNX : 1];
  __shared__ __align__(16) uint16_t bnd_h[BND ? 2 : 1][BND ? BNX : 1];
  __shared__ float4 bnd_g[BND ? BNT : 1][BND ? BNG : 1];
  __shared__ float4 bnd_b[(BND && BPH == 1) ? BNT : 1][BND ? 32 : 1];
  __shared__ float bnd_s[BND ? BNT : 1][2];
  __shared__ unsigned bnd_ep;
  unsigned bep = 0;
'''

EPI_OLD = "  if constexpr (S == 1) {\n    for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));\n"
EPI_NEW = r'''  if constexpr (S == 1) {
    if constexpr (BND == 0) {
      for (int i = (int)threadIdx.x; i < NO; i += NT) store(i, cta_value(i));
    } else {   // BND: the layer boundary on this CTA's columns, SGLang's arithmetic and reduction order
      const int tid = (int)threadIdx.x, bx = (int)blockIdx.x, nbx = (int)gridDim.x;
      constexpr int nsp = BSL ? 4 : 32, U = nsp + BSL;   // split 256-column blocks (+ the residual-only blocks' unit)
      const bool hx = BSL && bx < 32 - nsp;
      const int c0 = bx * BNC, cx = (nsp + bx) * 256, ngo = hx ? BNG : BNC / 8;
      if (tid == 0) bnd_ep = bep / (unsigned)nbx + 1u;
      for (int i = tid; i < NO; i += NT) {
        const float v = cta_value(i);
        const int l = i & 31, c = (i >> 5) & 3, r = i >> 7;
        bnd_v[(r % MT) * 8 + 2 * (l & 3) + (c & 1)][(r / MT) * 16 + (l >> 2) + 8 * (c >> 1)] = bnd_rbf(v);
        if (!A.bnob) store(i, v);
      }
      if (hx)
        for (int i = tid; i < BNT * 256; i += NT) bnd_v[i >> 8][BNC + (i & 255)] = 0.f;
      asm volatile("cp.async.wait_all;\n" ::: "memory");
      __syncthreads();
      const unsigned ep = bnd_ep;
      for (int it = tid; it < BNT * BNG; it += NT) {   // 8-column chains (one SGLang thread's elements)
        const int t = it / BNG, g = it - t * BNG;
        if (t >= A.M || g >= ngo) continue;
        const int j0 = g * 8;
        float bs = 0.f, rs = 0.f, cs = 0.f, ss = 0.f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
          const float pw = A.bpw ? __uint_as_float((uint32_t)bnd_h[0][j0 + q] << 16) : 1.f;
          float v = bnd_v[t][j0 + q];
          if (A.bisc) v = bnd_rbf(v * pw);
          const float mv = A.bisc ? v : v * pw;
          const float rr = bnd_w[1][j0 + q] * bnd_r[t][j0 + q];
          const float sb = bnd_w[2][j0 + q] * mv;
          bnd_v[t][j0 + q] = v;
          bnd_r[t][j0 + q] = rr;
          bs = fmaf(v, v, bs);
          rs = fmaf(rr, rr, rs);
          cs = fmaf(rr, sb, cs);
          ss = fmaf(sb, sb, ss);
        }
        if (g < BNC / 8) {
          unsigned long long* p = A.bpart + ((size_t)t * 1056 + c0 / 8 + g) * 4;
          bnd_st2(p, bnd_pk(ep, bs), bnd_pk(ep, rs));
          bnd_st2(p + 2, bnd_pk(ep, cs), bnd_pk(ep, ss));
        } else {
          bnd_g[t][g] = make_float4(bs, rs, cs, ss);
        }
      }
      if (hx) {
        __syncthreads();
        for (int t = warp; t < A.M && t < BNT; t += NW) {   // the SGLang warp tree of the residual-only block
          float4 a = bnd_g[t][BNC / 8 + lane];
          bnd_tree(a);
          if (lane == 0) {
            unsigned long long* p = A.bpart + ((size_t)t * 1056 + 1024 + nsp + bx) * 4;
            bnd_st2(p, bnd_pk(ep, a.x), bnd_pk(ep, a.y));
            bnd_st2(p + 2, bnd_pk(ep, a.z), bnd_pk(ep, a.w));
          }
        }
      }
      if (!A.bstop) {
        const int nun = BPH == 1 ? A.M * U : (bx < A.M ? U : 0);
        for (int base = warp; base < nun; base += 4 * NW) {   // poll the self-validating partial words (no barrier)
          unsigned long long w[4][4];
          bool all = false;
          for (unsigned itr = 0; !all; ++itr) {
            bool ok = true;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              const int un = base + u * NW;
              const int t = BPH == 1 ? un / U : bx, b = BPH == 1 ? un - (un / U) * U : un;
              if (un < nun && (b < nsp || lane < 32 - nsp)) {
                const unsigned long long* p = A.bpart + ((size_t)t * 1056 + (b < nsp ? b * 32 + lane : 1024 + nsp + lane)) * 4;
                bnd_ld2(p, w[u][0], w[u][1]);
                bnd_ld2(p + 2, w[u][2], w[u][3]);
                ok = ok && bnd_ok(w[u][0], ep) && bnd_ok(w[u][1], ep) && bnd_ok(w[u][2], ep) && bnd_ok(w[u][3], ep);
              }
            }
            all = __all_sync(0xffffffffu, ok);
            if (!all && itr > (1u << 20)) {
              if (lane == 0 && A.berr) atomicExch(A.berr, 1);
              break;
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            const int un = base + u * NW;
            if (un < nun) {
              const int t = BPH == 1 ? un / U : bx, b = BPH == 1 ? un - (un / U) * U : un;
              float4 a = make_float4(bnd_val(w[u][0]), bnd_val(w[u][1]), bnd_val(w[u][2]), bnd_val(w[u][3]));
              if (b < nsp) {
                bnd_tree(a);
                if (lane == 0) bnd_b[BPH == 1 ? t : 0][b] = a;
              } else if (lane < 32 - nsp) {
                bnd_b[BPH == 1 ? t : 0][nsp + lane] = a;
              }
            }
          }
        }
        __syncthreads();
        const int ntk = BPH == 1 ? A.M : (bx < A.M ? 1 : 0);
        if (tid < ntk) {
          const int t = BPH == 1 ? tid : bx, tq = BPH == 1 ? tid : 0;
          float tb = 0.f, tr = 0.f, tc = 0.f, ts = 0.f;
          for (int q = 0; q < 16; ++q) {
            float4 m = make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 p0 = bnd_b[tq][2 * q], p1 = bnd_b[tq][2 * q + 1];
            m.x += p0.x; m.y += p0.y; m.z += p0.z; m.w += p0.w;
            m.x += p1.x; m.y += p1.y; m.z += p1.z; m.w += p1.w;
            tb += m.x; tr += m.y; tc += m.z; ts += m.w;
          }
          const float post_rstd = rsqrtf(tb * (1.f / (float)8192) + A.bpe);
          const float pre_sum = fmaf(post_rstd, fmaf(post_rstd, ts, 2.f * tc), tr);
          const float pre_rstd = rsqrtf(fmaxf(pre_sum, 0.f) * (1.f / (float)8192) + A.bqe);
          if (BPH == 1) {
            bnd_s[t][0] = post_rstd;
            bnd_s[t][1] = pre_rstd;
          } else {
            bnd_st2(A.brs + 2 * t, bnd_pk(ep, post_rstd), bnd_pk(ep, pre_rstd));
          }
        }
        if constexpr (BPH == 2) {
          for (int t = tid; t < A.M; t += NT) {   // poll the reducers' scale words
            unsigned long long a0 = 0, a1 = 0;
            for (unsigned itr = 0;; ++itr) {
              bnd_ld2(A.brs + 2 * t, a0, a1);
              if (bnd_ok(a0, ep) && bnd_ok(a1, ep)) break;
              if (itr > (1u << 20)) {
                if (A.berr) atomicExch(A.berr, 1);
                break;
              }
            }
            bnd_s[t][0] = bnd_val(a0);
            bnd_s[t][1] = bnd_val(a1);
          }
        }
        __syncthreads();
        for (int it = tid; it < BNT * BNG; it += NT) {
          const int t = it / BNG, g = it - t * BNG;
          if (t >= A.M || g >= ngo) continue;
          const int j0 = g * 8, col = j0 < BNC ? c0 + j0 : cx + j0 - BNC;
          const float pr = bnd_s[t][0], qr = bnd_s[t][1];
          float nsv[8], zv[8];
          __align__(16) __nv_bfloat16 ob[8];
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            float n = bnd_v[t][j0 + q] * pr;
            if (!A.bisc && A.bpw) n *= __uint_as_float((uint32_t)bnd_h[0][j0 + q] << 16);
            n = bnd_rbf(n);
            const float u = fmaf(bnd_w[2][j0 + q], n, bnd_r[t][j0 + q]);
            nsv[q] = u;
            float z = u * qr;
            if (A.bqw) z *= __uint_as_float((uint32_t)bnd_h[1][j0 + q] << 16);
            zv[q] = z;
            ob[q] = __float2bfloat16_rn(z);
          }
          const size_t o = (size_t)t * 8192 + col;
          *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(A.bout) + o) = *reinterpret_cast<const uint4*>(ob);
          reinterpret_cast<float4*>(A.bns + o)[0] = make_float4(nsv[0], nsv[1], nsv[2], nsv[3]);
          reinterpret_cast<float4*>(A.bns + o)[1] = make_float4(nsv[4], nsv[5], nsv[6], nsv[7]);
          if (A.brt) {
            reinterpret_cast<float4*>(A.brt + o)[0] = make_float4(zv[0], zv[1], zv[2], zv[3]);
            reinterpret_cast<float4*>(A.brt + o)[1] = make_float4(zv[4], zv[5], zv[6], zv[7]);
          }
        }
      }
    }
'''

PRE_OLD = "  if constexpr (VF == 2) {   // the router input slice (the boundary's FP32 output), tokens >= M zero\n"
PRE_NEW = r'''  if constexpr (BND > 0) {   // BND: epoch ticket + cp.async of this CTA's stream rows / gate / norm weights (hidden by the GEMV)
    const int tid = (int)threadIdx.x, bx = (int)blockIdx.x;
    constexpr int nsp = BSL ? 4 : 32;
    const bool hx = BSL && bx < 32 - nsp;
    const int c0 = bx * BNC, cx = (nsp + bx) * 256, nx = hx ? BNX : BNC;
    if (tid == 0) bep = atomicAdd(A.bctr, 1u);
    for (int i = tid; i < A.M * (nx / 4); i += NT) {
      const int t = i / (nx / 4), j = (i - t * (nx / 4)) * 4;
      vf_cp16(&bnd_r[t][j], A.bst + (size_t)t * 8192 + (j < BNC ? c0 + j : cx + j - BNC));
    }
    for (int j = tid * 4; j < nx; j += NT * 4) {
      const int col = j < BNC ? c0 + j : cx + j - BNC;
      if (A.bal && A.bgcw) {
        vf_cp16(&bnd_w[1][j], A.bal + col);
        vf_cp16(&bnd_w[2][j], A.bbt + col);
      } else {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (A.bal) {
            bnd_cp4(&bnd_w[1][j + q], A.bal);
            bnd_cp4(&bnd_w[2][j + q], A.bbt);
          } else {
            bnd_w[1][j + q] = 1.f;
            bnd_w[2][j + q] = 1.f;
          }
        }
      }
    }
    for (int j = tid * 8; j < nx; j += NT * 8) {
      const int col = j < BNC ? c0 + j : cx + j - BNC;
      if (A.bpw) vf_cp16(&bnd_h[0][j], A.bpw + col);
      if (A.bqw) vf_cp16(&bnd_h[1][j], A.bqw + col);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
''' + PRE_OLD

HOST_ANCHOR = "int gd_unpack_cat_raw(const void* w, const void* esc, void* out, int K, int G, int wk, int kup, uint32_t tb, uint32_t tbr,\n                      cudaStream_t st) {"
HOST_NEW = r'''int gdb_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
                int nbx, int pdl, const float* bst, void* bout, float* bns, float* brt, const void* bpw, const void* bqw,
                const float* bal, const float* bbt, int bgcw, float bpe, float bqe, int bisc, void* bpart, void* brs,
                unsigned* bctr, int* berr, int bnob, cudaStream_t st) {
  Args A;
  A.w = reinterpret_cast<const uint4*>(w);
  A.esc = reinterpret_cast<const uint32_t*>(esc);
  A.x = reinterpret_cast<const uint16_t*>(x);
  A.out = out;
  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;
  A.trig = (pdl >> 1) & 1;
  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr; A.bst = bst;
  A.bout = bout; A.bns = bns; A.brt = brt;
  A.bpw = reinterpret_cast<const uint16_t*>(bpw); A.bqw = reinterpret_cast<const uint16_t*>(bqw); A.bal = bal; A.bbt = bbt;
  A.bgcw = bgcw; A.bpe = bpe; A.bqe = bqe; A.bisc = bisc;
  A.bpart = reinterpret_cast<unsigned long long*>(bpart); A.brs = reinterpret_cast<unsigned long long*>(brs); A.bctr = bctr;
  A.berr = berr; A.bnob = bnob & 1; A.bstop = (bnob >> 1) & 1;
  const int nz = 1;
  switch (inst) {
@INST@
    default: return -1;
  }
}

''' + HOST_ANCHOR

CPP_ANCHOR = "void gd_unpack(torch::Tensor w, torch::Tensor esc"
CPP_NEW = r'''int gdb_run_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo, uint32_t tb,
                int nbx, int pdl, const float* bst, void* bout, float* bns, float* brt, const void* bpw, const void* bqw,
                const float* bal, const float* bbt, int bgcw, float bpe, float bqe, int bisc, void* bpart, void* brs,
                unsigned* bctr, int* berr, int bnob, cudaStream_t st);
void gdb_run(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M, int64_t K,
             int64_t ldo, int64_t tb, int64_t nbx, int64_t pdl, torch::Tensor stream, torch::Tensor bout, torch::Tensor bns,
             torch::Tensor brt, torch::Tensor pw, torch::Tensor qw, torch::Tensor al, torch::Tensor bt, double pe, double qe,
             int64_t isc, torch::Tensor part, torch::Tensor rs, torch::Tensor ctr, torch::Tensor err, int64_t nob) {
  auto ok2 = [&](const torch::Tensor& t, at::ScalarType d) {
    return t.scalar_type() == d && t.is_contiguous() && t.dim() == 2 && t.size(0) == M && t.size(1) == 8192 &&
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
  };
  TORCH_CHECK(M >= 1 && ok2(stream, at::kFloat) && ok2(bout, at::kBFloat16) && ok2(bns, at::kFloat) &&
              (brt.numel() == 0 || ok2(brt, at::kFloat)) && part.scalar_type() == at::kLong && rs.scalar_type() == at::kLong &&
              ctr.scalar_type() == at::kInt && ctr.numel() >= 2 && err.scalar_type() == at::kInt &&
              (pw.numel() == 0 || pw.numel() == 8192) && (qw.numel() == 0 || qw.numel() == 8192) &&
              (al.numel() == 0 || al.numel() == 1 || al.numel() == 8192) && al.numel() == bt.numel(),
              "gdb_run: stream / outputs [M, 8192] (f32 / bf16 / f32 [/ f32]), scratch f32 / int32");
  const int e = gdb_run_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), (nob & 1) ? nullptr : out.data_ptr(), (int)M, (int)K,
                            (int)ldo, (uint32_t)tb, (int)nbx, (int)pdl, stream.data_ptr<float>(), bout.data_ptr(),
                            bns.data_ptr<float>(), brt.numel() ? brt.data_ptr<float>() : nullptr,
                            pw.numel() ? pw.data_ptr() : nullptr, qw.numel() ? qw.data_ptr() : nullptr,
                            al.numel() ? al.data_ptr<float>() : nullptr, bt.numel() ? bt.data_ptr<float>() : nullptr,
                            al.numel() == 8192 ? 1 : 0, (float)pe, (float)qe, (int)isc, part.data_ptr(), rs.data_ptr(),
                            reinterpret_cast<unsigned*>(ctr.data_ptr<int>()), err.data_ptr<int>(), (int)nob,
                            at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(e == 0, "gdb_run: inst ", inst, " CUDA error ", e, " ", cudaGetErrorString((cudaError_t)(e < 0 ? 1 : e)));
}
''' + CPP_ANCHOR

KEYS_OLD = "    for key in _gd_cat_keys():\n        if key not in keys:\n            keys.append(key)\n    return keys\n"
KEYS_NEW = ("    for key in _gd_cat_keys():\n        if key not in keys:\n            keys.append(key)\n"
            "    for key in _bnd_keys():\n        if key not in keys:\n            keys.append(key)\n    return keys\n")

UNP_OLD = """    ffn_output = _gd_unpool(state, expert_output, tokens)
    if ffn_output is None:
        ffn_output = _su1(state, "unpooler", expert_output, tokens)
"""
UNP_NEW = """    bnu = _bnd_unpool(state, expert_output, stream, tokens)
    if bnu is not None and bnu[0] == "fused":
        return bnu[1], bnu[2], topk_ids
    ffn_output = bnu[1] if bnu is not None else _gd_unpool(state, expert_output, tokens)
    if ffn_output is None:
        ffn_output = _su1(state, "unpooler", expert_output, tokens)
"""

O_OLD = """    attn_output = _gd(state, "o", o)
    if attn_output is None and ks1_ran:
        attn_output = _su1(state, "o", o, tokens)
    if attn_output is None:
        attn_output = _v20_p1m(state, "o", o)
    if attn_output is None:
        attn_output = _v13_p1(state, "o", o)
    if attn_output is None:
        attn_output = _dg1(state, "o", o)
    if attn_output is None:
        attn_output = _v12_gemm(o, dense["o"], "o", tokens)

    mode, place = _v8_router_mode(tokens, state)
    tail = V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens)) if state.get("v8") is not None else None
    if tail == "cat" and tokens <= ROUTER_FP32_MAX_TOKENS:
        tail = "sep"
    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T or mode not in ("stock", "f32") or tail == "cat"
    ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                              None if split_router else torch.float32, state, tokens)
"""
O_NEW = """    mode, place = _v8_router_mode(tokens, state)
    tail = V8_TAIL_ROUTE.get(_k2a_bucket_of(tokens)) if state.get("v8") is not None else None
    if tail == "cat" and tokens <= ROUTER_FP32_MAX_TOKENS:
        tail = "sep"
    split_router = tokens > ROUTER_FP32_MAX_TOKENS or R1_ROUTER_SPLIT_ALL_T or mode not in ("stock", "f32") or tail == "cat"
    bno = _bnd_o(state, o, data["stream"], None if split_router else torch.float32, tokens)
    if bno is not None and bno[0] == "fused":
        ffn_input, stream, router_input = bno[1]
    else:
        attn_output = bno[1] if bno is not None else _gd(state, "o", o)
        if attn_output is None and ks1_ran:
            attn_output = _su1(state, "o", o, tokens)
        if attn_output is None:
            attn_output = _v20_p1m(state, "o", o)
        if attn_output is None:
            attn_output = _v13_p1(state, "o", o)
        if attn_output is None:
            attn_output = _dg1(state, "o", o)
        if attn_output is None:
            attn_output = _v12_gemm(o, dense["o"], "o", tokens)
        ffn_input, stream, router_input = _v9_bnd("attn", data["stream"], attn_output,
                                                  None if split_router else torch.float32, state, tokens)
"""

PY_ANCHOR = "def _gd_k3trig(state, tokens):"
PY_NEW = '''BND_ENABLE = True
BND_UNP = @UNP@
BND_O = @O@
BND_PH = {("unpooler", 1): 1, ("unpooler", 8): 1, ("unpooler", 16): 2, ("unpooler", 32): 2, ("o", 1): 1, ("o", 8): 2, ("o", 16): 2}
BND_SELF_TEST_T = {1: (1,), 8: (8, 2), 16: (16, 9), 32: (32, 17)}
_BND = {"st": {}, "seen": set()}


def _bnd_key(proj, b):
    rt = GD_ROUTE.get((proj, b))
    ph = BND_PH.get((proj, b))
    if not rt or not ph or rt[5] != 1:
        return None
    n, k = _gd_nk(proj, b)
    key = _gd_key(n, k, rt)
    nbx = n // 16 // rt[1]
    sl = 1 if n < HIDDEN else 0
    if key is None or n % 256 or (sl and (HIDDEN - n) // 256 > nbx) or nbx < 8 * rt[0]:
        return None
    return key + (0, ph | (sl << 2))


def _bnd_keys():
    if not BND_ENABLE:
        return []
    return [k for k in (_bnd_key(p, b) for (p, b) in sorted(BND_PH, key=str)) if k is not None]


def _bnd_weights(op, dev):
    op._ensure_cached_weights(torch.bfloat16)
    e = torch.empty(0, device=dev)
    return (op._cc_post_weight.to(torch.bfloat16).contiguous() if op._post_norm.has_weight else e,
            op._cc_pre_weight.to(torch.bfloat16).contiguous() if op._pre_norm.has_weight else e,
            op._cc_alpha.float().contiguous() if op._use_residual_gate else e,
            op._cc_beta.float().contiguous() if op._use_residual_gate else e,
            1 if op._post_norm_is_input_scale else 0, float(op._post_norm.eps), float(op._pre_norm.eps))


def _bnd_run(e, x, stream, out, ns, rt, branch, ldo, nob, tokens):
    r = e["r"]
    pw, qw, al, bt, isc, pe, qe = e["w"]
    pdl = (1 if GD_PDL.get(e["proj"]) else 0) | (2 if GD_TRIG.get(e["proj"]) else 0)
    r["mod"].gdb_run(e["inst"], r["w"], r["esc"], x, branch, tokens, r["K"], ldo, r["tb"], r["nbx"], pdl, stream, out, ns, rt,
                     pw, qw, al, bt, pe, qe, isc, e["part"], e["rs"], e["ctr"], e["err"], nob)


def _bnd_entry(state, proj, x, stream, tokens, mode):
    b = _gd_bucket(tokens)
    e = _BND["st"].get((proj, b)) if (BND_ENABLE and mode) else None
    if e is None or _gd_live(state, proj, tokens) is not e["r"]:
        return None
    if not (x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == e["r"]["K"] and x.is_contiguous() and x.data_ptr() % 16 == 0
            and stream.dtype == torch.float32 and stream.is_contiguous() and tuple(stream.shape) == (tokens, HIDDEN)):
        return None
    return e


def _bnd_call(state, e, x, stream, rdt, tokens, mode):
    dev = x.device
    if (e["proj"], tokens, mode) not in _BND["seen"]:
        _BND["seen"].add((e["proj"], tokens, mode))
        print(f"[bnd] first call {e['proj']} T={tokens} mode {mode} ({e['desc']})", flush=True)
    if mode in (1, 3):   # 3: rider stopped after the partial words (cost split)
        ob, nso, rto = (v[:tokens] for v in e["scr"])
        if e["r"]["slice"]:
            zb = state.get("v12_zbuf")
            if not (V12_ZBUF and zb is not None and R1_UNPOOL_INTO_ZEROS and x.dtype == zb.dtype):
                return None
            branch, ldo = zb[:tokens], HIDDEN
        else:
            branch = torch.empty((tokens, e["r"]["N"]), dtype=torch.bfloat16, device=dev)
            ldo = e["r"]["N"]
        _bnd_run(e, x, stream, ob, nso, rto if (e["proj"] == "o" and tokens <= ROUTER_FP32_MAX_TOKENS) else e["e"], branch, ldo,
                 0 if mode == 1 else 2, tokens)
        return ("rider", branch)
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=dev)
    ns = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev)
    rt = torch.empty((tokens, HIDDEN), dtype=torch.float32, device=dev) if rdt is not None else e["e"]
    _bnd_run(e, x, stream, out, ns, rt, e["e"], HIDDEN, 1, tokens)
    return ("fused", out, ns, rt if rdt is not None else None)


def _bnd_unpool(state, x, stream, tokens):
    b = _gd_bucket(tokens)
    mode = BND_UNP.get(b, 0) if b is not None else 0
    e = _bnd_entry(state, "unpooler", x, stream, tokens, mode)
    if e is None:
        return None
    try:
        return _bnd_call(state, e, x, stream, None, tokens, mode)
    except Exception as ex:
        _BND["st"] = {}
        print(f"[bnd] unpooler raised at call time -> today's boundaries: {ex!r}", flush=True)
        return None


def _bnd_o(state, x, stream, rdt, tokens):
    b = _gd_bucket(tokens)
    mode = BND_O.get(b, 0) if b is not None else 0
    e = _bnd_entry(state, "o", x, stream, tokens, mode)
    if e is None:
        return None
    try:
        r = _bnd_call(state, e, x, stream, rdt, tokens, mode)
    except Exception as ex:
        _BND["st"] = {}
        print(f"[bnd] o raised at call time -> today's boundaries: {ex!r}", flush=True)
        return None
    if r is None or r[0] == "rider":
        return r
    return ("fused", r[1:])


def _bnd_selftest(state, e, which, b, dev):
    msgs, ok, bw = [], True, True
    for T in BND_SELF_TEST_T[b]:
        for rdt in ((None, torch.float32) if (which == "attn" and T <= ROUTER_FP32_MAX_TOKENS) else (None,)):
            x, _ = _gd_test_input(T, e["r"]["K"], 6100 + 37 * T + (rdt is not None), dev)
            x = x.contiguous()
            gen = torch.Generator(device=dev).manual_seed(6200 + T)
            stream = torch.randn((T, HIDDEN), generator=gen, device=dev)
            if e["r"]["slice"]:
                branch = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev)
                br = _gd(state, e["proj"], x, branch, HIDDEN)
            else:
                br = _gd(state, e["proj"], x)
            if br is None:
                return False, "GD route call failed", False
            ref = _v9_bnd(which, stream, br, rdt, state, T)
            got = [_bnd_call(state, e, x, stream, rdt, T, 2) for _ in range(2)]
            rid = _bnd_call(state, e, x, stream, rdt, T, 1)
            torch.cuda.synchronize()
            g = got[0]
            e0 = float((g[1].float() - ref[0].float()).norm() / ref[0].float().norm())
            e1 = float((g[2] - ref[1]).norm() / ref[1].norm())
            e2 = float((g[3] - ref[2]).norm() / ref[2].norm()) if rdt is not None else 0.0
            same = bool(torch.equal(g[1], ref[0])) and bool(torch.equal(g[2], ref[1])) and (rdt is None or bool(torch.equal(g[3], ref[2])))
            det = bool(torch.equal(got[1][1], g[1])) and bool(torch.equal(got[1][2], g[2]))
            sc = [v[:T] for v in e["scr"]]
            rid_ok = (rid is not None and bool(torch.equal(rid[1], br)) and bool(torch.equal(sc[0], g[1])) and bool(torch.equal(sc[1], g[2])))
            err = int(e["err"].item())
            good = e0 < 1e-2 and e1 < 1e-4 and e2 < 1e-2 and det and rid_ok and err == 0 and bool(torch.isfinite(g[2]).all())
            ok &= good
            bw &= same
            msgs.append(f"T{T}{'r' if rdt is not None else ''} {e0:.1e}/{e1:.1e}/{e2:.1e} bw {int(same)} det {int(det)} rider {int(rid_ok)} err {err}")
    return ok, "; ".join(msgs), bw


def _bnd_prepare(state, t0):
    _BND["st"] = {}
    gd = state.get("gd")
    if not BND_ENABLE or gd is None:
        return
    t1 = time.time()
    mod = gd["mod"]
    ids = {k: i for i, k in enumerate(_GD["keys"])}
    dev = state["router"].device
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    try:
        wts = {"unpooler": _bnd_weights(state["ffn_boundary"], dev), "o": _bnd_weights(state["attn_boundary"], dev)}
    except Exception as ex:
        print(f"[bnd] weights FAILED -> off: {ex!r}", flush=True)
        return
    live = []
    for (proj, b) in sorted(BND_PH, key=str):
        r = gd["route"].get((proj, b))
        key = _bnd_key(proj, b)
        if r is None or key is None or key not in ids:
            print(f"[bnd] {proj}/{b}: no GD route / key -> off", flush=True)
            continue
        info = mod.gd_info(ids[key])
        base = mod.gd_info(r["inst"])
        nbx = r["nbx"]
        desc = f"{r['desc']} bnd {key[-1]} regs {info[0]} (base {base[0]}) spill {info[1]} (base {base[1]}) occ {info[2]} smem {info[3]}"
        if info[1] > base[1] or info[2] * sms < nbx:
            print(f"[bnd] {proj}/{b}: {desc}: spills or not one wave ({nbx} CTAs) -> off", flush=True)
            continue
        T = r["T"]
        e = {"proj": proj, "inst": ids[key], "r": r, "w": wts[proj], "desc": desc, "e": torch.empty(0, device=dev),
             "part": torch.zeros((T * 1056 * 4,), dtype=torch.int64, device=dev),
             "rs": torch.zeros((T * 2,), dtype=torch.int64, device=dev),
             "ctr": torch.zeros((2,), dtype=torch.int32, device=dev), "err": torch.zeros((1,), dtype=torch.int32, device=dev),
             "scr": (torch.empty((T, HIDDEN), dtype=torch.bfloat16, device=dev), torch.empty((T, HIDDEN), dtype=torch.float32, device=dev),
                     torch.empty((T, HIDDEN), dtype=torch.float32, device=dev))}
        try:
            ok, msg, bw = _bnd_selftest(state, e, "ffn" if proj == "unpooler" else "attn", b, dev)
        except Exception as ex:
            ok, msg, bw = False, f"EXC {ex!r}", False
        print(f"[bnd] self-test {proj}/{b} {desc}: {msg} -> {'ok' if ok else 'FAIL: off'}{' (bitwise)' if ok and bw else ''}", flush=True)
        if ok:
            _BND["st"][(proj, b)] = e
            live.append(f"{proj}/{b}")
    print(f"[bnd] live {live} modes unp {BND_UNP} o {BND_O} ({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)


''' + PY_ANCHOR


def bnd(t, unp, o):
    a0 = t.index("GD_CUDA_SRC = r'''")
    a1 = t.index(NL + "'''" + NL, a0)
    c = t[a0:a1]
    c = sub(c, DEV, DEV_NEW, what="device helpers")
    c = sub(c, ARGS_OLD, ARGS_NEW, what="Args")
    c = sub(c, TPL_OLD, TPL_NEW, what="kernel template")
    c = sub(c, SMEM_OLD, SMEM_NEW, what="smem")
    c = sub(c, EPI_OLD, EPI_NEW, what="epilogue")
    c = sub(c, PRE_OLD, PRE_NEW, what="prefetch")
    c = sub(c, "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0>\n",
            "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int CAT = 0, int BND = 0>\n", 2,
            what="launch / info templates")
    c = sub(c, "gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT>", "gd_kernel<OUT, MT, WR, WK, KU, ST, S, MB, XP, 0, CAT, BND>", 2,
            what="launch / info instances")
    c = sub(c, HOST_ANCHOR, HOST_NEW, what="host gdb_run_raw")
    t = t[:a0] + c + t[a1:]
    p0 = t.index("GD_CPP_SRC = r'''")
    p1 = t.index(NL + "'''" + NL, p0)
    t = t[:p0] + sub(t[p0:p1], CPP_ANCHOR, CPP_NEW, what="cpp gdb_run") + t[p1:]
    t = sub(t, '"gdr_info", "gdc_run", "gd_unpack_cat"]', '"gdr_info", "gdb_run", "gdc_run", "gd_unpack_cat"]', what="functions")
    t = sub(t, KEYS_OLD, KEYS_NEW, what="_gd_keys")
    t = sub(t, "    _gd_cat_prepare(state, t0)\n", "    _gd_cat_prepare(state, t0)\n    _bnd_prepare(state, t0)\n", what="prepare")
    t = sub(t, UNP_OLD, UNP_NEW, what="forward unpooler")
    t = sub(t, O_OLD, O_NEW, what="forward o")
    t = sub(t, PY_ANCHOR, PY_NEW.replace("@UNP@", repr(unp)).replace("@O@", repr(o)), what="python block")
    return t


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    base = ROOT / ARGS.get("base", "kernels/muse_decode_layer/k3_gc_16.py")
    marks = "--marks" in sys.argv
    out = ROOT / f"kernels/lab/k3_{'bndlat' if marks else 'bnd'}_{n}.py"
    check(not out.exists(), f"{out.name} exists (never overwrite)")
    unp = dict(zip((1, 8, 16, 32), (int(v) for v in ARGS.get("unp", "2,2,2,2").split(","))))
    o = dict(zip((1, 8, 16), (int(v) for v in ARGS.get("o", "2,2,2").split(","))))
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    t = bnd(b, unp, o)
    what = f"BND boundary folds into GD unpooler / o epilogues, modes unp {unp} o {o}"
    if marks:
        _argv, sys.argv = sys.argv, [sys.argv[0]]
        import gdh_patches as LP  # noqa: E402
        sys.argv = _argv
        t = LP.sbb_marks(LP.gd_marks(t))   # gdh: lat_patches sbb marks with the trig_early anchors
        t = sub(t, "A.marks = nullptr; A.bst = bst;", "A.marks = gd_mk_take(1, inst, M, K, ldo, nbx, 1); A.bst = bst;", what="gdb marks")
        t = sub(t, "  KS1P(0);\n", "  KS1P(0);\n  if (p.prof != nullptr && threadIdx.x == 0) { unsigned sm_; asm volatile(\"mov.u32 %0, %%smid;\" : \"=r\"(sm_)); "
                "p.prof[cta * 16 + 15] = (long long)sm_; }\n", what="ks1 smid")
        what += " + GD / sbb1 marks (lat_patches, inert unless armed)"
    nonce = f"# nonce k3-{'bndlat' if marks else 'bnd'}-{n} ({base.stem} + {what}; tools/gen_k3_bnd.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    names = [getattr(x, "name", None) or (x.targets[0].id if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) else None)
             for x in ast.parse(t).body]
    dup = sorted({x for x in names if x and names.count(x) > 1})
    check(not dup, f"duplicate top-level names {dup}")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_bnd] OK {len(t.encode())} B (base {len(b.encode())} B)")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[gen_k3_bnd] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

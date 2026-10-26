"""ks4_patches.py: KS4 = one-launch FULL-layer decode attention at B = 1 (decode 1 full: q_len 1, 80000 keys ~ 20 MB FP8 K + V), replacing
FA4 v8's prologue -> core -> combine there. Exact text patches against kernels/muse_decode_layer/k3_v34.py (KS1 + KS2 + KS3 + DG1);
the kernel lives in KS1's extension (one build; DG1's fold is untouched).

  grid = NS CTAs (default 264 = 2 per SM), 256 threads = 2 KV heads x 4 key parts; split s owns old keys [s S, s S + S), S =
  ceil(n_old / NS) <= 128 NT. Before griddepcontrol.wait every tile of 128 keys (both heads, 128-B rows) is issued by cp.async
  into FP8 smem as its own commit group (inputs only: cache_seqlens, page table, cache rows); after the wait the warps start on
  tile 0 as soon as its group lands (wait_group NT - 1 - i), overlapping the remaining tiles' DRAM traffic. Fragments, online
  softmax and the new-key fold are KS3's (chunk32). Two-level merge: tickets per group of GS splits -> the group's last CTA merges
  its GS partials (in order) into a level-2 partial -> ticket on the level-2 counter -> the last of the NG group mergers merges the
  NG level-2 partials (in order), rounds to bf16, RMSNorm x sigmoid(gate). No spin waits; counters self-reset.

Layer glue: KS4_ENABLE / KS4_ROUTE / KS4_GEOM / KS4_TRIG (read at call time); routed only when the page table's width fits
NS x 128 NT keys (shape-only). prepare: _ks4_self_test on B = 1 full cases (80000 / 100000 / 3000 + row 0 / 40 keys): rel L2 vs
the FP32 reference <= 1.25 x the r1 path's + 5e-3, cache bytes == r1's, tickets back at zero, repeat bit-identical, graph replay.

Measured (box, layer_ab2 4 seeds x 4 pairs, check PASS, self-tests PASS): decode 1 full FA4 v8 95.7 us -> KS4 98.1 (x0.989; GS 33
98.7, GS 8 98.2, NS 256 98.3, trig 0 98.2) -> KS4_ENABLE = False. Anatomy (in-layer, from the first wait-return): CTAs enter
-3.1..-0.5 us (as the qkv GEMV's CTAs retire), late CTAs issue their tiles up to +4.7 us (page-table loads queue behind the
early CTAs' streams), all tiles done median 6.2 / max 9.6 us (~2.1 TB/s effective on 20.5 MB), then the merge chain: partial
10.2 -> ticket 10.9 -> level-1 merge 12.2 -> ticket 12.7 -> merged 15.1 (level 2: 17 partials in 3 serialized batches of L2
loads). To win: single-round-trip merges (~-2 us) AND a stream that starts under the qkv GEMV (blocked by P1's smem carveout).

    python -I tools/ks4_patches.py [k3_v34.py] [out.py]
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v34.py"

CPP = ('KS1_CPP_SRC = r"""', '"""')
CU = ('KS1_CUDA_SRC = r"""', '"""')
BUILD = ("def _ks1_build():", "\ndef ")
TABLES = ("KS3_ENABLE = True ", "\n_KS1 = {")
WANT = ("def _ks1_wanted(", "\ndef _ks3_routed(")
RUN = ("def _ks1_run(", "\ndef ")
PREP = ("def _ks1_prepare(state, t0):", "\n\n\n")

KS4_CPP = r"""void ks4_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor page_table,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters,
                 torch::Tensor out, torch::Tensor prof, int64_t NT, int64_t NS, int64_t GS, double qk_sm, double eps, int64_t trig, int64_t pdl);
std::vector<int64_t> ks4_info(int64_t NT);
"""

KS4_CU = r"""

// ============================================================================================ KS4 (tools/ks4_patches.py)
namespace ks4 {
using namespace ks3;   // ks1 helpers + KS3's chunk32 / norm_rope / cp_async16 / swizzles

constexpr int L2C = 1000;   // level-2 ticket counter index

struct Params4 {
  const __nv_bfloat16* qkv;   // [1, stride] q | k | v | gate
  const int* cache_seqlens;   // [1]
  const int64_t* cache_loc;   // [1]
  const int* page_table;      // [1, pt_cols]
  uint8_t* k_cache;
  uint8_t* v_cache;
  float* part_o;              // level 1 [NS][16][64], then level 2 [NG][16][64] (unnormalized O)
  float2* part_ml;            // level 1 [NS][16], then level 2 [NG][16] (max, sum) in the exp2 domain
  int* counters;              // [0, NG): level-1 tickets, [L2C]: level-2 ticket
  __nv_bfloat16* out;         // [1, 1024]
  long long* prof;            // lab: [NS][16] %globaltimer stamps, or nullptr
  int stride_qkv, pt_cols, NS, GS, trig;
  float qk_sm, eps;
};

template <int NT> struct G4 {
  static constexpr int TILE = 32768;                    // 128 keys x (K 128 B + V 128 B)
  static constexpr int OFF_KN = NT * TILE;              // new-key tiles: 32 rows each (rows >= 1 zero)
  static constexpr int OFF_VN = OFF_KN + 32 * 128;
  static constexpr int OFF_Q = OFF_VN + 32 * 128;
  static constexpr int OFF_ML = OFF_Q + 16 * 128;
  static constexpr int OFF_MISC = OFF_ML + 2 * NWARP * 8 * 4;
  static constexpr int SMEM = OFF_MISC + 64;
};

__device__ __forceinline__ void cp_wait_n(int n) {   // n is a compile-time constant after unrolling
  if (n <= 0) asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  else if (n == 1) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
  else if (n == 2) asm volatile("cp.async.wait_group 2;\n" ::: "memory");
  else asm volatile("cp.async.wait_group 3;\n" ::: "memory");
}

// Online merge of partials [first, first + count) of row mr, dims 4 mu .. + 3 (in index order): unnormalized O, (max, sum).
__device__ __forceinline__ void merge_range(const float* po, const float2* pml, int first, int count, int mr, int mu, float& M,
                                            float& den, float4& a) {
  M = neg_inf();
  den = 0.f;
  a = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int q0 = 0; q0 < count; q0 += 8) {
    float2 ml[8];
    float4 ov[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int q = q0 + k;
      if (q < count) {
        const size_t prow = (size_t)(first + q) * 16 + mr;
        ml[k] = __ldcg(pml + prow);
        ov[k] = __ldcg(reinterpret_cast<const float4*>(po + prow * 64 + 4 * mu));
      } else {
        ml[k] = make_float2(neg_inf(), 0.f);
        ov[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
    float bm = M;
#pragma unroll
    for (int k = 0; k < 8; ++k) bm = fmaxf(bm, ml[k].x);
    const float Mu = (bm == neg_inf()) ? 0.f : bm;
    const float sc = ex2(M - Mu);                               // 0 while M is -inf
    den *= sc;
    a.x *= sc; a.y *= sc; a.z *= sc; a.w *= sc;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float w = ex2(ml[k].x - Mu);
      den = fmaf(w, ml[k].y, den);
      a.x = fmaf(w, ov[k].x, a.x); a.y = fmaf(w, ov[k].y, a.y); a.z = fmaf(w, ov[k].z, a.z); a.w = fmaf(w, ov[k].w, a.w);
    }
    M = bm;
  }
}

#define KS4P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[s * 16 + (k)] = gtime(); } while (0)

template <int NT>
__global__ void __launch_bounds__(NTHR, 2) ks4_kernel(const __grid_constant__ Params4 p) {
#if KS1_ARCH_OK
  using G = G4<NT>;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + NWARP * 8;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, NS = p.NS;
  const bool tail = (s == NS - 1);                              // also attends the new key
  const int hw = warp & 1, q4 = warp >> 1;
  if (p.trig == 2) pdl_trigger();
  KS4P(0);

  // ---------------- pre-wait: inputs only; all NT tiles of this split -> FP8 smem, one cp.async group per tile
  const int n_old = max(__ldg(p.cache_seqlens) - 1, 0);
  const int S = (n_old + NS - 1) / NS;
  const int base = s * S;
  const int cnt = max(0, min(S, n_old - base));
  const int nt = (cnt + 127) >> 7;
  {
    int rows[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int lk = 128 * i + ((tid + NTHR * j) >> 3);
        rows[i][j] = (lk < cnt && base + lk < p.pt_cols) ? __ldg(p.page_table + base + lk) : -1;
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      uint8_t* Kt = smem + i * G::TILE;
      uint8_t* Vt = Kt + 16384;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int r = tid + NTHR * j, key = r >> 3, ch = r & 7;
        const bool ok = rows[i][j] >= 0;
        const size_t go = (size_t)(ok ? rows[i][j] : 0) * 128 + 16 * ch;
        cp_async16(Kt + key * 128 + 16 * kswz(key, ch), p.k_cache + go, ok ? 16 : 0);
        cp_async16(Vt + key * 128 + 16 * vswz(key, ch), p.v_cache + go, ok ? 16 : 0);
      }
      asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
  }
  long long loc = 0;
  if (tail) {
    loc = __ldg(p.cache_loc);
    reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);
    reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
  }
  KS4P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS4P(2);

  // ---------------- post-wait: q (16 heads, QK-norm, no RoPE), new k / v (tail), gate rows of the merge items
  const __nv_bfloat16* xr = p.qkv;
  const uint32_t qr0 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)warp * 64) + lane);
  const uint32_t qr1 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 8) * 64) + lane);
  uint32_t nr = 0u;
  if (tail && warp < 4) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);
  const int mr = tid >> 4, mu = tid & 15;
  const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
  const float2 cs0 = make_float2(1.f, 0.f);
  {
    const float2 a = norm_rope(qr0, cs0, 0, p.eps), c = norm_rope(qr1, cs0, 0, p.eps);
    reinterpret_cast<uint32_t*>(qs + warp * 128)[lane] = pack_h2(a.x, a.y);
    reinterpret_cast<uint32_t*>(qs + (warp + 8) * 128)[lane] = pack_h2(c.x, c.y);
  }
  if (tail && warp < 4) {
    const int kind = warp >> 1, head = warp & 1;
    float2 v = kind ? bf2f(nr) : norm_rope(nr, cs0, 0, p.eps);
    v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
    v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
    const uint32_t q8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = (uint16_t)q8;
    const int bb = 64 * head + 2 * lane;
    *reinterpret_cast<uint16_t*>((kind ? vn : kn) + 16 * (bb >> 4) + (bb & 15)) = (uint16_t)q8;
  }
  __syncthreads();
  KS4P(3);
  uint32_t qf[4][2];
  {
    const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + (hw * 8 + g) * 128);
#pragma unroll
    for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
  }
  const float cl2 = p.qk_sm * LOG2E;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = neg_inf(), l = 0.f;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    if (i < nt) {                                                // CTA-uniform
      cp_wait_n(NT - 1 - i);                                     // this thread's copies of tile i landed
      __syncthreads();                                           // ... and every thread's
      const uint8_t* Kt = smem + i * G::TILE;
      chunk32(Kt, Kt + 16384, 32 * q4, hw, qf, o, m, l, cl2, [&](int k) { return 128 * i + k < cnt; });
      if (i == 0) KS4P(4);
    }
  }
  if (tail && q4 == 3) chunk32(kn, vn, 0, hw, qf, o, m, l, cl2, [&](int k) { return k == 0; });
  l += __shfl_xor_sync(FULL, l, 1);
  l += __shfl_xor_sync(FULL, l, 2);
  KS4P(5);

  // ---------------- in-CTA merge (scratch over the tiles once every copy has landed), then the level-1 partial
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);
  if (t == 0) { mw[warp * 8 + g] = m; lw[warp * 8 + g] = l; }
#pragma unroll
  for (int jn = 0; jn < 8; ++jn) {
#pragma unroll
    for (int e = 0; e < 2; ++e) ow[(warp * 8 + g) * 64 + ((8 * (2 * t + e) + jn) ^ (4 * g))] = o[jn][e];
  }
  __syncthreads();
  {
    const int rh = mr >> 3, rg = mr & 7;
    float M = neg_inf();
#pragma unroll
    for (int k = 0; k < 4; ++k) M = fmaxf(M, mw[(2 * k + rh) * 8 + rg]);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float lt = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int w = 2 * k + rh;
      const float f = ex2(mw[w * 8 + rg] - Mu);
      lt = fmaf(f, lw[w * 8 + rg], lt);
      const float4 xv = *reinterpret_cast<const float4*>(ow + (w * 8 + rg) * 64 + ((4 * mu) ^ (4 * rg)));
      acc.x = fmaf(f, xv.x, acc.x); acc.y = fmaf(f, xv.y, acc.y); acc.z = fmaf(f, xv.z, acc.z); acc.w = fmaf(f, xv.w, acc.w);
    }
    const size_t prow = (size_t)s * 16 + mr;
    __stcg(reinterpret_cast<float4*>(p.part_o + prow * 64 + 4 * mu), acc);
    if (mu == 0) __stcg(p.part_ml + prow, make_float2(M, lt));
  }
  KS4P(6);
  const int GS = p.GS, gi = s / GS, g0 = gi * GS, gsz = min(GS, NS - g0), NG = (NS + GS - 1) / GS;
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + gi, 1) == gsz - 1) ? 1 : 0;
  __syncthreads();
  KS4P(7);
  if (misc[0] == 0) return;

  // ---------------- level 1: the group's last CTA merges its gsz partials -> level-2 partial gi
  float* po2 = p.part_o + (size_t)NS * 16 * 64;
  float2* pml2 = p.part_ml + (size_t)NS * 16;
  {
    float M, den;
    float4 a;
    merge_range(p.part_o, p.part_ml, g0, gsz, mr, mu, M, den, a);
    const size_t prow = (size_t)gi * 16 + mr;
    __stcg(reinterpret_cast<float4*>(po2 + prow * 64 + 4 * mu), a);
    if (mu == 0) __stcg(pml2 + prow, make_float2(M, den));
  }
  KS4P(8);
  __syncthreads();
  if (tid == 0) {
    p.counters[gi] = 0;                                          // self-reset (the whole group has arrived)
    misc[1] = (atom_add_acqrel(p.counters + L2C, 1) == NG - 1) ? 1 : 0;
  }
  __syncthreads();
  KS4P(9);
  if (misc[1] == 0) return;

  // ---------------- level 2: the last group merger merges the NG level-2 partials, bf16, RMSNorm x sigmoid(gate)
  {
    float M, den;
    float4 a;
    merge_range(po2, pml2, 0, NG, mr, mu, M, den, a);
    const float dd = (den > 0.f) ? den : 1.f;
    const float y0 = bfr(a.x / dd), y1 = bfr(a.y / dd), y2 = bfr(a.z / dd), y3 = bfr(a.w / dd);
    float ss = y0 * y0 + y1 * y1 + y2 * y2 + y3 * y3;
    ss += __shfl_xor_sync(FULL, ss, 8);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const float2 g01 = bf2f(graw.x), g23 = bf2f(graw.y);
    const __nv_bfloat162 o01 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-g01.x))), y1 * rs * (1.f / (1.f + __expf(-g01.y))));
    const __nv_bfloat162 o23 = __floats2bfloat162_rn(y2 * rs * (1.f / (1.f + __expf(-g23.x))), y3 * rs * (1.f / (1.f + __expf(-g23.y))));
    *reinterpret_cast<uint2*>(p.out + (size_t)mr * 64 + 4 * mu) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
  }
  if (tid == 0) p.counters[L2C] = 0;
  KS4P(10);
#endif
}

template <int NT>
static void launch4(const Params4& P, bool pdl, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks4_kernel<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, G4<NT>::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)P.NS, 1, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G4<NT>::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks4_kernel<NT>, P));
}

template <int NT>
static std::vector<int64_t> info4() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks4_kernel<NT>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks4_kernel<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, G4<NT>::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks4_kernel<NT>, NTHR, G4<NT>::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G4<NT>::SMEM, (int64_t)occ};
}

}  // namespace ks4

void ks4_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor page_table,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters,
                 torch::Tensor out, torch::Tensor prof, int64_t NT, int64_t NS, int64_t GS, double qk_sm, double eps, int64_t trig, int64_t pdl) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == 1 &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks4: qkv [1, 2304]");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.numel() == 1, "ks4: B = 1");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == 1, "ks4: cache_loc");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == 1 && page_table.is_contiguous(),
              "ks4: page_table");
  TORCH_CHECK(NS >= 2 && GS >= 1 && (NS + GS - 1) / GS <= 512 && page_table.size(1) - 1 <= NS * 128 * NT, "ks4: geometry");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks4: caches");
  const int64_t NG = (NS + GS - 1) / GS;
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= (NS + NG) * 16 * 64, "ks4: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= (NS + NG) * 16 * 2, "ks4: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() > ks4::L2C && NG <= ks4::L2C, "ks4: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == ks1::HQ * 64, "ks4: out");
  const at::cuda::CUDAGuard guard(qkv.device());
  ks4::Params4 P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.part_o = part_o.data_ptr<float>();
  P.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr<float>());
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 16 * NS), "ks4: prof");
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.NS = (int)NS;
  P.GS = (int)GS;
  P.trig = (int)trig;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (NT == 3) { ks4::launch4<3>(P, pdl != 0, st); return; }
  if (NT == 2) { ks4::launch4<2>(P, pdl != 0, st); return; }
  TORCH_CHECK(false, "ks4: NT must be 2 or 3");
}

std::vector<int64_t> ks4_info(int64_t NT) {
  if (NT == 3) return ks4::info4<3>();
  if (NT == 2) return ks4::info4<2>();
  return {};
}
"""

KS4_TABLES = '''KS4_ENABLE = False                   # KS4: one-launch full-layer decode attention at B = 1 -- LOSES in-layer (decode 1 full
#   95.7 -> 98.1 us, box_out/ks4/ab2.log; anatomy box_out/ks4/anat.log): off, prepare skips its self-test (read at call time)
KS4_ROUTE = {1: True}                # q_len -> KS4 on full layers at B = 1 (read at call time)
KS4_GEOM = {1: (3, 264, 16)}         # q_len -> (NT tiles of 128 keys, NS splits, GS splits per level-1 merge group)
KS4_TRIG = {1: 1}                    # q_len -> launch_dependents mode (as KS1_TRIG)
KS4_INSTS = ((3, 264, 16),)          # geometries self-tested in prepare (lab: (3, 264, 33), (3, 264, 8), (3, 256, 16): all lose)
KS4_SELF_TEST_CASES = ((80000, 981, False), (100000, 982, False), (3000, 983, True), (40, 984, False))   # (ctx, seed, row0): B 1 full
'''

KS4_PY = '''def _ks4_wanted(qkv, data, cfg, state):
    """KS4 for this call? Full layer (window 0, no RoPE), B = 1, q_len in KS4_ROUTE, page 1, unit KV scale, tp8 heads, the page table
    no wider than the geometry's NS x 128 NT keys (+ new), dtypes / layouts as KS1; needs the prepare self-test. Shape-only."""
    L = cfg["q_len"]
    geo = KS4_GEOM.get(L)
    if not (KS4_ENABLE and KS4_ROUTE.get(L) and geo is not None and (L, *geo) in (state.get("ks4") or set())
            and cfg["batch"] == 1 and cfg["window"] == 0 and not cfg["rope_theta"] and cfg["page_size"] == 1
            and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2):
        return False
    pt = data["page_table"]
    return (pt.dtype == torch.int32 and pt.dim() == 2 and pt.is_contiguous() and pt.shape[0] == 1
            and pt.shape[1] - L <= geo[1] * 128 * geo[0]
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == 1
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].numel() == 1
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].numel() == 1)


def _ks4_call(qkv, data, geom=None, trig=None, pdl=None):
    """One KS4 launch on qkv [1, 2304] (full layer, q_len 1): writes the new K / V cache row, returns [1, 1024] bf16."""
    cfg = data["config"]
    L = cfg["q_len"]
    nt, ns, gs = KS4_GEOM[L] if geom is None else geom
    trig = KS4_TRIG.get(L, 0) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    dev = qkv.device
    ng = (ns + gs - 1) // gs
    po = torch.empty(((ns + ng) * 16 * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty(((ns + ng) * 16 * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((1, 16 * 64), dtype=torch.bfloat16, device=dev)
    _KS1["mod"].ks4_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                            data["v_cache"].view(torch.uint8), po, pml, _ks1_counters(dev), out,
                            KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev),
                            int(nt), int(ns), int(gs),
                            float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS, int(trig), 1 if pdl else 0)
    return out


def _ks4_self_test(device, verbose=False):
    """KS4 per KS4_SELF_TEST_CASES case and KS4_INSTS geometry: KS1's checks (rel L2 vs the FP32 reference <= 1.25 x the r1 path's +
    5e-3, finite, new cache bytes == r1's, tickets back at zero, repeat with the other trigger bit-identical); then a graph replay
    of KS4_GEOM[1] at ctx 80000 identical to the eager call. Returns the passing (q_len, NT, NS, GS)."""
    cnt = _ks1_counters(device)
    bad, seen = set(), set()
    for ctx, seed, row0 in KS4_SELF_TEST_CASES:
        ref = _k1fa4_reference(_k1fa4_case(1, ctx, 1, 0, seed, row0, device)).view(1, -1)
        d1 = _k1fa4_case(1, ctx, 1, 0, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(1, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for geo in (tuple(x) for x in KS4_INSTS):
            if geo in bad or d1["page_table"].shape[1] - 1 > geo[1] * 128 * geo[0]:
                continue
            dk = _k1fa4_case(1, ctx, 1, 0, seed, row0, device)
            qkv = dk["qkv"].view(1, -1)
            o_a = _ks4_call(qkv, dk, geo, 0).clone()
            o_b = _ks4_call(qkv, dk, geo, 1)
            torch.cuda.synchronize(device)
            ek = float((o_a.float() - ref).norm() / ref.norm())
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and tick == 0 and same
            seen.add(geo)
            if not ok:
                bad.add(geo)
            if verbose or not ok:
                print(f"[ks4] self-test ctx{ctx}{' row0' if row0 else ''} geo {geo}: {'PASS' if ok else 'FAIL'} err {ek:.4e} (r1 {e1:.4e}) "
                      f"cache bytes != r1 {nbad} tickets {tick} repeat identical {same}", flush=True)
            del dk, o_a, o_b
    ok_geo = {(1, *g) for g in seen if g not in bad}
    geo = KS4_GEOM.get(1)
    if geo is not None and (1, *geo) in ok_geo:
        d = _k1fa4_case(1, 80000, 1, 0, 990, False, device)
        qkv = d["qkv"].view(1, -1)
        o1 = _ks4_call(qkv, d, geo).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks4_call(qkv, d, geo)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks4] graph check ctx80000 {geo}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_geo.discard((1, *geo))
    print(f"[ks4] geometries ok: {sorted(ok_geo)} (failed {sorted(bad)})", flush=True)
    return ok_geo


def _ks1_run('''

PATCHES = [
    ("CPP: ks4 declarations", CPP, "std::vector<int64_t> ks3_info(int64_t NS);\n", "std::vector<int64_t> ks3_info(int64_t NS);\n" + KS4_CPP),
    ("CU: ks4 kernel + host", CU, "  if (NS == 16) return ks3::info3<16>();\n  return {};\n}\n",
     "  if (NS == 16) return ks3::info3<16>();\n  return {};\n}\n" + KS4_CU),
    ("PY: build exports ks4", BUILD, '["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"]',
     '["ks1_forward", "ks1_info", "ks3_forward", "ks3_info", "ks4_forward", "ks4_info"]'),
    ("PY: KS4 tables", TABLES, "KS3_SELF_TEST_CASES = (", KS4_TABLES + "KS3_SELF_TEST_CASES = ("),
    ("PY: wanted -> KS4 on full layers", WANT, '    B, L = cfg["batch"], cfg["q_len"]\n    if B == 1:\n',
     '    B, L = cfg["batch"], cfg["q_len"]\n    if cfg["window"] == 0:   # KS4: full layers (dispatched in _ks1_run)\n'
     '        return _ks4_wanted(qkv, data, cfg, state)\n    if B == 1:\n'),
    ("PY: KS4 functions", RUN, "def _ks1_run(", KS4_PY),
    ("PY: run dispatches KS4", RUN, '    cfg = data["config"]\n    if cfg["batch"] > 1 and _ks3_routed(',
     '    cfg = data["config"]\n    if cfg["window"] == 0:\n        return _ks4_call(qkv, data)\n    if cfg["batch"] > 1 and _ks3_routed('),
    ("PREP: state ks4", PREP, '    state["ks3"] = set()\n', '    state["ks3"] = set()\n    state["ks4"] = set()\n'),
    ("PREP: KS4 self-test", PREP, '            state["ks3"] = _ks3_self_test(state["cos_sin"], device)\n',
     '            state["ks3"] = _ks3_self_test(state["cos_sin"], device)\n'
     '        if KS4_ENABLE:\n'
     '            for nt in sorted({g[0] for g in KS4_INSTS}):\n'
     '                regs, local, smem_b, occ = (int(x) for x in mod.ks4_info(nt))\n'
     '                print(f"[ks4] NT{nt}: {regs} regs, {local} B local, {smem_b} B smem, {occ} CTAs/SM", flush=True)\n'
     '            state["ks4"] = _ks4_self_test(device)\n'),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"ks4: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"ks4 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v34_ks4_check.py")
    text = src.read_text(encoding="utf-8")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

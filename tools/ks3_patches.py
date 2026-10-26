"""ks3_patches.py: KS3 = one-launch sliding attention at q_len 1 for B > 1 with BOTH KV heads per CTA (128-B K / V rows), K / V staged
as FP8 in shared memory by cp.async issued before griddepcontrol.wait (the whole split in flight at once), K1's register fragments
(lane t reads dims 16t..16t+15 of a key, e4m3 -> f16 at use, byte-permuted V fragments), the new key folded into the last split, and
KS1's ticket / last-CTA merge / RMSNorm x sigmoid(gate) epilogue. Exact text patches against kernels/muse_decode_layer/k3_v32.py
(= v31 + tools/ks2_patches.py); the kernel lives in KS1's extension (one build).

  grid = (NS splits, B requests), 256 threads = 2 KV heads x 4 key parts; split s holds window slots [s KMAX, (s + 1) KMAX), KMAX =
  2048 / NS; 32-key online-softmax chunks per warp; warps of part 3 of the last split also attend the new key (16-slot tile).
  Partials: [B][NS][16 q heads][64] fp32 + (max, sum); ticket per request (atom.acq_rel); merge in split order (deterministic).

Layer glue: KS3_ENABLE / KS3_ROUTE / KS3_NS / KS3_TRIG keyed by (batch, q_len), read at call time; a KS3 route takes precedence over
KS2 for that (batch, q_len). prepare: _ks3_self_test (B 1 / 3 / 16 / 32, requests shorter than the window, cache row 0): error <=
1.25 x r1's + 5e-3, new cache bytes == r1's, tickets back at zero, repeat with the other trigger bit-identical, graph replay identical.

Measured (box, layer_ab2 4 seeds x 4 pairs, check PASS): decode 32 sliding 189.4 -> 187.0 us (x1.018 vs FA4 v8; NS 4 187.0, NS 16 189.4),
decode 16 sliding 146.1 -> 142.8 us (x1.021 vs KS2; NS 16 141.9). Not done: full B1 (KS3's B32 rate, ~12.4 us for 16.8 MB, projects
~15-16 us for 20.5 MB + a second merge level vs FA4's 14.8), verify sliding B16/32 (128 rows per request -> 256 KB merges).

    python -I tools/ks3_patches.py [k3_v32.py] [out.py]
    python3 tools/ks2_patches.py --build <patched layer>      # box: compile-only check (KS1 extension incl. KS3)
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYER = ROOT / "kernels/muse_decode_layer/k3_v32.py"

CPP = ('KS1_CPP_SRC = r"""', '"""')
CU = ('KS1_CUDA_SRC = r"""', '"""')
BUILD = ("def _ks1_build():", "\ndef ")
TABLES = ("KS2_ENABLE = True ", "\n_KS1 = {")
PY = ("def _ks1_wanted(", "\ndef _ks1_case(")
PREP = ("def _ks1_prepare(state, t0):", "\n\n\n")

KS3_CPP = r"""void ks3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> ks3_info(int64_t NS);
"""

KS3_CU = r"""
// ============================================================================================ KS3 (tools/ks3_patches.py)
namespace ks3 {
using namespace ks1;

struct Params3 {
  const __nv_bfloat16* qkv;   // [B, stride] q | k | v | gate (q_len 1)
  const int* cache_seqlens;   // [B]
  const int64_t* cache_loc;   // [B]
  const int64_t* positions;   // [B]
  const int* page_table;      // [B, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of 2 x 64 B (both KV heads)
  uint8_t* v_cache;
  const float* cos_sin;
  float* part_o;              // [B][NS][16][64]
  float2* part_ml;            // [B][NS][16]
  int* counters;              // [B]
  __nv_bfloat16* out;         // [B, 1024]
  long long* prof;            // [B NS][8] or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

template <int NS> struct G3 {
  static constexpr int KMAX = WIN / NS;                 // old-key slots per split
  static constexpr int KPW = KMAX / 4;                  // slots per warp (4 key parts per KV head)
  static constexpr int NCH = KPW / 32;                  // 32-key chunks per warp
  static constexpr int OFF_V = KMAX * 128;
  static constexpr int OFF_KN = 2 * KMAX * 128;         // new-key tiles: 32 rows each (rows >= 1 zero)
  static constexpr int OFF_VN = OFF_KN + 32 * 128;
  static constexpr int OFF_Q = OFF_VN + 32 * 128;       // q fp16 [16 heads][64]
  static constexpr int OFF_ML = OFF_Q + 16 * 128;       // per warp (max, sum) of its 8 rows
  static constexpr int OFF_MISC = OFF_ML + 2 * NWARP * 8 * 4;
  static constexpr int SMEM = OFF_MISC + 64;
  static_assert(KPW % 32 == 0 && 2 * KMAX * 128 >= NWARP * 8 * 64 * 4 && (KMAX * 8) % NTHR == 0, "KS3 geometry");
};

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gptr, int src_bytes) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gptr), "r"(src_bytes) : "memory");
}
// 16-B chunk ch (head ch >> 2) of key row x: K at ch ^ 4 (x & 1), V at ch ^ 2 ((x >> 1) & 3) (K1 v7's conflict-free maps)
__device__ __forceinline__ int kswz(int x, int ch) { return ch ^ (4 * (x & 1)); }
__device__ __forceinline__ int vswz(int x, int ch) { return ch ^ (2 * ((x >> 1) & 3)); }
__device__ __forceinline__ uint2 vld(const uint8_t* Vs, int x, int h, int g) {
  return *reinterpret_cast<const uint2*>(Vs + x * 128 + 16 * vswz(x, 4 * h + (g >> 1)) + 8 * (g & 1));
}
#define KS3P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

// SGLang FusedQKNorm arithmetic for one (even, odd) pair per lane: fp32 RMS over 64 dims, bf16, RoPE on the bf16 pair, bf16
__device__ __forceinline__ float2 norm_rope(uint32_t raw, float2 cs, int rope, float eps) {
  const float2 x = bf2f(raw);
  const float var = warp_sum(x.x * x.x + x.y * x.y) * (1.f / 64.f);
  const float rs = rsqrtf(var + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, cs.x, -__fmul_rn(o, cs.y))), bfr(__fmaf_rn(e, cs.y, __fmul_rn(o, cs.x))));
}

// 32 keys [x0, x0 + 32) of KV head h for the warp's 8 q rows (rows g; A rows g + 8 are zero): QK m16n8k16 FP16 with K1's dim
// permutation, online softmax in the exp2 domain, PV with byte-permuted V fragments (n-tile jn column n <-> output dim 8n + jn).
template <typename F>
__device__ __forceinline__ void chunk32(const uint8_t* Ks, const uint8_t* Vs, const int x0, const int h, const uint32_t (&qf)[4][2],
                                        float (&o)[8][4], float& m, float& l, const float cl2, F valid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float sc[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int key = x0 + 8 * j + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Ks + key * 128 + 16 * kswz(key, 4 * h + t));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t a[4] = {qf[s][0], 0u, qf[s][1], 0u};
      mma16816(c, a, e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
    }
    sc[j][0] = c[0];
    sc[j][1] = c[1];
  }
  float x = neg_inf();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      sc[j][e] = valid(x0 + 8 * j + 2 * t + e) ? sc[j][e] * cl2 : neg_inf();
      x = fmaxf(x, sc[j][e]);
    }
  }
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 1));
  x = fmaxf(x, __shfl_xor_sync(FULL, x, 2));
  const float nm = fmaxf(m, x);
  const float u = (nm == neg_inf()) ? 0.f : nm;
  const float al = ex2(m - u);
  m = nm;
  l *= al;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= al; o[n][1] *= al; }
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const float p00 = ex2(sc[2 * kk][0] - u), p01 = ex2(sc[2 * kk][1] - u);
    const float p10 = ex2(sc[2 * kk + 1][0] - u), p11 = ex2(sc[2 * kk + 1][1] - u);
    l += (p00 + p01) + (p10 + p11);
    const uint32_t pa[4] = {pack_h2(p00, p01), 0u, pack_h2(p10, p11), 0u};
    const int xa = x0 + 16 * kk + 2 * t;
    const uint2 va = vld(Vs, xa, h, g), vb = vld(Vs, xa + 1, h, g), vc = vld(Vs, xa + 8, h, g), vd = vld(Vs, xa + 9, h, g);
    const uint32_t w01[4] = {__byte_perm(va.x, vb.x, 0x5140), __byte_perm(va.x, vb.x, 0x7362),
                             __byte_perm(va.y, vb.y, 0x5140), __byte_perm(va.y, vb.y, 0x7362)};
    const uint32_t w23[4] = {__byte_perm(vc.x, vd.x, 0x5140), __byte_perm(vc.x, vd.x, 0x7362),
                             __byte_perm(vc.y, vd.y, 0x5140), __byte_perm(vc.y, vd.y, 0x7362)};
#pragma unroll
    for (int jn = 0; jn < 8; ++jn) {
      const uint32_t s01 = w01[jn >> 1], s23 = w23[jn >> 1];
      mma16816(o[jn], pa, e4m3x2_to_h2((jn & 1) ? (s01 >> 16) : s01), e4m3x2_to_h2((jn & 1) ? (s23 >> 16) : s23));
    }
  }
}

template <int NS>
__global__ void __launch_bounds__(NTHR, 2) ks3_kernel(const __grid_constant__ Params3 p) {
#if KS1_ARCH_OK
  using G = G3<NS>;
  constexpr int KMAX = G::KMAX, KPW = G::KPW;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + G::OFF_V;
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + NWARP * 8;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y;
  const int cta = b * NS + s;
  const bool tail = (s == NS - 1);                              // also attends the new key
  const int hw = warp & 1, q4 = warp >> 1;                      // KV head, key part
  if (p.trig == 2) pdl_trigger();
  KS3P(0);

  // ---------------- pre-wait: kernel inputs only; the split's K / V rows (both heads, 128 B) -> FP8 smem (cp.async, zero-filled)
  const int seq = max(__ldg(p.cache_seqlens + b) - 1, 0);
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  {
    constexpr int PER = KMAX * 8 / NTHR;                        // (key, 16-B chunk) pairs per thread
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
    int rows[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int key = (tid + NTHR * j) >> 3;
      const int gs = s * KMAX + key;
      rows[j] = (gs < n_old && kstart + gs < p.pt_cols) ? __ldg(pt + kstart + gs) : -1;
    }
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const int r = tid + NTHR * j, key = r >> 3, ch = r & 7;
      const bool ok = rows[j] >= 0;
      const size_t go = (size_t)(ok ? rows[j] : 0) * 128 + 16 * ch;
      cp_async16(Ks + key * 128 + 16 * kswz(key, ch), p.k_cache + go, ok ? 16 : 0);
      cp_async16(Vs + key * 128 + 16 * vswz(key, ch), p.v_cache + go, ok ? 16 : 0);
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  long long loc = 0;
  if (tail) {
    loc = __ldg(p.cache_loc + b);
    reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);          // kn | vn: 8 KB, zero (rows >= 1 stay zero)
    reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
  }
  KS3P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS3P(2);

  // ---------------- post-wait: q (16 heads), new k / v (tail), gate rows of the merge items
  const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
  const uint32_t qr0 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)warp * 64) + lane);
  const uint32_t qr1 = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 8) * 64) + lane);
  uint32_t nr = 0u;
  if (tail && warp < 4) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);   // k0 k1 v0 v1
  const int mr = tid >> 4, mu = tid & 15;                         // merge item: q head mr, dims 4 mu .. 4 mu + 3
  const uint2 graw = __ldcg(reinterpret_cast<const uint2*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 4 * mu));
  {
    const float2 a = norm_rope(qr0, cs, p.rope, p.eps), c = norm_rope(qr1, cs, p.rope, p.eps);
    reinterpret_cast<uint32_t*>(qs + warp * 128)[lane] = pack_h2(a.x, a.y);
    reinterpret_cast<uint32_t*>(qs + (warp + 8) * 128)[lane] = pack_h2(c.x, c.y);
  }
  if (tail && warp < 4) {                                        // warp 0 / 1: K head 0 / 1, warp 2 / 3: V head 0 / 1
    const int kind = warp >> 1, head = warp & 1;
    float2 v = kind ? bf2f(nr) : norm_rope(nr, cs, p.rope, p.eps);
    v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
    v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
    const uint32_t q8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = (uint16_t)q8;
    const int bb = 64 * head + 2 * lane;                          // key 0 of the new tile: chunk bb >> 4 (no swizzle at row 0)
    *reinterpret_cast<uint16_t*>((kind ? vn : kn) + 16 * (bb >> 4) + (bb & 15)) = (uint16_t)q8;
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  KS3P(3);

  // ---------------- attention: warp (head hw, part q4) = slots [q4 KPW, (q4 + 1) KPW) of its head, 32-key chunks
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
  const int base = s * KMAX;
#pragma unroll
  for (int c = 0; c < G::NCH; ++c)
    chunk32(Ks, Vs, q4 * KPW + 32 * c, hw, qf, o, m, l, cl2, [&](int k) { return base + k < n_old; });
  if (tail && q4 == 3) chunk32(kn, vn, 0, hw, qf, o, m, l, cl2, [&](int k) { return k == 0; });
  l += __shfl_xor_sync(FULL, l, 1);
  l += __shfl_xor_sync(FULL, l, 2);
  KS3P(4);

  // ---------------- in-CTA merge of the 4 warps per head (scratch over the K / V tiles), then this split's partial
  __syncthreads();
  float* ow = reinterpret_cast<float*>(smem);                   // [8 warps][8 rows][64], dims XOR 4 g
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
    const size_t prow = ((size_t)b * NS + s) * 16 + mr;
    __stcg(reinterpret_cast<float4*>(p.part_o + prow * 64 + 4 * mu), acc);
    if (mu == 0) __stcg(p.part_ml + prow, make_float2(M, lt));
  }
  KS3P(5);
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  KS3P(6);
  if (misc[0] == 0) return;

  // ---------------- last CTA of request b: merge the NS partials in split order, bf16, RMSNorm x sigmoid(gate)
  {
    float2 ml[NS];
    float4 ov[NS];
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      const size_t prow = ((size_t)b * NS + q) * 16 + mr;
      ml[q] = __ldcg(p.part_ml + prow);
      ov[q] = __ldcg(reinterpret_cast<const float4*>(p.part_o + prow * 64 + 4 * mu));
    }
    float M = neg_inf();
#pragma unroll
    for (int q = 0; q < NS; ++q) M = fmaxf(M, ml[q].x);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    float den = 0.f;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int q = 0; q < NS; ++q) {
      const float w = ex2(ml[q].x - Mu);
      den += w * ml[q].y;
      a.x = fmaf(w, ov[q].x, a.x); a.y = fmaf(w, ov[q].y, a.y); a.z = fmaf(w, ov[q].z, a.z); a.w = fmaf(w, ov[q].w, a.w);
    }
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
    *reinterpret_cast<uint2*>(p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 4 * mu) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&o01), *reinterpret_cast<const uint32_t*>(&o23));
  }
  if (tid == 0) p.counters[b] = 0;                               // self-reset (every split of request b has arrived)
  KS3P(7);
#endif
}

template <int NS>
static void launch3(const Params3& P, bool pdl, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G3<NS>::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G3<NS>::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks3_kernel<NS>, P));
}

template <int NS>
static std::vector<int64_t> info3() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks3_kernel<NS>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks3_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G3<NS>::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks3_kernel<NS>, NTHR, G3<NS>::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G3<NS>::SMEM, (int64_t)occ};
}

}  // namespace ks3
"""

KS3_HOST = r"""
void ks3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks3: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && B >= 1, "ks3: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B && cache_loc.is_contiguous(), "ks3: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B && positions.is_contiguous(), "ks3: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "ks3: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks3: caches");
  TORCH_CHECK(!rope || (cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.size(1) == 64), "ks3: cos_sin");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= B * NS * 16 * 64, "ks3: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= B * NS * 16 * 2, "ks3: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ks3: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * ks1::HQ * 64, "ks3: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ks3: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ks3::Params3 P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.part_o = part_o.data_ptr<float>();
  P.part_ml = reinterpret_cast<float2*>(part_ml.data_ptr<float>());
  P.counters = counters.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.stride_qkv = (int)qkv.stride(0);
  P.pt_cols = (int)page_table.size(1);
  P.B = (int)B;
  P.trig = (int)trig;
  P.rope = (int)rope;
  P.qk_sm = (float)qk_sm;
  P.eps = (float)eps;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (NS == 4) { ks3::launch3<4>(P, pdl != 0, st); return; }
  if (NS == 8) { ks3::launch3<8>(P, pdl != 0, st); return; }
  if (NS == 16) { ks3::launch3<16>(P, pdl != 0, st); return; }
  TORCH_CHECK(false, "ks3: NS must be 4, 8 or 16");
}

std::vector<int64_t> ks3_info(int64_t NS) {
  if (NS == 4) return ks3::info3<4>();
  if (NS == 8) return ks3::info3<8>();
  if (NS == 16) return ks3::info3<16>();
  return {};
}
"""

KS3_TABLES = '''
KS3_ENABLE = True                    # KS3: both-heads-per-CTA FP8-smem kernel on B > 1 sliding q_len 1 (read at call time)
KS3_ROUTE = {(32, 1): True, (16, 1): True}    # (batch, q_len) -> KS3 (takes precedence over KS2; box_out/ks3/ab1.log: dec32 x1.018
#   vs FA4, dec16 x1.021 vs KS2; read at call time)
KS3_NS = {(32, 1): 8, (16, 1): 8}    # (batch, q_len) -> splits (4 / 8 / 16)
KS3_TRIG = {(32, 1): 1, (16, 1): 1}  # (batch, q_len) -> launch_dependents mode (as KS1_TRIG)
KS3_INSTS = (8,)                     # NS values self-tested in prepare (lab: 4 = 1 CTA/SM + 8 B spill, 16 = x0.996 at dec32)
KS3_SELF_TEST_CASES = ((32, 2600, 961, True), (16, 3000, 962, False), (3, 2100, 963, False))   # (B, ctx, seed, row0)
'''

KS3_PY = '''def _ks3_routed(key, state):
    return bool(KS3_ENABLE and key[1] == 1 and KS3_ROUTE.get(key) and KS3_NS.get(key) in (state.get("ks3") or set()))


def _ks3_call(qkv, data, cos_sin, ns=None, trig=None, pdl=None):
    """One KS3 launch on qkv [B, 2304] (q_len 1, sliding): writes the new K / V cache rows, returns [B, 1024] bf16."""
    cfg = data["config"]
    B = cfg["batch"]
    ns = KS3_NS[(B, 1)] if ns is None else ns
    trig = KS3_TRIG.get((B, 1), 0) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    dev = qkv.device
    po = torch.empty((B * ns * 16 * 64,), dtype=torch.float32, device=dev)
    pml = torch.empty((B * ns * 16 * 2,), dtype=torch.float32, device=dev)
    out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
    prof = KS1_PROF_BUF if KS1_PROF_BUF is not None else torch.empty((0,), dtype=torch.int64, device=dev)
    _KS1["mod"].ks3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, po, pml,
                            _ks1_counters(dev), out, prof, int(ns), float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS,
                            int(trig), 1 if pdl else 0, 1 if cfg["rope_theta"] else 0)
    return out


def _ks3_self_test(cos_sin, device, verbose=False):
    """KS3 per KS3_SELF_TEST_CASES case and NS: KS1's checks (rel L2 vs the FP32 reference <= 1.25 x the r1 path's + 5e-3, finite,
    new cache bytes == r1's, tickets back at zero, repeat with the other trigger bit-identical); then a graph replay at B 32 with
    KS3_NS[(32, 1)] identical to the eager call. Returns the NS values that passed."""
    cnt = _ks1_counters(device)
    bad, seen = set(), set()
    for batch, ctx, seed, row0 in KS3_SELF_TEST_CASES:
        ref = _k1fa4_reference(_k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)).view(batch, -1)
        d1 = _k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)
        o1 = _k1fa4_r1_path(d1).float().view(batch, -1)
        torch.cuda.synchronize(device)
        e1 = float((o1 - ref).norm() / ref.norm())
        rows = d1["cache_loc"]
        k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
        v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
        for ns in KS3_INSTS:
            if ns in bad:
                continue
            dk = _k1fa4_case(batch, ctx, 1, 2048, seed, row0, device)
            qkv = dk["qkv"].view(batch, -1)
            o_a = _ks3_call(qkv, dk, cos_sin, ns, 0).clone()
            o_b = _ks3_call(qkv, dk, cos_sin, ns, 1)
            torch.cuda.synchronize(device)
            ek = float((o_a.float() - ref).norm() / ref.norm())
            kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
            vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
            nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
            tick = int(cnt.abs().sum())
            same = bool(torch.equal(o_a, o_b))
            ok = bool(torch.isfinite(o_a).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and tick == 0 and same
            seen.add(ns)
            if not ok:
                bad.add(ns)
            if verbose or not ok:
                print(f"[ks3] self-test B{batch} NS{ns} ctx{ctx}{' row0' if row0 else ''}: {'PASS' if ok else 'FAIL'} err {ek:.4e} "
                      f"(r1 {e1:.4e}) cache bytes != r1 {nbad} tickets {tick} repeat identical {same}", flush=True)
            del dk, o_a, o_b
    ok_ns = {n for n in seen if n not in bad}
    ns = KS3_NS.get((32, 1))
    if ns in ok_ns:
        d = _k1fa4_case(32, 3000, 1, 2048, 970, False, device)
        qkv = d["qkv"].view(32, -1)
        o1 = _ks3_call(qkv, d, cos_sin, ns).clone()
        torch.cuda.synchronize(device)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            og = _ks3_call(qkv, d, cos_sin, ns)
        gr.replay()
        gr.replay()
        torch.cuda.synchronize(device)
        same = bool(torch.equal(og, o1)) and int(cnt.abs().sum()) == 0
        print(f"[ks3] graph check B32 NS{ns}: replay identical {same}", flush=True)
        del gr, og
        if not same:
            ok_ns.discard(ns)
    print(f"[ks3] NS ok: {sorted(ok_ns)} (failed {sorted(bad)})", flush=True)
    return ok_ns


def _ks1_run(qkv, data, state):
    cfg = data["config"]
    if cfg["batch"] > 1 and _ks3_routed((cfg["batch"], cfg["q_len"]), state):
        return _ks3_call(qkv, data, state["cos_sin"])
    return _ks1_call(qkv, data, state["cos_sin"])
'''

PATCHES = [
    ("CPP: ks3 declarations", CPP, "std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG);\n",
     "std::vector<int64_t> ks1_info(int64_t L, int64_t NS, int64_t NTG);\n" + KS3_CPP),
    ("CU: ks3 kernel", CU, "}  // namespace ks1\n\n#define KS1_INSTS @INSTS@\n", "}  // namespace ks1\n" + KS3_CU + "\n#define KS1_INSTS @INSTS@\n"),
    ("CU: ks3 host", CU, "  KS1_INSTS\n#undef KS1_X\n  return {};\n}\n", "  KS1_INSTS\n#undef KS1_X\n  return {};\n}\n" + KS3_HOST),
    ("PY: build exports ks3", BUILD, 'functions=["ks1_forward", "ks1_info"],', 'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"],'),
    ("PY: KS3 tables", TABLES,
     "KS2_SELF_TEST_CASES = ((16, 3000, 1, 941, False), (32, 2600, 1, 942, True), (3, 2100, 1, 943, False))   # q_len 8 at B > 1: unrouted\n",
     "KS2_SELF_TEST_CASES = ((16, 3000, 1, 941, False), (32, 2600, 1, 942, True), (3, 2100, 1, 943, False))   # q_len 8 at B > 1: unrouted\n"
     + KS3_TABLES),
    ("PY: wanted takes KS3 routes", PY,
     '        geo, passed = (KS2_GEOM.get((B, L)) if (KS2_ENABLE and KS2_ROUTE.get((B, L))) else None), state.get("ks2") or set()\n',
     '        geo, passed = (KS2_GEOM.get((B, L)) if (KS2_ENABLE and KS2_ROUTE.get((B, L))) else None), state.get("ks2") or set()\n'
     '        if _ks3_routed((B, L), state):   # KS3 (dispatched in _ks1_run)\n'
     '            geo, passed = (0, 0), {(L, 0, 0)}\n'),
    ("PY: KS3 call / self-test / dispatch", PY, 'def _ks1_run(qkv, data, state):\n    return _ks1_call(qkv, data, state["cos_sin"])\n', KS3_PY),
    ("PREP: state ks3", PREP, '    state["ks2"] = set()\n', '    state["ks2"] = set()\n    state["ks3"] = set()\n'),
    ("PREP: KS3 self-test", PREP,
     '            state["ks2"] = _ks2_self_test(state["cos_sin"], device, [g for g in KS1_INSTS if g in ok_geo])\n',
     '            state["ks2"] = _ks2_self_test(state["cos_sin"], device, [g for g in KS1_INSTS if g in ok_geo])\n'
     '        if KS3_ENABLE:\n'
     '            for n in KS3_INSTS:\n'
     '                regs, local, smem_b, occ = (int(x) for x in mod.ks3_info(n))\n'
     '                print(f"[ks3] NS{n}: {regs} regs, {local} B local, {smem_b} B smem, {occ} CTAs/SM", flush=True)\n'
     '            state["ks3"] = _ks3_self_test(state["cos_sin"], device)\n'),
]


def _scope(text, scope):
    a = text.index(scope[0])
    if text.count(scope[0]) != 1:
        raise RuntimeError(f"ks3: scope start {scope[0]!r} occurs {text.count(scope[0])} times")
    b = text.find(scope[1], a + len(scope[0]))
    return a, (len(text) if b < 0 else b)


def apply(text, patches=PATCHES):
    for what, scope, old, new in patches:
        a, b = _scope(text, scope)
        n = text[a:b].count(old)
        if n != 1:
            raise RuntimeError(f"ks3 patch {what!r}: `old` occurs {n} times in its scope (want 1)")
        i = text.index(old, a)
        text = text[:i] + new + text[i + len(old):]
    return text


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("k3_v32_ks3_check.py")
    text = src.read_text(encoding="utf-8")
    patched = apply(text)
    out.write_bytes(patched.encode("utf-8"))   # LF as read
    py_compile.compile(str(out), doraise=True)
    print(f"all {len(PATCHES)} patches hit once in scope; patched copy {out} ({len(patched) - len(text):+d} chars) compiles")

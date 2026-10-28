"""vsq_patches.py: KS8 = one-launch q_len 8 multi-request SLIDING attention (verify16 / verify32), notes/study/vsq_attn.md.

Inside KS1's extension next to KS3 (namespace ks3, Params3): grid (NS, B), CTA = (request, split) with both KV heads and all
128 query rows; warp = (KV head, row tile of 2 tokens x 8 heads) walks every key of its split (no in-CTA key split): K / V of
the split land in FP8 smem by cp.async before the PDL wait (KS3 layout), q goes straight into mma A fragments (RMSNorm by
quad shuffles, RoPE, KS1's bf16 rounding), the tail split writes the 8 new keys (norm / RoPE / FP8, cache store) and attends
them causally; KS3's 32-key chunks with full 16-row A fragments and KS1's two-row online softmax; merge = absolute exp2
frame red.global.add.v4 into VSL's zeroed workspace, acq_rel ticket per request, the last CTA finalizes 128 rows
(normalize, bf16, RMSNorm * sigmoid(gate)) and zeroes what it read.
Routing through VSL: variant ("ks8", NS), VSL_ROUTE {(16, 8): ("ks8", NS), (32, 8): ...}; prepare() self-test per NS
(torch reference, cache bytes == r1, workspace / tickets clean, repeat, graph replay), fallback = the FA4 v6 chain.

    python -I tools/vsq_patches.py [--n=1] [--base=k3_slim_4] [--route=...] [--ks8=8,16] [--nolab]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
NL = chr(10)

KS8_CUDA = r'''
// ---- KS8 (tools/vsq_patches.py): one-launch q_len 8 multi-request sliding attention ----
template <int NS> struct G8 {
  static constexpr int KMAX = WIN / NS;                 // old-key slots per split
  static constexpr int NCH = KMAX / 32;                 // 32-key chunks (every warp walks all of them)
  static constexpr int OFF_V = KMAX * 128;
  static constexpr int OFF_KN = 2 * KMAX * 128;         // new-key tiles: 32 rows each (rows >= 8 zero)
  static constexpr int OFF_VN = OFF_KN + 32 * 128;
  static constexpr int OFF_MISC = OFF_VN + 32 * 128;
  static constexpr int SMEM = OFF_MISC + 64;
  static_assert(KMAX % 32 == 0 && (KMAX * 8) % NTHR == 0, "KS8 geometry");
};
#define KS8P(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

// 32 keys x 16 rows (rows g: token l0, g + 8: token l0 + 1), KS3's FP8 smem B fragments, KS1's two-row online softmax
template <typename F>
__device__ __forceinline__ void chunk8x16(const uint8_t* Ks, const uint8_t* Vs, const int x0, const int h, const uint32_t (&qa)[4][4],
                                          float (&o)[8][4], float (&m)[2], float (&ls)[2], const float cl2, F valid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  float sc[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int key = x0 + 8 * j + g;
    const uint4 kv = *reinterpret_cast<const uint4*>(Ks + key * 128 + 16 * kswz(key, 4 * h + t));
    const uint32_t wv[4] = {kv.x, kv.y, kv.z, kv.w};
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 4; ++s) mma16816(c, qa[s], e4m3x2_to_h2(wv[s]), e4m3x2_to_h2(wv[s] >> 16));
    sc[j][0] = c[0]; sc[j][1] = c[1]; sc[j][2] = c[2]; sc[j][3] = c[3];
  }
  float x0m = neg_inf(), x1m = neg_inf();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int k = x0 + 8 * j + 2 * t + e;
      sc[j][e] = valid(k, 0) ? sc[j][e] * cl2 : neg_inf();
      sc[j][2 + e] = valid(k, 1) ? sc[j][2 + e] * cl2 : neg_inf();
      x0m = fmaxf(x0m, sc[j][e]);
      x1m = fmaxf(x1m, sc[j][2 + e]);
    }
  }
  x0m = fmaxf(x0m, __shfl_xor_sync(FULL, x0m, 1));
  x0m = fmaxf(x0m, __shfl_xor_sync(FULL, x0m, 2));
  x1m = fmaxf(x1m, __shfl_xor_sync(FULL, x1m, 1));
  x1m = fmaxf(x1m, __shfl_xor_sync(FULL, x1m, 2));
  const float n0 = fmaxf(m[0], x0m), n1 = fmaxf(m[1], x1m);
  const float u0 = (n0 == neg_inf()) ? 0.f : n0, u1 = (n1 == neg_inf()) ? 0.f : n1;
  const float a0 = ex2(m[0] - u0), a1 = ex2(m[1] - u1);
  m[0] = n0; m[1] = n1;
  ls[0] *= a0; ls[1] *= a1;
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] *= a0; o[n][1] *= a0; o[n][2] *= a1; o[n][3] *= a1; }
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const float p00 = ex2(sc[2 * kk][0] - u0), p01 = ex2(sc[2 * kk][1] - u0);
    const float p02 = ex2(sc[2 * kk][2] - u1), p03 = ex2(sc[2 * kk][3] - u1);
    const float p10 = ex2(sc[2 * kk + 1][0] - u0), p11 = ex2(sc[2 * kk + 1][1] - u0);
    const float p12 = ex2(sc[2 * kk + 1][2] - u1), p13 = ex2(sc[2 * kk + 1][3] - u1);
    ls[0] += (p00 + p01) + (p10 + p11);
    ls[1] += (p02 + p03) + (p12 + p13);
    const uint32_t pa[4] = {pack_h2(p00, p01), pack_h2(p02, p03), pack_h2(p10, p11), pack_h2(p12, p13)};
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
__global__ void __launch_bounds__(NTHR, 2) ks8_kernel(const __grid_constant__ Params3 p) {
#if KS1_ARCH_OK
  using G = G8<NS>;
  constexpr int KMAX = G::KMAX;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + G::OFF_V;
  uint8_t* kn = smem + G::OFF_KN;
  uint8_t* vn = smem + G::OFF_VN;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y;
  const int cta = b * NS + s;
  const bool tail = (s == NS - 1);                              // also writes / attends the 8 new keys
  const int hw = warp & 1, l0 = 2 * (warp >> 1);                // KV head, first token of the row tile
  if (p.trig == 2) pdl_trigger();
  KS8P(0);

  const int seq = max(__ldg(p.cache_seqlens + b) - 8, 0);
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  {
    constexpr int PER = KMAX * 8 / NTHR;
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
  float4 cq[2][2], sn[2][2];                                    // cos / sin of RoPE pairs 8 t .. 8 t + 7 of tokens l0, l0 + 1
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    if (p.rope) {
      const long long pos = __ldg(p.positions + b * 8 + l0 + hf);
      const float4* cp = reinterpret_cast<const float4*>(p.cos_sin + pos * 64);
      cq[hf][0] = __ldg(cp + 2 * t); cq[hf][1] = __ldg(cp + 2 * t + 1);
      sn[hf][0] = __ldg(cp + 8 + 2 * t); sn[hf][1] = __ldg(cp + 8 + 2 * t + 1);
    } else {
      cq[hf][0] = cq[hf][1] = make_float4(1.f, 1.f, 1.f, 1.f);
      sn[hf][0] = sn[hf][1] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  float2 csk[4];
  long long loc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { csk[i] = make_float2(1.f, 0.f); loc[i] = 0; }
  if (tail) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {                               // new rows r = warp + 8 i: token r & 7, (kind, head) = r >> 3
      const int l = (warp + 8 * i) & 7;
      loc[i] = __ldg(p.cache_loc + b * 8 + l);
      if (p.rope) {
        const long long pos = __ldg(p.positions + b * 8 + l);
        csk[i] = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
      }
    }
    reinterpret_cast<uint4*>(kn)[tid] = make_uint4(0u, 0u, 0u, 0u);          // kn | vn: 8 KB, zero
    reinterpret_cast<uint4*>(kn)[tid + NTHR] = make_uint4(0u, 0u, 0u, 0u);
    __syncthreads();
  }
  KS8P(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  KS8P(2);

  const size_t sq = (size_t)p.stride_qkv;
  const __nv_bfloat16* xr = p.qkv + (size_t)b * 8 * sq;         // token l of request b: row xr + l sq
  uint4 qr[2][2];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    const uint4* qp = reinterpret_cast<const uint4*>(xr + (size_t)(l0 + hf) * sq + (size_t)(8 * hw + g) * 64 + 16 * t);
    qr[hf][0] = __ldcg(qp);
    qr[hf][1] = __ldcg(qp + 1);
  }
  uint32_t nr[4] = {0u, 0u, 0u, 0u};
  if (tail) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = warp + 8 * i, l = r & 7, idx = r >> 3;     // idx: 0 K h0, 1 K h1, 2 V h0, 3 V h1
      nr[i] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)l * sq + (size_t)(HQ + idx) * 64) + lane);
    }
  }
  uint32_t qa[4][4];
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {
    float v[16];
    const uint32_t w[8] = {qr[hf][0].x, qr[hf][0].y, qr[hf][0].z, qr[hf][0].w, qr[hf][1].x, qr[hf][1].y, qr[hf][1].z, qr[hf][1].w};
#pragma unroll
    for (int i = 0; i < 8; ++i) { const float2 x = bf2f(w[i]); v[2 * i] = x.x; v[2 * i + 1] = x.y; }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss = fmaf(v[i], v[i], ss);
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const float cc[8] = {cq[hf][0].x, cq[hf][0].y, cq[hf][0].z, cq[hf][0].w, cq[hf][1].x, cq[hf][1].y, cq[hf][1].z, cq[hf][1].w};
    const float ssn[8] = {sn[hf][0].x, sn[hf][0].y, sn[hf][0].z, sn[hf][0].w, sn[hf][1].x, sn[hf][1].y, sn[hf][1].z, sn[hf][1].w};
    uint32_t pr[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float e = bfr(v[2 * i] * rs), od = bfr(v[2 * i + 1] * rs);
      const float e2 = p.rope ? bfr(__fmaf_rn(e, cc[i], -__fmul_rn(od, ssn[i]))) : e;
      const float o2 = p.rope ? bfr(__fmaf_rn(e, ssn[i], __fmul_rn(od, cc[i]))) : od;
      pr[i] = pack_h2(e2, o2);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) { qa[k][hf] = pr[2 * k]; qa[k][2 + hf] = pr[2 * k + 1]; }
  }
  if (tail) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = warp + 8 * i, l = r & 7, idx = r >> 3, kind = idx >> 1, head = idx & 1;
      float2 v = kind ? bf2f(nr[i]) : norm_rope(nr[i], csk[i], p.rope, p.eps);
      v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
      v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
      const uint32_t q8 = (uint32_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc[i] * 128 + head * 64 + 2 * lane) = (uint16_t)q8;
      const int bb = 64 * head + 2 * lane, ch = bb >> 4;
      uint8_t* dst = kind ? (vn + l * 128 + 16 * vswz(l, ch)) : (kn + l * 128 + 16 * kswz(l, ch));
      *reinterpret_cast<uint16_t*>(dst + (bb & 15)) = (uint16_t)q8;
    }
  }
  asm volatile("cp.async.wait_all;\n" ::: "memory");
  __syncthreads();
  KS8P(3);

  const float cl2 = p.qk_sm * LOG2E;
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m[2] = {neg_inf(), neg_inf()}, ls[2] = {0.f, 0.f};
  const int base = s * KMAX;
#pragma unroll 1
  for (int c = 0; c < G::NCH; ++c) {
    if (base + 32 * c >= n_old) break;                          // (warp-uniform) the rest of the split is past the cache
    chunk8x16(Ks, Vs, 32 * c, hw, qa, o, m, ls, cl2, [&](int k, int hf) {
      const int gs = base + k;
      return gs < n_old && gs >= n_old + l0 + hf + 1 - WIN;     // > position(l) - WIN
    });
  }
  if (tail) chunk8x16(kn, vn, 0, hw, qa, o, m, ls, cl2, [&](int k, int hf) { return k < 8 && k <= l0 + hf; });
  ls[0] += __shfl_xor_sync(FULL, ls[0], 1);
  ls[0] += __shfl_xor_sync(FULL, ls[0], 2);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 1);
  ls[1] += __shfl_xor_sync(FULL, ls[1], 2);
  KS8P(4);

  float* wsl = reinterpret_cast<float*>(p.part_ml);
#pragma unroll
  for (int hf = 0; hf < 2; ++hf) {                              // row R = token * 16 + q head
    const int R = (l0 + hf) * 16 + 8 * hw + g;
    const float f = (m[hf] == neg_inf()) ? 0.f : ex2(fminf(m[hf], 126.f));
    float* wa = p.part_o + ((size_t)b * 128 + R) * 64;
#pragma unroll
    for (int e = 0; e < 2; ++e) {                               // o[jn][2 hf + e] = dim 8 (2 t + e) + jn
      const int d0 = 8 * (2 * t + e), q = 2 * hf + e;
      vsl_red4(wa + d0, o[0][q] * f, o[1][q] * f, o[2][q] * f, o[3][q] * f);
      vsl_red4(wa + d0 + 4, o[4][q] * f, o[5][q] * f, o[6][q] * f, o[7][q] * f);
    }
    if (t == 0) vsl_red1(wsl + (size_t)b * 128 + R, ls[hf] * f);
  }
  KS8P(5);
  __syncthreads();
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  KS8P(6);
  if (misc[0] == 0) return;

  {                                                             // last CTA of request b: 128 rows x 2 halves of 32 dims
    const int R = tid >> 1, hf = tid & 1, l = R >> 4, hq = R & 15;
    float* wa = p.part_o + ((size_t)b * 128 + R) * 64 + 32 * hf;
    float* wl = wsl + (size_t)b * 128 + R;
    const uint4* gp = reinterpret_cast<const uint4*>(xr + (size_t)l * sq + (size_t)(HQ + 2 * HKV) * 64 + (size_t)hq * 64 + 32 * hf);
    uint4 graw[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) graw[i] = __ldcg(gp + i);
    float a[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float4 x = __ldcg(reinterpret_cast<const float4*>(wa) + i);
      a[4 * i] = x.x; a[4 * i + 1] = x.y; a[4 * i + 2] = x.z; a[4 * i + 3] = x.w;
    }
    float den = (hf == 0) ? __ldcg(wl) : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) __stcg(reinterpret_cast<float4*>(wa) + i, make_float4(0.f, 0.f, 0.f, 0.f));
    if (hf == 0) __stcg(wl, 0.f);
    den = __shfl_sync(FULL, den, lane & ~1);
    const float dd = (den > 0.f) ? den : 1.f;
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) { a[i] = bfr(a[i] / dd); ss = fmaf(a[i], a[i], ss); }
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    uint32_t ow[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t gw[4] = {graw[i].x, graw[i].y, graw[i].z, graw[i].w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 gg = bf2f(gw[k]);
        const int d = 8 * i + 2 * k;
        const __nv_bfloat162 o2 = __floats2bfloat162_rn(a[d] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                        a[d + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
        ow[4 * i + k] = *reinterpret_cast<const uint32_t*>(&o2);
      }
    }
    uint4* op = reinterpret_cast<uint4*>(p.out + (size_t)(b * 8 + l) * (HQ * 64) + (size_t)hq * 64 + 32 * hf);
#pragma unroll
    for (int i = 0; i < 4; ++i) op[i] = make_uint4(ow[4 * i], ow[4 * i + 1], ow[4 * i + 2], ow[4 * i + 3]);
  }
  if (tid == 0) p.counters[b] = 0;
  KS8P(7);
#endif
}

template <int NS>
static void launch8(const Params3& P, bool pdl, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ks8_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G8<NS>::SMEM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTHR, 1, 1);
  cfg.dynamicSmemBytes = G8<NS>::SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1u : 0u;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ks8_kernel<NS>, P));
}

template <int NS>
static std::vector<int64_t> info8() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ks8_kernel<NS>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ks8_kernel<NS>, cudaFuncAttributeMaxDynamicSharedMemorySize, G8<NS>::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ks8_kernel<NS>, NTHR, G8<NS>::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)G8<NS>::SMEM, (int64_t)occ};
}

'''

KS8_HOST = r'''
#define KS8_INSTS @K8INSTS@

void ks8_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B * 8 &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64, "ks8: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && B >= 1, "ks8: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B * 8 && cache_loc.is_contiguous(), "ks8: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B * 8 && positions.is_contiguous(), "ks8: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "ks8: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ks8: caches");
  TORCH_CHECK(!rope || (cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.size(1) == 64), "ks8: cos_sin");
  TORCH_CHECK(part_o.scalar_type() == at::kFloat && part_o.numel() >= B * 128 * 64, "ks8: part_o");
  TORCH_CHECK(part_ml.scalar_type() == at::kFloat && part_ml.numel() >= B * 128, "ks8: part_ml");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ks8: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * 8 * ks1::HQ * 64, "ks8: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ks8: prof");
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
#define KS8_X(NS_) if (NS == NS_) { ks3::launch8<NS_>(P, pdl != 0, st); return; }
  KS8_INSTS
#undef KS8_X
  TORCH_CHECK(false, "ks8: NS not built");
}

std::vector<int64_t> ks8_info(int64_t NS) {
#define KS8_X(NS_) if (NS == NS_) return ks3::info8<NS_>();
  KS8_INSTS
#undef KS8_X
  return {};
}
'''

KS8_CPP = ("void ks8_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,\n"
           "                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,\n"
           "                 torch::Tensor part_o, torch::Tensor part_ml, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,\n"
           "                 int64_t NS, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);\n"
           "std::vector<int64_t> ks8_info(int64_t NS);\n")

PY_CALL_OLD = "    else:\n        _, ns, mg = v\n        out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)\n"
PY_CALL_NEW = ("    elif v[0] == \"ks8\":\n"
               "        out = torch.empty((B * 8, 16 * 64), dtype=torch.bfloat16, device=dev)\n"
               "        _KS1[\"mod\"].ks8_forward(qkv, data[\"cache_seqlens\"], data[\"cache_loc\"], data[\"positions\"], data[\"page_table\"],\n"
               "                                data[\"k_cache\"].view(torch.uint8), data[\"v_cache\"].view(torch.uint8), cos_sin, po, pml,\n"
               "                                _ks1_counters(dev), out, prof, int(v[1]), qk, KS1_EPS, int(VSQ_TRIG), 1 if KS1_PDL else 0,\n"
               "                                1 if cfg[\"rope_theta\"] else 0)\n"
               + PY_CALL_OLD)

PY_EDITS = [
    ("cpp decl", "std::vector<int64_t> ks3_info(int64_t NS);\n", "std::vector<int64_t> ks3_info(int64_t NS);\n" + KS8_CPP),
    ("build insts", '    cuda = (KS1_CUDA_SRC.replace("@INSTS@", insts).replace("@MINSTS@", minsts).replace("@M3INSTS@", m3insts)\n',
     '    k8insts = " ".join(f"KS8_X({n})" for n in VSQ_KS8_INSTS) if VSL_ENABLE else ""\n'
     '    cuda = (KS1_CUDA_SRC.replace("@INSTS@", insts).replace("@MINSTS@", minsts).replace("@M3INSTS@", m3insts)\n'
     '            .replace("@K8INSTS@", k8insts)\n'),
    ("build fns", 'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"] + _dg1_fold("fns"),',
     'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info", "ks8_forward", "ks8_info"] + _dg1_fold("fns"),'),
    ("consts", "VSL_PREPARE_BY_S = 240.0\n", "VSL_PREPARE_BY_S = 240.0\nVSQ_KS8_INSTS = @K8I@\nVSQ_TRIG = 1\n"),
    ("call", PY_CALL_OLD, PY_CALL_NEW),
    ("variants", '    return ([("ks1", l, n, g, m) for l, n, g, m in VSL_KS1_INSTS] + [("ks3", n, m) for n, m in VSL_KS3_INSTS])\n',
     '    return ([("ks1", l, n, g, m) for l, n, g, m in VSL_KS1_INSTS] + [("ks3", n, m) for n, m in VSL_KS3_INSTS]\n'
     '            + [("ks8", n) for n in VSQ_KS8_INSTS])\n'),
    ("cases", "def _vsl_cases(v):\n",
     "def _vsl_cases(v):\n"
     "    if v[0] == \"ks8\":\n"
     "        return [(16, 3000, 8, 1101, False), (32, 2600, 8, 1102, True), (3, 2100, 8, 1103, False), (2, 100, 8, 1104, False),\n"
     "                (16, 2049, 8, 1105, False), (4, 5, 8, 1106, False), (2, 2047, 8, 1107, False)]\n"),
    ("graph case", '                B, ctx, L = (1, 3000, v[1]) if v[0] == "ks1" else (32, 3000, 1)\n',
     '                B, ctx, L = (1, 3000, v[1]) if v[0] == "ks1" else ((16, 3000, 8) if v[0] == "ks8" else (32, 3000, 1))\n'),
    ("info", '        state["vsl"] = _vsl_self_test(state["cos_sin"], device)\n',
     '        for n in VSQ_KS8_INSTS:\n'
     '            print(f"[vsq] ks8 NS{n}: regs / local / smem / CTAs per SM {list(mod.ks8_info(n))}", flush=True)\n'
     '        state["vsl"] = _vsl_self_test(state["cos_sin"], device)\n'),
]


def check(ok, msg):
    if not ok:
        raise SystemExit(f"[vsq_patches] FAIL: {msg}")


def apply(text, route=None, ks8=(8, 16), log=None):
    """text = a stripped layer file with KS1 / KS3 + VSL (k3_slim_4 lineage); route = VSL_ROUTE literal (None: keep the base's)."""
    i0 = text.index('KS1_CUDA_SRC = r"""')
    i1 = text.index('"""', i0 + len('KS1_CUDA_SRC = r"""'))
    cu = text[i0:i1]
    a = "}  // namespace ks3\n"
    check(cu.count(a) == 1, "anchor namespace ks3")
    cu = cu.replace(a, KS8_CUDA + a)
    a = "std::vector<int64_t> ks3_info(int64_t NS) {\n"
    check(cu.count(a) == 1, "anchor ks3_info")
    cu = cu.replace(a, KS8_HOST + "\n" + a)
    t = text[:i0] + cu + text[i1:]
    for what, old, new in PY_EDITS:
        new = new.replace("@K8I@", repr(tuple(ks8)))
        check(t.count(old) == 1, f"Python anchor {what!r} found {t.count(old)} times")
        t = t.replace(old, new)
    if route is not None:
        lines = [ln for ln in t.split(NL) if ln.startswith("VSL_ROUTE = ")]
        check(len(lines) == 1, "VSL_ROUTE line")
        t = t.replace(lines[0], f"VSL_ROUTE = {route}")
    if log:
        log(f"VSQ applied: +{len(t.encode()) - len(text.encode())} B, KS8 NS {ks8}, route {route or '(base)'}")
    return t


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import gen_k3_m32 as GM
    sys.argv = _argv
    n = args.get("n", "1")
    base = ROOT / f"kernels/muse_decode_layer/{args.get('base', 'k3_slim_4')}.py"
    route = args.get("route")
    ks8 = tuple(int(x) for x in args.get("ks8", "8,16").split(","))
    out = ROOT / f"kernels/muse_decode_layer/k3_vsq_{n}.py"
    lab = ROOT / f"kernels/lab/k3_vsq_{n}_ab.py"
    check(not out.exists() or "--force" in sys.argv, f"{out.name} exists (new names only; --force to regenerate my own file)")
    tag = f"k3-vsq-{n}"
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    fb = GM.finish(b, bn, bn.split()[2], chk_base, [])
    check(fb == b, f"{base.name} is not a finish() fixed point")
    nonce = (f"# nonce {tag} ({base.stem} + VSQ: KS8 one-launch q_len 8 multi-request sliding attention, NS {list(ks8)}, "
             f"route {(route or 'base').replace(' ', '')}, self-tested with fallback)")
    t = apply(b.replace(bn, nonce), route, ks8, rep.append)
    s = GM.finish(t, nonce, tag, chk_base, rep)
    for r in rep:
        print(f"[vsq_patches] OK  {r}")
    out.write_text(s, encoding="utf-8", newline=NL)
    if "--nolab" not in sys.argv:
        lab.write_text(s, encoding="utf-8", newline=NL)
    print(f"[vsq_patches] wrote {out.relative_to(ROOT)}{'' if '--nolab' in sys.argv else ' and lab copy'} ({len(s.encode())} B, "
          f"headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

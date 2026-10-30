"""ax3_patches.py: AX3 = decode SLIDING attention at B = 16 / 32 (q_len 1) as 4-warp CTAs over 64-key tiles with per-tile
consumption of the gather, our replacement for KS3 (ks3_kernel: 8 warps, waits for the WHOLE split before any math).

Why (vjeux's public kernel, read for ideas only; notes/lab.md R-ax3 rows): on the attention board its d16s / d32s are 7.8 / 9.6 us
vs KS3's 9.7 / 12.9; R-ks3g-0 put KS3's excess over a null gather in the post-landing chain (attention 1.5-2 us + merge 2.5 us).
AX3 changes, vs KS3 (all our own code; KS3's chunk32 / norm_rope / swizzles are reused unchanged):
  * 128-thread CTAs (warp w: KV head w & 1, key half w >> 1 of every 64-key tile), smem = the split's K/V + 2 KB q (NS 8: 66 KB);
  * the page-table entries of the whole split in one round trip, then every tile's K/V cp.async in flight at once, one commit
    group per tile; the tile loop (rolled: one cold copy of the math) consumes tile x after wait_group(NTL - 1 - x) + bar.sync;
  * the new key sits in its window slot (slot n_old of the split that owns it; written with st.shared after the PDL wait), no
    extra 32-key tile and no extra chunk32 copy;
  * CTA merge of the two key halves through smem, then red.global.add in the absolute exp2 frame (KS3 MG 1 / VSL numerics) into a
    zeroed fp32 workspace, acq_rel ticket, the last CTA finalizes 16 rows x 64 (8 dims per thread) and re-zeroes.
Lab knobs (template VAR): bit 0 = wait for the whole split before the first tile (KS3's order; isolates the pipelining).

Routing: AX3_ROUTE {(batch, 1): (NS, VAR)} read at call time; only variants that pass prepare()'s self-test (torch reference,
new-key cache bytes == r1, workspace / tickets zero after every call, repeat, CUDA-graph replay) are routable; anything else keeps
KS3 / VSL. --sweep (board lab only) adds an in-prepare timed sweep (CUDA graph, 256 MB L2 flush + 300 us gate per replay, arms
interleaved) of KS3 as shipped vs every AX3 instance on board-like inputs (B 16 / 32, ctx 80000, page 1, random rows), with marks.

    python -I tools/ax3_patches.py --base=kernels/lab/k3_rider3_1.py --out=kernels/lab/k3_ax3_1.py
    python -I tools/ax3_patches.py --base=kernels/muse_attention_decode/vfa_gc16as.py \
        --out=kernels/muse_attention_decode/ax3_1.py --sweep
    options: --route="{(16, 1): (16, 0), (32, 1): (8, 0)}"  --insts="((8, 0), (16, 0))"  --nonce=<text>
"""
import argparse
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)

DEF_INSTS = ((16, 0), (8, 0), (8, 2), (8, 4), (4, 2), (4, 4), (16, 2))
DEF_ROUTE = {(16, 1): (8, 2), (32, 1): (8, 2)}

# ---------------------------------------------------------------------------------------------------------------------------- CUDA
CPP_DECL = r'''void ax3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t VAR, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope);
std::vector<int64_t> ax3_info(int64_t NS, int64_t VAR);
'''

CUDA_SRC = r'''
// ---- AX3 (tools/ax3_patches.py): decode SLIDING attention, B >= 2, q_len 1; 4-warp CTAs over 64-key tiles ----
namespace ax3 {
using namespace ks1;

constexpr int NTA = 128;   // warp w: KV head (w & 1), key half (w >> 1) of every 64-key tile

struct PA {
  const __nv_bfloat16* qkv;   // [B, stride] q | k | v | gate
  const int* cache_seqlens;   // [B] (old length + 1)
  const int64_t* cache_loc;   // [B]
  const int64_t* positions;   // [B]
  const int* page_table;      // [B, pt_cols]
  uint8_t* k_cache;           // e4m3 rows of 2 x 64 B (both KV heads), page size 1
  uint8_t* v_cache;
  const float* cos_sin;       // [P, 64] cos | sin
  float* ws_o;                // [B][16][64] fp32, zero between calls (the request's last CTA re-zeroes its rows)
  float* ws_l;                // [B][16]
  int* counters;              // [B] arrival tickets, zero between calls
  __nv_bfloat16* out;         // [B, 1024]
  long long* prof;            // [B NS][8] %globaltimer marks, or nullptr
  int stride_qkv, pt_cols, B, trig, rope;
  float qk_sm, eps;
};

template <int NS, int AH> struct GA {
  static constexpr int KMAX = WIN / NS;           // window slots per split (2047 old keys + the new one = 2048)
  static constexpr int NTL = KMAX / 64;           // 64-key tiles per split
  static constexpr int NST = AH == 0 ? NTL : (AH + 1 < NTL ? AH + 1 : NTL);   // K / V stages (a ring when AH > 0)
  static constexpr int OFF_V = NST * 64 * 128;
  static constexpr int OFF_Q = 2 * NST * 64 * 128;  // q fp16 [16 heads][64]
  static constexpr int OFF_ML = OFF_Q + 16 * 128;   // [4 warps][8 rows] max, then sum
  static constexpr int OFF_MISC = OFF_ML + 2 * 4 * 8 * 4;
  static constexpr int SMEM = OFF_MISC + 16;
  static_assert(NTL >= 1 && NTL <= 8 && (AH > 0 || NTL <= 4) && 2 * NST * 64 * 128 >= 4 * 8 * 64 * 4, "AX3 geometry");
};

template <int N> __device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cpwait_rt(int n) {   // n = groups still allowed in flight (0..3)
  if (n <= 0) cpwait<0>();
  else if (n == 1) cpwait<1>();
  else if (n == 2) cpwait<2>();
  else cpwait<3>();
}
#define AXP(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[cta * 8 + (k)] = gtime(); } while (0)

// VAR bit 0 (lab): whole split landed before the first tile (KS3's order); bit 1 / bit 2: a ring with AH = 2 / 3 tiles in
// flight, the loop issues tile x + AH right after the barrier of tile x (VAR 0: every tile of the split in flight at once);
// bit 3: L2 prefetch of the merge workspace rows, row sums and ticket at entry (cold lines: the reds / ticket miss to DRAM).
template <int NS, int VAR>
__global__ void __launch_bounds__(NTA, 4) ax3_kernel(const __grid_constant__ PA p) {
#if KS1_ARCH_OK
  constexpr int AH = (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0);
  using G = GA<NS, AH>;
  constexpr int KMAX = G::KMAX, NTL = G::NTL, NST = G::NST;
  constexpr bool WALL = AH == 0 && (VAR & 1) != 0;
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* Ks = smem;
  uint8_t* Vs = smem + G::OFF_V;
  uint8_t* qs = smem + G::OFF_Q;
  float* mw = reinterpret_cast<float*>(smem + G::OFF_ML);
  float* lw = mw + 32;
  int* misc = reinterpret_cast<int*>(smem + G::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int s = blockIdx.x, b = blockIdx.y, cta = b * NS + s;
  const int hw = warp & 1, kh = warp >> 1;
  if (p.trig == 2) pdl_trigger();
  AXP(0);
  if constexpr ((VAR & 8) != 0) {                            // the red / ticket lines toward L2 now, not at the merge
    if (tid < 34) {
      const void* a = tid < 32 ? (const void*)(p.ws_o + (size_t)b * 1024 + 32 * tid)
                               : (tid == 32 ? (const void*)(p.ws_l + (size_t)b * 16) : (const void*)(p.counters + b));
      asm volatile("prefetch.global.L2 [%0];" ::"l"(a));
    }
  }

  const int seq = max(__ldg(p.cache_seqlens + b) - 1, 0);   // old keys; the new key sits at position seq
  const int n_old = min(seq, WIN - 1);
  const int kstart = seq - n_old;
  const int base = s * KMAX;                                 // first window slot of this split
  const int nk = min(KMAX, n_old + 1 - base);                // slots with a key here (old ones + the new one); <= 0: none
  const int ntl = nk > 0 ? (nk + 63) >> 6 : 0;
  const int jn = n_old - base;                               // the new key's slot (this CTA owns it if 0 <= jn < KMAX)
  const bool owner = jn >= 0 && jn < KMAX;
  const int xn = jn >> 6;                                    // (owner) the new key's tile
  const int jrow = (xn % NST) * 64 + (jn & 63);              // (owner) its row in the stage buffers
  // thread: 16-B chunk c of rows r0 + 16 k of a tile; every page entry of the split first (one round trip, registers)
  const int c = tid & 7, r0 = tid >> 3;
  int rows[NTL][4];
  {
    const int* pt = p.page_table + (size_t)b * p.pt_cols;
#pragma unroll
    for (int x = 0; x < NTL; ++x)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int j = base + 64 * x + r0 + 16 * k;
        rows[x][k] = (j < n_old && kstart + j < p.pt_cols) ? __ldg(pt + kstart + j) : -1;
      }
  }
  auto issue = [&](const int x, const int (&rw)[4]) {      // tile x -> stage x % NST (the new key's slot is skipped)
    const int sb = (x % NST) * 64;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = r0 + 16 * k, rs = sb + r;
      if (64 * x + r != jn) {
        const bool ok = rw[k] >= 0;                          // slots past the window end: zero fill (V must be finite)
        const size_t go = (size_t)(ok ? rw[k] : 0) * 128 + 16 * c;
        ks3::cp_async16(Ks + rs * 128 + 16 * ks3::kswz(rs, c), p.k_cache + go, ok ? 16 : 0);
        ks3::cp_async16(Vs + rs * 128 + 16 * ks3::vswz(rs, c), p.v_cache + go, ok ? 16 : 0);
      }
    }
  };
  constexpr int PRE = AH == 0 ? NTL : AH;                    // tiles issued before the PDL wait (one commit group each)
#pragma unroll
  for (int x = 0; x < PRE; ++x) {
    if (x < ntl) issue(x, rows[x]);
    asm volatile("cp.async.commit_group;\n" ::: "memory");
  }
  float2 cs = make_float2(1.f, 0.f);
  if (p.rope) {
    const long long pos = __ldg(p.positions + b);
    cs = make_float2(__ldg(p.cos_sin + pos * 64 + lane), __ldg(p.cos_sin + pos * 64 + 32 + lane));
  }
  const long long loc = owner ? __ldg(p.cache_loc + b) : 0;
  AXP(1);
  pdl_wait();
  if (p.trig == 1) pdl_trigger();

  const __nv_bfloat16* xr = p.qkv + (size_t)b * p.stride_qkv;
  const int mr = tid >> 3, mp = tid & 7;                     // merge / finalize item: q head mr, dims 8 mp .. 8 mp + 7
  const uint4 graw = __ldcg(reinterpret_cast<const uint4*>(xr + (size_t)(HQ + 2 * HKV) * 64 + (size_t)mr * 64 + 8 * mp));
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) { o[n][0] = 0.f; o[n][1] = 0.f; o[n][2] = 0.f; o[n][3] = 0.f; }
  float m = neg_inf(), l = 0.f;
  if (nk > 0) {
    uint32_t qr[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) qr[i] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(warp + 4 * i) * 64) + lane);
    uint32_t nr = 0u;
    if (owner) nr = __ldcg(reinterpret_cast<const unsigned int*>(xr + (size_t)(HQ + warp) * 64) + lane);   // k0 k1 v0 v1
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 a = ks3::norm_rope(qr[i], cs, p.rope, p.eps);
      reinterpret_cast<uint32_t*>(qs + (warp + 4 * i) * 128)[lane] = pack_h2(a.x, a.y);
    }
    uint16_t nq8 = 0;
    uint8_t* ndst = smem;
    if (owner) {                                             // warp 0 / 1: K head 0 / 1, warp 2 / 3: V head 0 / 1
      const int kind = warp >> 1, head = warp & 1;
      float2 v = kind ? bf2f(nr) : ks3::norm_rope(nr, cs, p.rope, p.eps);
      v.x = fminf(fmaxf(v.x, -FP8_MAX), FP8_MAX);
      v.y = fminf(fmaxf(v.y, -FP8_MAX), FP8_MAX);
      nq8 = (uint16_t)__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + (size_t)loc * 128 + head * 64 + 2 * lane) = nq8;
      const int bb = 64 * head + 2 * lane, cc = bb >> 4;
      ndst = (kind ? (Vs + jrow * 128 + 16 * ks3::vswz(jrow, cc)) : (Ks + jrow * 128 + 16 * ks3::kswz(jrow, cc))) + (bb & 15);
      if (xn < PRE) *reinterpret_cast<uint16_t*>(ndst) = nq8;   // its tile is issued already (else: right after its issue)
    }
    AXP(2);
    if constexpr (WALL) cpwait<0>();
    __syncthreads();                                         // q of every warp, the new key's slot
    uint32_t qf[4][2];
    {
      const uint32_t* qa = reinterpret_cast<const uint32_t*>(qs + (hw * 8 + g) * 128);
#pragma unroll
      for (int k = 0; k < 4; ++k) { qf[k][0] = qa[8 * t + 2 * k]; qf[k][1] = qa[8 * t + 2 * k + 1]; }
    }
    const float cl2 = p.qk_sm * LOG2E;
#pragma unroll 1
    for (int x = 0; x < ntl; ++x) {
      if constexpr (AH == 0) {
        if constexpr (!WALL) {
          cpwait_rt(NTL - 1 - x);
          __syncthreads();                                   // tile x landed (every thread's copies)
        }
      } else {
        cpwait<AH - 1>();
        __syncthreads();                                     // tile x landed; every warp is done with tile x - 1's stage
        const int y = x + AH;
        if (y < ntl) {
          int rw[4] = {-1, -1, -1, -1};
#pragma unroll
          for (int yy = AH; yy < NTL; ++yy)
            if (yy == y) { rw[0] = rows[yy][0]; rw[1] = rows[yy][1]; rw[2] = rows[yy][2]; rw[3] = rows[yy][3]; }
          issue(y, rw);
          if (owner && xn == y) *reinterpret_cast<uint16_t*>(ndst) = nq8;
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
      }
      const int sb = (x % NST) * 64;
      ks3::chunk32(Ks, Vs, sb + 32 * kh, hw, qf, o, m, l, cl2, [&](int k) { return 64 * x + (k - sb) < nk; });
    }
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
  }
  AXP(3);

  __syncthreads();                                           // K / V reads done: the merge scratch aliases them
  float* ow = reinterpret_cast<float*>(smem);                // [4 warps][8 rows][64], dims XOR 4 g
  if (nk > 0) {
    if (t == 0) { mw[warp * 8 + g] = m; lw[warp * 8 + g] = l; }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      ow[(warp * 8 + g) * 64 + ((16 * t + j) ^ (4 * g))] = o[j][0];
      ow[(warp * 8 + g) * 64 + ((16 * t + 8 + j) ^ (4 * g))] = o[j][1];
    }
  }
  __syncthreads();
  if (nk > 0) {                                              // two key halves of row mr -> absolute exp2 frame -> red.add
    const int rh = mr >> 3, rg = mr & 7, w0 = rh, w1 = rh + 2;
    const float m0 = mw[w0 * 8 + rg], m1 = mw[w1 * 8 + rg];
    const float M = fmaxf(m0, m1);
    const float Mu = (M == neg_inf()) ? 0.f : M;
    const float fa = (M == neg_inf()) ? 0.f : ex2(fminf(M, 126.f));
    const float f0 = ex2(m0 - Mu) * fa, f1 = ex2(m1 - Mu) * fa;
    const float lt = f0 * lw[w0 * 8 + rg] + f1 * lw[w1 * 8 + rg];
    const float* r0p = ow + (w0 * 8 + rg) * 64;
    const float* r1p = ow + (w1 * 8 + rg) * 64;
    const int ia = (8 * mp) ^ (4 * rg), ib = (8 * mp + 4) ^ (4 * rg);
    const float4 x0 = *reinterpret_cast<const float4*>(r0p + ia), y0 = *reinterpret_cast<const float4*>(r1p + ia);
    const float4 x1 = *reinterpret_cast<const float4*>(r0p + ib), y1 = *reinterpret_cast<const float4*>(r1p + ib);
    float* wa = p.ws_o + ((size_t)b * 16 + mr) * 64 + 8 * mp;
    vsl_red4(wa, f0 * x0.x + f1 * y0.x, f0 * x0.y + f1 * y0.y, f0 * x0.z + f1 * y0.z, f0 * x0.w + f1 * y0.w);
    vsl_red4(wa + 4, f0 * x1.x + f1 * y1.x, f0 * x1.y + f1 * y1.y, f0 * x1.z + f1 * y1.z, f0 * x1.w + f1 * y1.w);
    if (mp == 0) vsl_red1(p.ws_l + (size_t)b * 16 + mr, lt);
  }
  AXP(4);
  __syncthreads();                                           // every thread's reds before the release (bar.sync cumulativity)
  if (tid == 0) misc[0] = (atom_add_acqrel(p.counters + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  AXP(5);
  if (misc[0] == 0) return;
  {                                                          // last CTA of request b: 16 rows x 64 from the workspace
    float* wa = p.ws_o + ((size_t)b * 16 + mr) * 64 + 8 * mp;
    float* wl = p.ws_l + (size_t)b * 16 + mr;
    float den = (mp == 0) ? __ldcg(wl) : 0.f;
    const float4 v0 = __ldcg(reinterpret_cast<const float4*>(wa));
    const float4 v1 = __ldcg(reinterpret_cast<const float4*>(wa + 4));
    __stcg(reinterpret_cast<float4*>(wa), make_float4(0.f, 0.f, 0.f, 0.f));
    __stcg(reinterpret_cast<float4*>(wa + 4), make_float4(0.f, 0.f, 0.f, 0.f));
    if (mp == 0) __stcg(wl, 0.f);
    den = __shfl_sync(FULL, den, lane & ~7);
    const float dd = (den > 0.f) ? den : 1.f;
    const float y[8] = {bfr(v0.x / dd), bfr(v0.y / dd), bfr(v0.z / dd), bfr(v0.w / dd),
                        bfr(v1.x / dd), bfr(v1.y / dd), bfr(v1.z / dd), bfr(v1.w / dd)};
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(y[i], y[i], ss);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 1);
    const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const uint32_t gw[4] = {graw.x, graw.y, graw.z, graw.w};
    uint32_t ov[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 gg = bf2f(gw[i]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * i] * rs * (1.f / (1.f + __expf(-gg.x))),
                                                      y[2 * i + 1] * rs * (1.f / (1.f + __expf(-gg.y))));
      ov[i] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    *reinterpret_cast<uint4*>(p.out + (size_t)b * (HQ * 64) + (size_t)mr * 64 + 8 * mp) = make_uint4(ov[0], ov[1], ov[2], ov[3]);
  }
  if (tid == 0) p.counters[b] = 0;                           // self-reset (every split of request b has arrived)
  AXP(6);
#endif
}
#undef AXP

template <int NS, int VAR>
static void launch_a(const PA& P, bool pdl, cudaStream_t st) {
  constexpr int SM = GA<NS, (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0)>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ax3_kernel<NS, VAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)NS, (unsigned)P.B, 1);
  cfg.blockDim = dim3((unsigned)NTA, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = st;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ax3_kernel<NS, VAR>, P));
}

template <int NS, int VAR>
static std::vector<int64_t> info_a() {
  constexpr int SM = GA<NS, (VAR & 2) ? 2 : ((VAR & 4) ? 3 : 0)>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ax3_kernel<NS, VAR>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ax3_kernel<NS, VAR>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ax3_kernel<NS, VAR>, NTA, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

}  // namespace ax3

#define AX3_INSTS @AX3INSTS@

void ax3_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor positions,
                 torch::Tensor page_table, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor cos_sin,
                 torch::Tensor ws_o, torch::Tensor ws_l, torch::Tensor counters, torch::Tensor out, torch::Tensor prof,
                 int64_t NS, int64_t VAR, double qk_sm, double eps, int64_t trig, int64_t pdl, int64_t rope) {
  const int64_t B = cache_seqlens.numel();
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == B &&
              qkv.size(1) >= (2 * ks1::HQ + 2 * ks1::HKV) * 64 && qkv.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "ax3: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && B >= 1, "ax3: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.numel() == B && cache_loc.is_contiguous(), "ax3: cache_loc");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.numel() == B && positions.is_contiguous(), "ax3: positions");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.dim() == 2 && page_table.size(0) == B && page_table.is_contiguous(),
              "ax3: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && v_cache.scalar_type() == at::kByte && k_cache.is_contiguous() &&
              v_cache.is_contiguous() && k_cache.numel() % (ks1::HKV * 64) == 0, "ax3: caches");
  TORCH_CHECK(!rope || (cos_sin.scalar_type() == at::kFloat && cos_sin.is_contiguous() && cos_sin.size(1) == 64), "ax3: cos_sin");
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.numel() >= B * 16 * 64 && ws_l.scalar_type() == at::kFloat &&
              ws_l.numel() >= B * 16, "ax3: workspace");
  TORCH_CHECK(counters.scalar_type() == at::kInt && counters.numel() >= B, "ax3: counters");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == B * ks1::HQ * 64, "ax3: out");
  TORCH_CHECK(prof.numel() == 0 || (prof.scalar_type() == at::kLong && prof.numel() >= 8 * B * NS), "ax3: prof");
  const at::cuda::CUDAGuard guard(qkv.device());
  ax3::PA P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.cache_seqlens = cache_seqlens.data_ptr<int>();
  P.cache_loc = cache_loc.data_ptr<int64_t>();
  P.positions = positions.data_ptr<int64_t>();
  P.page_table = page_table.data_ptr<int>();
  P.k_cache = k_cache.data_ptr<uint8_t>();
  P.v_cache = v_cache.data_ptr<uint8_t>();
  P.cos_sin = cos_sin.data_ptr<float>();
  P.ws_o = ws_o.data_ptr<float>();
  P.ws_l = ws_l.data_ptr<float>();
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
#define AX3_X(NS_, VAR_) if (NS == NS_ && VAR == VAR_) { ax3::launch_a<NS_, VAR_>(P, pdl != 0, st); return; }
  AX3_INSTS
#undef AX3_X
  TORCH_CHECK(false, "ax3: variant not built");
}

std::vector<int64_t> ax3_info(int64_t NS, int64_t VAR) {
#define AX3_X(NS_, VAR_) if (NS == NS_ && VAR == VAR_) return ax3::info_a<NS_, VAR_>();
  AX3_INSTS
#undef AX3_X
  return {};
}
'''

# -------------------------------------------------------------------------------------------------------------------------- Python
PY_BLOCK = r'''AX3_ENABLE = True

AX3_INSTS = @PYINSTS@

AX3_ROUTE = @PYROUTE@

AX3_TRIG = {(16, 1): 1, (32, 1): 1}

AX3_SELF_TEST_CASES = ((32, 2600, 961, True), (16, 3000, 962, False), (3, 2100, 963, False), (5, 100, 965, False),
                       (4, 1, 966, False), (2, 2048, 967, False), (2, 2049, 968, True), (32, 2047, 969, False),
                       (16, 80000, 964, False))

AX3_PREPARE_BY_S = 245.0

AX3_SWEEP = @PYSWEEP@

_AX3 = {"ws": {}, "seen": set()}


def _ax3_ws(dev):
    key = str(dev)
    if key not in _AX3["ws"]:
        _AX3["ws"][key] = (torch.zeros(512 * 16 * 64, dtype=torch.float32, device=dev),
                           torch.zeros(512 * 16, dtype=torch.float32, device=dev),
                           torch.zeros(512, dtype=torch.int32, device=dev))
    return _AX3["ws"][key]


def _ax3_pick(cfg, state):
    if not AX3_ENABLE or cfg["q_len"] != 1 or cfg["batch"] > 512:
        return None
    v = AX3_ROUTE.get((cfg["batch"], 1))
    if v is None:
        return None
    v = tuple(v)
    return v if v in (state.get("ax3") or ()) else None


def _ax3_call(qkv, data, cos_sin, v, trig=None, pdl=None, prof=None):
    cfg = data["config"]
    B = cfg["batch"]
    ns, var = v
    dev = qkv.device
    wo, wl, cnt = _ax3_ws(dev)
    out = torch.empty((B, 16 * 64), dtype=torch.bfloat16, device=dev)
    if prof is None:
        prof = torch.empty((0,), dtype=torch.int64, device=dev)
    trig = AX3_TRIG.get((B, 1), 1) if trig is None else trig
    pdl = KS1_PDL if pdl is None else pdl
    _KS1["mod"].ax3_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                            data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, wo, wl, cnt, out,
                            prof, int(ns), int(var), float(cfg["sm_scale"] * cfg["qk_scale_factor"]), KS1_EPS, int(trig),
                            1 if pdl else 0, 1 if cfg["rope_theta"] else 0)
    if v not in _AX3["seen"]:
        _AX3["seen"].add(v)
        print(f"[ax3] first call {v} B={B}", flush=True)
    return out


def _ax3_self_test(cos_sin, device):
    wo, wl, cnt = _ax3_ws(device)
    ok = set()
    for v in AX3_INSTS:
        v = tuple(v)
        good, worst = True, 0.0
        cnt.zero_()
        wo.zero_()
        wl.zero_()
        try:
            for B, ctx, seed, row0 in AX3_SELF_TEST_CASES:
                ref = _k1fa4_reference(_k1fa4_case(B, ctx, 1, 2048, seed, row0, device)).view(B, -1)
                d1 = _k1fa4_case(B, ctx, 1, 2048, seed, row0, device)
                o1 = _k1fa4_r1_path(d1).float().view(B, -1)
                torch.cuda.synchronize(device)
                e1 = float((o1 - ref).norm() / ref.norm())
                rows = d1["cache_loc"]
                k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                dk = _k1fa4_case(B, ctx, 1, 2048, seed, row0, device)
                qkv = dk["qkv"].view(B, -1)
                o_a = _ax3_call(qkv, dk, cos_sin, v).clone()
                o_b = _ax3_call(qkv, dk, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                ek = float((o_a.float() - ref).norm() / ref.norm())
                kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
                clean = int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0 and float(wl.abs().sum()) == 0.0
                rep = float((o_a.float() - o_b.float()).abs().max()) <= 0.02 * float(o_a.float().abs().max()) + 1e-3
                c_ok = bool(torch.isfinite(o_a.float()).all()) and ek <= 1.25 * e1 + 5e-3 and nbad == 0 and clean and rep
                worst = max(worst, ek)
                if not c_ok:
                    good = False
                    print(f"[ax3] self-test {v} B{B} ctx{ctx}{' row0' if row0 else ''}: FAIL err {ek:.4e} (r1 {e1:.4e}) "
                          f"cache bytes != r1 {nbad} clean {clean} repeat {rep}", flush=True)
                    cnt.zero_()
                    wo.zero_()
                    wl.zero_()
                del dk, o_a, o_b, d1, o1, ref
                if not good:
                    break
            if good:
                d = _k1fa4_case(32, 3000, 1, 2048, 977, False, device)
                qkv = d["qkv"].view(32, -1)
                o1 = _ax3_call(qkv, d, cos_sin, v).clone()
                torch.cuda.synchronize(device)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    og = _ax3_call(qkv, d, cos_sin, v)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize(device)
                g_ok = (float((og.float() - o1.float()).abs().max()) <= 0.02 * float(o1.float().abs().max()) + 1e-3
                        and int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0)
                del gr, og, d
                if not g_ok:
                    good = False
                    print(f"[ax3] graph check {v}: FAIL", flush=True)
        except Exception as ex:  # noqa: BLE001
            good = False
            print(f"[ax3] self-test {v} raised {repr(ex)[-600:]}", flush=True)
            cnt.zero_()
            wo.zero_()
            wl.zero_()
        print(f"[ax3] variant {v}: {'ok' if good else 'FAILED'} (worst err {worst:.3e})", flush=True)
        if good:
            ok.add(v)
    return ok


def _ax3_prepare(state, device, t0):
    state["ax3"] = set()
    if not AX3_ENABLE or _KS1["mod"] is None or not hasattr(_KS1["mod"], "ax3_forward"):
        return
    if time.time() - t0 > AX3_PREPARE_BY_S:
        print(f"[ax3] skipped: prepare already took {time.time() - t0:.1f}s: KS3 stays", flush=True)
        return
    t1 = time.time()
    try:
        for ns, var in AX3_INSTS:
            print(f"[ax3] NS{ns} VAR{var}: regs / local / smem / CTAs per SM {list(_KS1['mod'].ax3_info(ns, var))}", flush=True)
        state["ax3"] = _ax3_self_test(state["cos_sin"], device)
    except Exception as ex:  # noqa: BLE001
        print(f"[ax3] self-test FAILED {repr(ex)[-1000:]}: KS3 stays", flush=True)
        state["ax3"] = set()
    live = {k: tuple(v) for k, v in AX3_ROUTE.items() if tuple(v) in state["ax3"]}
    print(f"[ax3] route live {live} ({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    if AX3_SWEEP:
        try:
            _ax3_sweep(state, device)
        except Exception as ex:  # noqa: BLE001
            print(f"[ax3] sweep FAILED {repr(ex)[-1000:]}", flush=True)
        torch.cuda.synchronize(device)


'''

SWEEP_BLOCK = r'''AX3_SWEEP_REPS = 40


def _ax3_board_case(B, seed, device):
    # board-like decode sliding input: ctx 80000 +- 7.5 % (agentic), page 1, scattered rows (as the reference builds them)
    g = torch.Generator(device=device).manual_seed(seed)
    ctx = 80000
    seq = (ctx * (1 + 0.075 * (2 * torch.rand((B,), generator=g, device=device) - 1))).long().clamp(1, 86000)
    max_pages = 86000 + 1
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=device).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b in range(B):
        n = int(seq[b]) + 1
        pt[b, :n] = perm[off:off + n]
        off += n
    f8 = torch.float8_e4m3fn
    k_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    v_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    for c in (k_cache, v_cache):
        cv = c.view(-1, 128)
        for i in range(0, cv.shape[0], 1 << 20):
            cv[i:i + (1 << 20)] = torch.randn((min(1 << 20, cv.shape[0] - i), 128), generator=g, device=device).to(f8)
    q = (2 * torch.randn((B, 1, 16, 64), generator=g, device=device)).to(torch.bfloat16)
    k = (2 * torch.randn((B, 1, 2, 64), generator=g, device=device)).to(torch.bfloat16)
    v = torch.randn((B, 1, 2, 64), generator=g, device=device).to(torch.bfloat16)
    gate = torch.randn((B, 1, 1024), generator=g, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2), gate], -1)
    positions = seq.clone()
    cache_loc = torch.gather(pt.long(), 1, positions.view(B, 1)).reshape(-1)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {"qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": (seq + 1).to(torch.int32),
            "k_cache": k_cache, "v_cache": v_cache, "k_scale": one, "v_scale": one.clone(), "page_table": pt,
            "seq_lens": seq.to(torch.int32),
            "config": {"batch": B, "q_len": 1, "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64, "page_size": 1,
                       "window": 2048, "rope_theta": 500000.0, "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125,
                       "kv_scale": 1.0}}


def _ax3_sweep(state, device):
    import statistics
    cs = state["cos_sin"]
    flush = torch.ones((64 << 20,), dtype=torch.float32, device=device)   # 256 MB, read before every timed replay
    sink = torch.empty((), dtype=torch.float32, device=device)
    stream = torch.cuda.Stream(device=device)
    cyc = int(300 * 1980)
    for _ in range(3):   # calibrate a ~300 us spin gate (as the judge's)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        torch.cuda._sleep(cyc)
        b.record()
        b.synchronize()
        cyc = int(cyc * 300.0 / max(a.elapsed_time(b) * 1e3, 1.0))

    def capture(fn):
        with torch.cuda.stream(stream):
            fn()
            torch.cuda.synchronize(device)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=stream):
                out = fn()
        torch.cuda.synchronize(device)
        return gr, out

    tiny = torch.zeros(1, device=device)
    g_empty, _ = capture(lambda: tiny.add_(1.0))
    for B in (16, 32):
        d = _ax3_board_case(B, 7100 + B, device)
        qkv = d["qkv"].view(B, -1)
        arms = []
        if B == 16 and ("ks3", 16, 1) in (state.get("vsl") or ()):
            arms.append(("KS3 ship NS16 MG1", lambda: _vsl_call(qkv, d, cs, ("ks3", 16, 1)), None))
        elif _ks3_routed((B, 1), state):
            arms.append((f"KS3 ship NS{KS3_NS.get((B, 1))} MG0", lambda: _ks3_call(qkv, d, cs), None))
        for vv in (state.get("vsl") or ()):
            if vv[0] == "ks3" and not (B == 16 and tuple(vv) == ("ks3", 16, 1)):
                arms.append((f"KS3 NS{vv[1]} MG{vv[2]}", lambda vv=vv: _vsl_call(qkv, d, cs, vv), None))
        for v in AX3_INSTS:
            v = tuple(v)
            if v in state["ax3"]:
                arms.append((f"AX3 NS{v[0]} VAR{v[1]}", lambda v=v: _ax3_call(qkv, d, cs, v), v))
        graphs = []
        for name, fn, v in arms:
            gr, out = capture(fn)
            graphs.append((name, gr, out, v))   # (the graph's output buffer: read after the replays)
        ts = {name: [] for name, _, _, _ in graphs}
        te = []
        for _ in range(AX3_SWEEP_REPS):
            for name, gr, _, _ in graphs + [("empty", g_empty, None, None)]:
                with torch.cuda.stream(stream):
                    torch.sum(flush, dim=0, out=sink)
                    torch.cuda._sleep(cyc)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    gr.replay()
                    e1.record()
                e1.synchronize()
                (te if name == "empty" else ts[name]).append(e0.elapsed_time(e1) * 1e3)
        ovh = statistics.median(te)
        ref = graphs[0][2].float()
        base = statistics.median(ts[graphs[0][0]])
        print(f"[ax3] sweep B{B}: replay overhead {ovh:.2f} us; raw median us (net = raw - overhead):", flush=True)
        for name, gr, out, v in graphs:
            med = statistics.median(ts[name])
            err = float((out.float() - ref).norm() / ref.norm())
            line = f"[ax3]   {name:18s} raw {med:6.2f} net {med - ovh:6.2f} (vs first {med - base:+.2f}) err vs first {err:.2e}"
            if v is not None:
                ns = v[0]
                prof = torch.zeros(8 * B * ns, dtype=torch.int64, device=device)
                g2, _ = capture(lambda v=v, prof=prof: _ax3_call(qkv, d, cs, v, prof=prof))
                reps = []
                for _ in range(8):
                    prof.zero_()
                    with torch.cuda.stream(stream):
                        torch.sum(flush, dim=0, out=sink)
                        torch.cuda._sleep(cyc)
                        g2.replay()
                    torch.cuda.synchronize(device)
                    pm = prof.view(-1, 8)[:, :7].double().cpu()
                    t0 = pm[:, 0].min()
                    reps.append(torch.where(pm > 0, (pm - t0) / 1000.0, torch.full_like(pm, float("nan"))))
                st = torch.stack(reps)
                med_m = st.nanmedian(dim=1).values.median(dim=0).values
                mx = torch.stack([torch.tensor([float(r[:, k][~torch.isnan(r[:, k])].max()) if (~torch.isnan(r[:, k])).any()
                                                else float("nan") for k in range(7)]) for r in reps]).median(dim=0).values
                line += " | marks " + " ".join(f"{k}:{med_m[k]:.2f}/{mx[k]:.2f}" for k in range(7))
                del g2, prof
            print(line, flush=True)
        del graphs, d, qkv
        torch.cuda.synchronize(device)
    del flush


'''


KS3PF_OLD1 = "  if constexpr (MG == 2) vsl_carrive_relaxed();\n  if (p.trig == 2) pdl_trigger();\n  KS3P(0);\n"
KS3PF_NEW1 = ("  if constexpr (MG == 2) vsl_carrive_relaxed();\n  if (p.trig == 2) pdl_trigger();\n  KS3P(0);\n"
              "  if constexpr ((MG & 4) != 0) {                         // AX3 rider: red / ticket lines toward L2 at entry\n"
              "    if (threadIdx.x < 34) {\n"
              "      const void* a = threadIdx.x < 32 ? (const void*)(p.part_o + (size_t)b * 1024 + 32 * threadIdx.x)\n"
              "                      : (threadIdx.x == 32 ? (const void*)(reinterpret_cast<float*>(p.part_ml) + (size_t)b * 16)\n"
              "                                           : (const void*)(p.counters + b));\n"
              "      asm volatile(\"prefetch.global.L2 [%0];\" ::\"l\"(a));\n"
              "    }\n"
              "  }\n")
KS3PF_OLD2 = "    if constexpr (MG == 1) {   // VSL red merge (see KS1)"
KS3PF_NEW2 = "    if constexpr ((MG & 3) == 1) {   // VSL red merge (see KS1; MG 5 = + the AX3 entry prefetch)"


def _lean_cuda(cu):
    """--lean (layer size budget): no comment text, no lab marks (the prof argument stays and is ignored)."""
    out = []
    for line in cu.split(NL):
        s = line.strip()
        if s.startswith("//") or s.startswith("AXP(") or s.startswith("#define AXP") or s == "#undef AXP":
            continue
        k = line.find("//")
        if k >= 0 and '"' not in line[k:]:
            line = line[:k].rstrip()
        out.append(line)
    return NL.join(out)


def _lean_py(py):
    out = []
    for line in py.split(NL):
        if line.strip().startswith("#"):
            continue
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return NL.join(out)


def _sub_once(src, old, new, what):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ax3_patches: anchor '{what}' found {n} times (need 1)")
    return src.replace(old, new)


def patch(src, insts, route, sweep, nonce, lean=False, ks3pf=None):
    cu_insts = " ".join(f"AX3_X({n}, {v})" for n, v in insts)
    # 1. C++ declarations
    src = _sub_once(src, 'std::vector<int64_t> ks3_info(int64_t NS);' + NL + '"""',
                    'std::vector<int64_t> ks3_info(int64_t NS);' + NL + CPP_DECL + '"""', "KS1_CPP_SRC end")
    # 2. CUDA: append after ks3_info's definition, before the end of KS1_CUDA_SRC
    anchor = 'std::vector<int64_t> ks3_info(int64_t NS) {'
    i = src.index(anchor)
    j = src.index(NL + '"""' + NL, i)
    if src.count(anchor) != 1:
        raise SystemExit("ax3_patches: ks3_info definition not unique")
    cu = CUDA_SRC.replace("@AX3INSTS@", cu_insts)
    src = src[:j + 1] + (_lean_cuda(cu) if lean else cu) + src[j + 1:]
    # 3. build: export the new functions
    src = _sub_once(src, 'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info"]',
                    'functions=["ks1_forward", "ks1_info", "ks3_forward", "ks3_info", "ax3_forward", "ax3_info"]', "_ks1_build functions")
    # 4. _ks1_wanted: an AX3 route makes the shape KS-routable (the same window / rope / page / scale / head guards follow)
    src = _sub_once(src, 'or _vsl_pick(cfg, state) is not None) and cfg["window"] == 2048',
                    'or _vsl_pick(cfg, state) is not None or _ax3_pick(cfg, state) is not None) and cfg["window"] == 2048',
                    "_ks1_wanted guard")
    # 5. _ks1_run: AX3 first
    src = _sub_once(src, 'def _ks1_run(qkv, data, state):' + NL + '    cfg = data["config"]' + NL,
                    'def _ks1_run(qkv, data, state):' + NL + '    cfg = data["config"]' + NL
                    + '    a3 = _ax3_pick(cfg, state) if (qkv.stride(0) % 8 == 0 and qkv.data_ptr() % 16 == 0) else None' + NL
                    + '    if a3 is not None:' + NL
                    + '        return _ax3_call(qkv, data, state["cos_sin"], a3)' + NL, "_ks1_run head")
    # 6. Python block before _ks1_run
    py = (PY_BLOCK.replace("@PYINSTS@", repr(tuple(tuple(x) for x in insts))).replace("@PYROUTE@", repr(route))
          .replace("@PYSWEEP@", "True" if sweep else "False"))
    if sweep:
        py += SWEEP_BLOCK
    if lean:
        py = _lean_py(py)
    src = _sub_once(src, NL + 'def _ks1_run(qkv, data, state):', NL + py + 'def _ks1_run(qkv, data, state):', "_ks1_run def")
    # 7. prepare
    src = _sub_once(src, '    _vsl_prepare(state, device, t0)' + NL,
                    '    _vsl_prepare(state, device, t0)' + NL + '    _ax3_prepare(state, device, t0)' + NL, "_ks1_prepare hook")
    # 7b. KS3 rider (MG 5 = MG 1 + entry prefetch): source, VSL instances, VSL route at d16
    if ks3pf is not None:
        src = _sub_once(src, KS3PF_OLD1, KS3PF_NEW1, 'ks3 entry')
        src = _sub_once(src, KS3PF_OLD2, KS3PF_NEW2, 'ks3 MG 1 merge')
        src = _sub_once(src, 'VSL_KS3_INSTS = ((16, 1),)', 'VSL_KS3_INSTS = ' + repr(tuple(ks3pf['insts'])), 'VSL_KS3_INSTS')
        if ks3pf.get('route16'):
            src = _sub_once(src, "(16, 1): ('ks3', 16, 1)}", "(16, 1): " + repr(tuple(ks3pf['route16'])) + "}", 'VSL_ROUTE d16')
    # 8. nonce (line 2)
    lines = src.split(NL)
    if len(lines) > 1 and lines[1].startswith("# nonce"):
        lines[1] = "# nonce " + nonce
    else:
        lines.insert(1, "# nonce " + nonce)
    return NL.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--insts", default=repr(DEF_INSTS))
    ap.add_argument("--route", default=repr(DEF_ROUTE))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--nonce", default=None)
    ap.add_argument("--ks3pf", default=None, help="KS3 rider: {'insts': ((16, 1), (16, 5)), 'route16': ('ks3', 16, 5)}")
    ap.add_argument("--lean", action="store_true", help="layer size budget: no comments / lab marks in the added code")
    a = ap.parse_args()
    insts = tuple(tuple(x) for x in ast.literal_eval(a.insts))
    route = {tuple(k): tuple(v) for k, v in ast.literal_eval(a.route).items()}
    for v in route.values():
        if v not in insts:
            raise SystemExit(f"ax3_patches: route {v} not in insts {insts}")
    base = (ROOT / a.base) if not pathlib.Path(a.base).is_absolute() else pathlib.Path(a.base)
    out = (ROOT / a.out) if not pathlib.Path(a.out).is_absolute() else pathlib.Path(a.out)
    src = base.read_text(encoding="utf-8")
    nonce = a.nonce or (f"{out.stem} ({base.stem} + AX3 4-warp pipelined sliding decode B16/32, tools/ax3_patches.py; "
                        f"route {route}, insts {insts}{', sweep' if a.sweep else ''})")
    res = patch(src, insts, route, a.sweep, nonce, a.lean, ast.literal_eval(a.ks3pf) if a.ks3pf else None)
    out.write_bytes(res.encode("utf-8"))
    compile(res, str(out), "exec")
    print(f"ax3_patches: {base.name} -> {out} ({len(res)} B, +{len(res) - len(src)} B); route {route}; insts {insts}; "
          f"sweep {a.sweep}")


if __name__ == "__main__":
    main()

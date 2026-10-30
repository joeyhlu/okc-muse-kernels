"""ax3q_src.py: AX3Q = one-launch VERIFY SLIDING attention (q_len 8, B >= 2, page 1), replacing the FA4 3-launch chain
(prologue + FA4 core + combine_gate) at v16s / v32s. Inserted by tools/ax3_patches.py --q8.

Structure (vjeux's attn_kernel<8, false, 3> read for ideas only; the q_len 8 core is our own FX-S / AX1 wgmma consumer,
tools/ax1_src.py fb1_l8, adapted): grid (NS, B), 256 threads = 2 warpgroups (WG = KV head), each CTA one split of
request b's key domain [max(0, seq - 2047), seq + 8) in whole 64-key tiles; seq_lens -> page entries (4-B cp.async,
2 ST - 2 tiles ahead) -> K / V 64-key tiles in a ST-stage cp.async ring (ST - 1 tiles in flight), issued before the PDL
wait; after it: q RMSNorm + RoPE -> e4m3 (wgmma A tile), the 8 new K (RMSNorm + RoPE, KS1's rounding) / V rows -> e4m3
(cache write by the CTA owning the position, patched into the tiles); per tile QK = 2 x m64n64k32 e4m3 SS, V^T built
by the WG (key-permuted, + 8 rows of ones for the row sums), PV = 2 x m64n72k32 RS, software-pipelined (QK / V^T of
tile i + 1 issued while PV(i) runs), running max with offset 8 (FA4 PV8 numerics: P = e4m3(2^(s - max + 8))); causal
and per-token window masks (token j sees [seq + j - 2047, seq + j]) only on edge tiles. Merge: o 2^off, l 2^off by
red.v4 into request b's zeroed workspace, acq_rel ticket, the LAST CTA finalizes the request's 8 x 16 rows
(norm, RMSNorm, * sigmoid(gate)) and re-zeroes (no spin, no residency requirement).
"""

AX3Q_CPP_SRC = r"""
#include <torch/extension.h>
void ax3q_forward(torch::Tensor qkv, torch::Tensor seqlens, torch::Tensor cloc, torch::Tensor pos, torch::Tensor pt,
                  torch::Tensor kc, torch::Tensor vc, torch::Tensor cos_sin, torch::Tensor wo, torch::Tensor wl,
                  torch::Tensor cnt, torch::Tensor out, int64_t ns, int64_t st, double sc2, double qks, double eps,
                  int64_t rope, int64_t trig, int64_t pdl, torch::Tensor prof);
std::vector<int64_t> ax3q_info(int64_t st);
"""

AX3Q_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <vector>

namespace ax3q {

constexpr int TILE = 64, ROWB = 128, HQ = 16, HKV = 2, WIN = 2048, L = 8;
constexpr unsigned FULL = 0xffffffffu;
constexpr int TB = TILE * ROWB;
constexpr int VTB = 72 * 64;
constexpr int QB8 = 64 * 64;
constexpr int NTH = 256;
constexpr uint64_t HI64 = ((uint64_t)(512 >> 4) | (2ull << 30)) << 32;    // SBO 512, 64B swizzle
constexpr uint64_t HI128 = ((uint64_t)(1024 >> 4) | (1ull << 30)) << 32;  // SBO 1024, 128B swizzle

struct Params {
  const __nv_bfloat16* qkv;   // [B L, sq]: q (16 x 64) | k (2 x 64) | v (2 x 64) | gate (16 x 64)
  const int* seqlens;         // [B] cached length + L
  const int64_t* cloc;        // [B L] cache rows of the new tokens
  const int64_t* pos;         // [B L] positions (RoPE)
  const int* pt;              // [B, ptc] page table (page size 1)
  uint8_t* kc;                // e4m3 rows of HKV * 64 bytes
  uint8_t* vc;
  const float* cs;            // [P, 64] cos | sin
  float* wo;                  // [B][128][64] fp32 (zero between calls)
  float* wl;                  // [B][128]
  int* cnt;                   // [B] arrival tickets (zero between calls)
  __nv_bfloat16* out;         // [B L, 1024]
  long long* prof;            // [B NS][8] %globaltimer marks or nullptr
  int sq, ptc, rope, trig;
  float sc2, qks, eps;
};

template <int ST> struct Lay {
  static constexpr int OFF_V = ST * TB;
  static constexpr int OFF_VT = 2 * ST * TB;               // [head][buffer] V^T
  static constexpr int OFF_Q = OFF_VT + 4 * VTB;           // [head] e4m3 Q
  static constexpr int OFF_IDX = OFF_Q + 2 * QB8;
  static constexpr int OFF_KN = OFF_IDX + 2 * ST * TILE * 4;   // new keys [8][128], new values [8][128]
  static constexpr int OFF_MISC = OFF_KN + 2 * 8 * ROWB;
  static constexpr int SMEM = OFF_MISC + 64 + 1024;        // + 1 KB alignment slack (128B swizzle atoms)
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(uint32_t d, const void* s, int n) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(s), "r"(n) : "memory");
}
__device__ __forceinline__ void cpa4(uint32_t d, const void* s) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(s) : "memory");
}
__device__ __forceinline__ void cpc() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpw() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float ninf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a) : "memory");
}
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;\n" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ int atom_acqrel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ long long gtime() { long long v; asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(v)); return v; }
#define AQP(k) do { if (p.prof != nullptr && threadIdx.x == 0) p.prof[(size_t)cta * 8 + (k)] = gtime(); } while (0)
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_go() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void fpa() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }
__device__ __forceinline__ void wgf() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgc() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgw0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void barn(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void sts32(uint32_t a, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory"); }
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {   // 4 x e4m3 (satfinite), a in the low byte
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}\n"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
#define AQ_RF(x) asm volatile("" : "+f"(x))
#define AQ_RR(x) asm volatile("" : "+r"(x))
#define AQ_D32(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
#define AQ_R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"
#define AQ_R36 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35}"
__device__ __forceinline__ void wg_qk(float (&d)[32], uint64_t da, uint64_t db, int scale_d) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " AQ_R32 ", %32, %33, p, 1, 1;\n}\n"
               : AQ_D32(d) : "l"(da), "l"(db), "r"(scale_d));
}
__device__ __forceinline__ void wg_pv(float (&d)[36], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n72k32.f32.e4m3.e4m3 " AQ_R36 ", {%36,%37,%38,%39}, %40, p, 1, 1;\n}\n"
               : AQ_D32(d), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}
// KS1's q / k prep rounding: bf16(x rs), RoPE on the bf16 pair with bf16 results (lane = dims 2 lane, 2 lane + 1)
__device__ __forceinline__ float2 norm_rope(uint32_t raw, float2 c, int rope, float eps) {
  const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw));
  float ss = x.x * x.x + x.y * x.y;            // (KS3's expression: the same FMA contraction -> the same cache bytes)
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) ss += __shfl_xor_sync(FULL, ss, d);
  const float rs = rsqrtf(ss * (1.f / 64.f) + eps);
  const float e = bfr(x.x * rs), o = bfr(x.y * rs);
  if (!rope) return make_float2(e, o);
  return make_float2(bfr(__fmaf_rn(e, c.x, -__fmul_rn(o, c.y))), bfr(__fmaf_rn(e, c.y, __fmul_rn(o, c.x))));
}

template <int ST>
__global__ void __launch_bounds__(NTH, 2) ax3q_kernel(const __grid_constant__ Params p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  using Y = Lay<ST>;
  extern __shared__ uint8_t sm_raw[];
  uint8_t* sm = sm_raw + ((1024u - (sa(sm_raw) & 1023u)) & 1023u);
  uint8_t* Ks = sm;
  int* idx = reinterpret_cast<int*>(sm + Y::OFF_IDX);
  uint8_t* kn = sm + Y::OFF_KN;
  uint8_t* vn = kn + 8 * ROWB;
  volatile int* misc = reinterpret_cast<volatile int*>(sm + Y::OFF_MISC);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int s = blockIdx.x, b = blockIdx.y, NS = gridDim.x, cta = b * NS + s;
  if (p.trig == 2) pdl_go();
  AQP(0);
  const int seq = max(__ldg(p.seqlens + b) - L, 0);          // cached keys; positions seq .. seq + 7 are the block's
  const int lo = max(0, seq - (WIN - 1));                     // key domain [lo, seq + 8): every token's window
  const int nt = (seq + L - lo + TILE - 1) / TILE;
  const int tps = (nt + NS - 1) / NS;                          // whole tiles per split
  const int start = lo + s * tps * TILE, end = min(seq + L, start + tps * TILE);
  const int ntiles = start < end ? (end - start + TILE - 1) / TILE : 0;
  const int cend = min(end, seq);                              // cached keys of this split
  const int* ptr = p.pt + (size_t)b * p.ptc;

  auto issue_idx = [&](int y) {
    if (tid < TILE && y < ntiles) {
      const int ps = start + y * TILE + tid;
      if (ps < cend) cpa4(sa(idx + (y % (2 * ST)) * TILE + tid), ptr + ps);
    }
  };
#pragma unroll
  for (int y = 0; y < ST - 1; ++y) issue_idx(y);
  cpc();
#pragma unroll
  for (int y = ST - 1; y < 2 * ST - 2; ++y) issue_idx(y);
  cpc();
  const int cc = tid & 7, crw = tid >> 3;      // copy role: 16-B chunk cc of rows crw, crw + 32
  auto issue_kv = [&](int x, bool with_new) {
    const int st = x % ST;
    const int* ix = idx + (x % (2 * ST)) * TILE;
    uint8_t* kd = Ks + st * TB;
    uint8_t* vd = Ks + Y::OFF_V + st * TB;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int r = crw + 32 * k, ps = start + x * TILE + r;
      const int o = r * ROWB + ((cc ^ (r & 7)) << 4);
      if (ps < cend) {
        const size_t row = (size_t)(uint32_t)ix[r] * ROWB + cc * 16;
        cpa16(sa(kd + o), p.kc + row, 16);
        cpa16(sa(vd + o), p.vc + row, 16);
      } else if (ps >= end) {
        cpa16(sa(kd + o), p.kc, 0);
        cpa16(sa(vd + o), p.vc, 0);
      } else if (with_new) {
        *reinterpret_cast<uint4*>(kd + o) = *reinterpret_cast<const uint4*>(kn + (ps - seq) * ROWB + cc * 16);
        *reinterpret_cast<uint4*>(vd + o) = *reinterpret_cast<const uint4*>(vn + (ps - seq) * ROWB + cc * 16);
      }
    }
  };
  cpw<1>();
  __syncthreads();                             // page entries of the first ST - 1 tiles
#pragma unroll
  for (int x = 0; x < ST - 1; ++x) {
    if (x < ntiles) issue_kv(x, false);
    cpc();
  }
  if (tid < 128)   // rows 64..71 of the four V^T buffers: e4m3 ones (0x38; every byte equal, no swizzle)
    *reinterpret_cast<uint4*>(sm + Y::OFF_VT + (tid >> 5) * VTB + 4096 + (tid & 31) * 16) =
        make_uint4(0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u);
  AQP(1);
  pdl_wait();
  if (p.trig == 1) pdl_go();
  const bool mine = ntiles > 0 && seq + L > start && seq < end;   // this CTA holds some of the block's new keys
  if (ntiles > 0) {   // warp = token jt: q RMSNorm + RoPE -> e4m3; new K (RMSNorm + RoPE) / V -> e4m3
    const int jt = warp;
    const __nv_bfloat16* xr = p.qkv + ((size_t)b * L + jt) * p.sq;
    uint32_t qr[HQ];
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) qr[hd] = __ldcg(reinterpret_cast<const unsigned int*>(xr + hd * 64) + lane);
    uint32_t nr[4] = {0u, 0u, 0u, 0u};
    if (mine) {
#pragma unroll
      for (int k = 0; k < 4; ++k) nr[k] = __ldcg(reinterpret_cast<const unsigned int*>(xr + (HQ + k) * 64) + lane);
    }
    float2 c = make_float2(1.f, 0.f);
    if (p.rope) {
      const long long ps = __ldg(p.pos + (size_t)b * L + jt);
      c = make_float2(__ldg(p.cs + ps * 64 + lane), __ldg(p.cs + ps * 64 + 32 + lane));
    }
    const int ch = lane >> 3, bo = (2 * lane) & 15;
#pragma unroll
    for (int hd = 0; hd < HQ; ++hd) {
      const float2 x = norm_rope(qr[hd], c, p.rope, p.eps);
      const uint16_t q8 = (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
      const int h = hd >> 3, r = 8 * jt + (hd & 7);
      *reinterpret_cast<uint16_t*>(sm + Y::OFF_Q + h * QB8 + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4) + bo) = q8;
    }
    if (mine) {
      const long long loc = __ldg(p.cloc + (size_t)b * L + jt);
      const bool wr = seq + jt >= start && seq + jt < end;   // each new row is written by exactly one CTA
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 x = k < 2 ? norm_rope(nr[k], c, p.rope, p.eps)
                         : __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&nr[k]));
        x.x = fminf(fmaxf(x.x, -448.f), 448.f);
        x.y = fminf(fmaxf(x.y, -448.f), 448.f);
        const uint16_t e8 = (uint16_t)__nv_cvt_float2_to_fp8x2(x, __NV_SATFINITE, __NV_E4M3);
        const int head = k & 1;
        *reinterpret_cast<uint16_t*>((k < 2 ? kn : vn) + jt * ROWB + head * 64 + 2 * lane) = e8;
        if (wr) *reinterpret_cast<uint16_t*>((k < 2 ? p.kc : p.vc) + (size_t)loc * ROWB + head * 64 + 2 * lane) = e8;
      }
    }
  }
  __syncthreads();
  if (mine && tid < 128) {   // new rows into the tiles issued above (later tiles take them in issue_kv)
    const int kind = tid >> 6, jn = (tid >> 3) & 7, c = tid & 7, ps = seq + jn;
    if (ps >= start && ps < end) {
      const int x = (ps - start) / TILE, r = (ps - start) % TILE;
      if (x < ST - 1)
        *reinterpret_cast<uint4*>(Ks + (kind ? Y::OFF_V : 0) + x * TB + r * ROWB + ((c ^ (r & 7)) << 4)) =
            *reinterpret_cast<const uint4*>((kind ? vn : kn) + jn * ROWB + c * 16);
    }
  }
  fpa();

  const int h = warp >> 2, w = warp & 3, g = lane >> 2, tig = lane & 3;
  const float ssc = p.sc2 * p.qks;             // raw score -> log2 units
  float acc[36];
#pragma unroll
  for (int e = 0; e < 36; ++e) { acc[e] = 0.f; AQ_RF(acc[e]); }
  float cr0 = ninf(), cr1 = ninf();
  const uint32_t qlo = __shfl_sync(FULL, ((sa(sm) + Y::OFF_Q + h * QB8) >> 4) | 0x10000u, 0);
  const uint32_t klo = __shfl_sync(FULL, ((sa(sm) + h * 64) >> 4) | 0x10000u, 0);
  const uint32_t vtlo = __shfl_sync(FULL, ((sa(sm) + Y::OFF_VT + h * 2 * VTB) >> 4) | 0x10000u, 0);
  const int vkey = 16 * w + 8 * ((lane >> 3) & 1) + (lane & 7);
  const uint32_t vld0 = sa(sm) + Y::OFF_V + vkey * ROWB + (((4 * h + (lane >> 4)) ^ (lane & 7)) << 4);
  const uint32_t vst = sa(sm) + Y::OFF_VT + h * 2 * VTB + 128 * g + ((w ^ (g & 3)) << 4) + 4 * tig;
  const uint32_t vsel_a = (g >= 4) ? 0x7531u : 0x6420u, vsel_b = (g >= 4) ? 0x6420u : 0x7531u;
  const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);
  const int hi0 = seq + 2 * w, hi1 = hi0 + 1;          // row g (token 2 w) / g + 8 (token 2 w + 1): keys <= hi
  const int lb0 = hi0 - (WIN - 1), lb1 = hi1 - (WIN - 1);  // ... and >= lb (the window)

  auto acquire = [&](int j) {   // tile j landed (all threads); the next copies go into the stage tile j - 1 freed
    cpw<ST - 2>();
    fpa();
    __syncthreads();
    if (j + ST - 1 < ntiles) issue_kv(j + ST - 1, true);
    issue_idx(j + 2 * ST - 2);
    cpc();
  };
  float sc[32];
  auto qk_issue = [&](int j) {
    const uint32_t kl = klo + (uint32_t)((j % ST) * (TB >> 4));
    wgf();
    wg_qk(sc, HI64 | qlo, HI128 | kl, 0);
    wg_qk(sc, HI64 | (qlo + 2u), HI128 | (kl + 2u), 1);
    wgc();
  };
  auto vt_build = [&](int j) {   // tile j's V (head h) -> key-permuted V^T, buffer j & 1
    const uint32_t vsrc = vld0 + (uint32_t)((j % ST) * TB);
    const uint32_t da = vst_a + (uint32_t)((j & 1) * VTB), db = vst_b + (uint32_t)((j & 1) * VTB);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      uint32_t r[4];
      ldsm4t(r, u ? (vsrc ^ 32u) : vsrc);
      sts32(da + 1024 * (2 * u), __byte_perm(r[0], r[1], vsel_a));
      sts32(db + 1024 * (2 * u), __byte_perm(r[0], r[1], vsel_b));
      sts32(da + 1024 * (2 * u + 1), __byte_perm(r[2], r[3], vsel_a));
      sts32(db + 1024 * (2 * u + 1), __byte_perm(r[2], r[3], vsel_b));
    }
  };
  uint32_t pa[2][4] = {{0u, 0u, 0u, 0u}, {0u, 0u, 0u, 0u}};
  AQP(2);
  if (ntiles > 0) {
    acquire(0);
    qk_issue(0);
    vt_build(0);
  }
#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    wgw0();   // QK(i) and (committed before it) PV(i - 1) done
#pragma unroll
    for (int e = 0; e < 32; ++e) AQ_RF(sc[e]);
#pragma unroll
    for (int e = 0; e < 36; ++e) AQ_RF(acc[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) AQ_RR(pa[e >> 2][e & 3]);
    const int t0 = start + i * TILE;
    if (t0 + TILE - 1 > hi0 || t0 < lb1) {   // edge tiles: causal end, window start, past the range
#pragma unroll
      for (int nt8 = 0; nt8 < 8; ++nt8)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int ps = t0 + 8 * nt8 + 2 * tig + e;
          if (ps > hi0 || ps < lb0) sc[4 * nt8 + e] = ninf();
          if (ps > hi1 || ps < lb1) sc[4 * nt8 + 2 + e] = ninf();
        }
    }
    float m0 = ninf(), m1 = ninf();
#pragma unroll
    for (int nt8 = 0; nt8 < 8; ++nt8) {
      m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
      m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
    }
    m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2)) * ssc;
    m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2)) * ssc;
    const float n0 = fmaxf(cr0, m0 - 8.f), n1 = fmaxf(cr1, m1 - 8.f);
    const float a0 = (n0 == ninf()) ? 1.f : ex2(cr0 - n0), a1 = (n1 == ninf()) ? 1.f : ex2(cr1 - n1);
#pragma unroll
    for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
    cr0 = n0; cr1 = n1;
    const float u0 = (n0 == ninf()) ? 0.f : n0, u1 = (n1 == ninf()) ? 0.f : n1;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {   // P = e4m3(2^(s ssc - u)) of keys 32 kk .. +31, in the PV A-fragment (key-permuted) order
      const float* sv = sc + 16 * kk;
      pa[kk][0] = pk4(ex2(fmaf(sv[0], ssc, -u0)), ex2(fmaf(sv[1], ssc, -u0)), ex2(fmaf(sv[4], ssc, -u0)), ex2(fmaf(sv[5], ssc, -u0)));
      pa[kk][1] = pk4(ex2(fmaf(sv[2], ssc, -u1)), ex2(fmaf(sv[3], ssc, -u1)), ex2(fmaf(sv[6], ssc, -u1)), ex2(fmaf(sv[7], ssc, -u1)));
      pa[kk][2] = pk4(ex2(fmaf(sv[8], ssc, -u0)), ex2(fmaf(sv[9], ssc, -u0)), ex2(fmaf(sv[12], ssc, -u0)), ex2(fmaf(sv[13], ssc, -u0)));
      pa[kk][3] = pk4(ex2(fmaf(sv[10], ssc, -u1)), ex2(fmaf(sv[11], ssc, -u1)), ex2(fmaf(sv[14], ssc, -u1)), ex2(fmaf(sv[15], ssc, -u1)));
    }
#pragma unroll
    for (int e = 0; e < 36; ++e) AQ_RF(acc[e]);
#pragma unroll
    for (int e = 0; e < 8; ++e) AQ_RR(pa[e >> 2][e & 3]);
    fpa();                    // V^T(i) (st.shared) -> async proxy
    barn(1 + h, 128);         // all 4 warps' V^T(i) rows written
    const uint32_t vl = vtlo + (uint32_t)((i & 1) * (VTB >> 4));
    wgf();
    wg_pv(acc, pa[0], HI64 | vl);
    wg_pv(acc, pa[1], HI64 | (vl + 2u));
    wgc();
    if (i + 1 < ntiles) {     // meanwhile: the next tile's copies, QK and V^T (buffer (i + 1) & 1: PV(i - 1) done above)
      acquire(i + 1);
      qk_issue(i + 1);
      vt_build(i + 1);
    }
  }
  wgw0();
#pragma unroll
  for (int e = 0; e < 36; ++e) AQ_RF(acc[e]);
#pragma unroll
  for (int e = 0; e < 8; ++e) AQ_RR(pa[e >> 2][e & 3]);
  AQP(3);
  if (p.trig == 0) pdl_go();
  float* wob = p.wo + (size_t)b * 128 * 64;
  float* wlb = p.wl + (size_t)b * 128;
  if (ntiles > 0) {   // o 2^off, l 2^off (lanes tig, tig ^ 1 swap so each lane adds 4 consecutive dims)
    const int R0 = (2 * w) * 16 + 8 * h + g, R1 = R0 + 16;
    const float f0 = (cr0 == ninf()) ? 0.f : ex2(fminf(cr0, 126.f)), f1 = (cr1 == ninf()) ? 0.f : ex2(fminf(cr1, 126.f));
    float* w0 = wob + (size_t)R0 * 64;
    float* w1 = wob + (size_t)R1 * 64;
    const bool odd = tig & 1;
#pragma unroll
    for (int t = 0; t < 8; t += 2) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const float f = r ? f1 : f0;
        const float x0 = acc[4 * t + 2 * r] * f, x1 = acc[4 * t + 2 * r + 1] * f;
        const float y0 = acc[4 * (t + 1) + 2 * r] * f, y1 = acc[4 * (t + 1) + 2 * r + 1] * f;
        const float s0 = odd ? x0 : y0, s1 = odd ? x1 : y1;
        const float q0 = __shfl_xor_sync(FULL, s0, 1), q1 = __shfl_xor_sync(FULL, s1, 1);
        red4((r ? w1 : w0) + 8 * (t + (odd ? 1 : 0)) + 4 * (tig >> 1), odd ? q0 : x0, odd ? q1 : x1, odd ? y0 : q0, odd ? y1 : q1);
      }
    }
    if (tig == 0) {
      red1(wlb + R0, acc[32] * f0);
      red1(wlb + R1, acc[34] * f1);
    }
  }
  AQP(4);
  __syncthreads();                             // every thread's reds before the release (bar.sync cumulativity)
  if (tid == 0) misc[1] = (atom_acqrel(p.cnt + b, 1) == NS - 1) ? 1 : 0;
  __syncthreads();
  AQP(5);
  if (!misc[1]) return;
  // last CTA of request b: 128 rows (token j = R >> 4, head R & 15) x 64 dims, 8 dims per thread, 4 passes
  const int part = tid & 7;
  uint4 graw[4];
  float4 xa[4], xb[4];
  float lc[4];
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int R = u * 32 + (tid >> 3), j = R >> 4, hh = R & 15;
    graw[u] = __ldcg(reinterpret_cast<const uint4*>(p.qkv + ((size_t)b * L + j) * p.sq + (HQ + 2 * HKV) * 64 + hh * 64 + 8 * part));
    xa[u] = __ldcg(reinterpret_cast<const float4*>(wob + (size_t)R * 64 + 8 * part));
    xb[u] = __ldcg(reinterpret_cast<const float4*>(wob + (size_t)R * 64 + 8 * part + 4));
    lc[u] = (part == 0) ? __ldcg(wlb + R) : 0.f;
  }
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int R = u * 32 + (tid >> 3), j = R >> 4, hh = R & 15;
    __stcg(reinterpret_cast<float4*>(wob + (size_t)R * 64 + 8 * part), make_float4(0.f, 0.f, 0.f, 0.f));
    __stcg(reinterpret_cast<float4*>(wob + (size_t)R * 64 + 8 * part + 4), make_float4(0.f, 0.f, 0.f, 0.f));
    if (part == 0) __stcg(wlb + R, 0.f);
    const float ls = __shfl_sync(FULL, lc[u], lane & ~7);
    const float inv = ls > 0.f ? 1.f / ls : 0.f;
    float y[8] = {xa[u].x, xa[u].y, xa[u].z, xa[u].w, xb[u].x, xb[u].y, xb[u].z, xb[u].w};
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { y[e] = bfr(y[e] * inv); ss = fmaf(y[e], y[e], ss); }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    const float rn = rsqrtf(ss * (1.f / 64.f) + p.eps);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&graw[u]);
    uint4 ov;
    uint32_t* ow = reinterpret_cast<uint32_t*>(&ov);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 gt = __bfloat1622float2(g2[e]);
      const __nv_bfloat162 o2 = __floats2bfloat162_rn(y[2 * e] * rn / (1.f + __expf(-gt.x)), y[2 * e + 1] * rn / (1.f + __expf(-gt.y)));
      ow[e] = *reinterpret_cast<const uint32_t*>(&o2);
    }
    *reinterpret_cast<uint4*>(p.out + ((size_t)b * L + j) * (HQ * 64) + hh * 64 + 8 * part) = ov;
  }
  if (tid == 0) p.cnt[b] = 0;
  AQP(6);
#endif
}

template <int ST>
static void launch(const Params& P, int ns, int B, bool pdl, cudaStream_t stream) {
  constexpr int SM = Lay<ST>::SMEM;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(ax3q_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)ns, (unsigned)B, 1);
  cfg.blockDim = dim3(NTH, 1, 1);
  cfg.dynamicSmemBytes = SM;
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, ax3q_kernel<ST>, P));
}

template <int ST>
static std::vector<int64_t> info() {
  constexpr int SM = Lay<ST>::SMEM;
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, ax3q_kernel<ST>));
  C10_CUDA_CHECK(cudaFuncSetAttribute(ax3q_kernel<ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, ax3q_kernel<ST>, NTH, SM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)SM, (int64_t)occ};
}

}  // namespace ax3q

void ax3q_forward(torch::Tensor qkv, torch::Tensor seqlens, torch::Tensor cloc, torch::Tensor pos, torch::Tensor pt,
                  torch::Tensor kc, torch::Tensor vc, torch::Tensor cos_sin, torch::Tensor wo, torch::Tensor wl,
                  torch::Tensor cnt, torch::Tensor out, int64_t ns, int64_t st, double sc2, double qks, double eps,
                  int64_t rope, int64_t trig, int64_t pdl, torch::Tensor prof) {
  const int64_t B = seqlens.numel();
  TORCH_CHECK(qkv.size(0) == B * 8 && qkv.stride(1) == 1 && qkv.size(1) >= 36 * 64 && qkv.stride(0) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "ax3q: qkv");
  TORCH_CHECK(cloc.numel() == B * 8 && pos.numel() == B * 8 && pt.size(0) == B && pt.is_contiguous(), "ax3q: metadata");
  TORCH_CHECK(wo.numel() >= B * 128 * 64 && wl.numel() >= B * 128 && cnt.numel() >= B, "ax3q: workspace");
  TORCH_CHECK(ns >= 1 && ns <= 64, "ax3q: splits");
  TORCH_CHECK(prof.numel() == 0 || prof.numel() >= B * ns * 8, "ax3q: prof");
  ax3q::Params P;
  P.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  P.seqlens = seqlens.data_ptr<int>();
  P.cloc = cloc.data_ptr<int64_t>();
  P.pos = pos.data_ptr<int64_t>();
  P.pt = pt.data_ptr<int>();
  P.kc = reinterpret_cast<uint8_t*>(kc.data_ptr());
  P.vc = reinterpret_cast<uint8_t*>(vc.data_ptr());
  P.cs = cos_sin.data_ptr<float>();
  P.wo = wo.data_ptr<float>();
  P.wl = wl.data_ptr<float>();
  P.cnt = cnt.data_ptr<int>();
  P.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  P.prof = prof.numel() ? reinterpret_cast<long long*>(prof.data_ptr<int64_t>()) : nullptr;
  P.sq = (int)qkv.stride(0);
  P.ptc = (int)pt.size(1);
  P.rope = (int)rope;
  P.trig = (int)trig;
  P.sc2 = (float)sc2;
  P.qks = (float)qks;
  P.eps = (float)eps;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (st == 3) ax3q::launch<3>(P, (int)ns, (int)B, pdl != 0, stream);
  else if (st == 4) ax3q::launch<4>(P, (int)ns, (int)B, pdl != 0, stream);
  else TORCH_CHECK(false, "ax3q: stages 3 / 4");
}

std::vector<int64_t> ax3q_info(int64_t st) { return st == 3 ? ax3q::info<3>() : ax3q::info<4>(); }
"""

AX3Q_PY = r'''AX3Q_ENABLE = True

AX3Q_ROUTE = @Q8ROUTE@

AX3Q_ARMS = @Q8ARMS@

AX3Q_TRIG = @Q8TRIG@

AX3Q_SELF_TEST = ((16, 3000, 931, False), (32, 2600, 932, True), (3, 2100, 933, False), (2, 2047, 934, False),
                  (4, 100, 935, False), (2, 5, 936, False), (16, 80000, 937, False))

AX3Q_PREPARE_BY_S = 240.0

AX3Q_SWEEP = @Q8SWEEP@

_AX3Q = {"mod": None, "ws": {}, "seen": set()}


def _ax3q_build():
    _pt1_wait("ax3q")
    if _AX3Q["mod"] is None:
        from torch.utils.cpp_extension import load_inline
        tag = hashlib.sha1((AX3Q_CPP_SRC + AX3Q_CUDA_SRC).encode()).hexdigest()[:10]
        _AX3Q["mod"] = load_inline(name=f"ax3q_{tag}", cpp_sources=AX3Q_CPP_SRC, cuda_sources=AX3Q_CUDA_SRC,
                                   functions=["ax3q_forward", "ax3q_info"],
                                   extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                                   verbose=False)
    return _AX3Q["mod"]


def _ax3q_ws(dev):
    key = str(dev)
    if key not in _AX3Q["ws"]:
        _AX3Q["ws"][key] = (torch.zeros(64 * 128 * 64, dtype=torch.float32, device=dev),
                            torch.zeros(64 * 128, dtype=torch.float32, device=dev),
                            torch.zeros(64, dtype=torch.int32, device=dev))
    return _AX3Q["ws"][key]


def _ax3q_call(qkv, data, cfg, cos_sin, v, prof=None, trig=None):
    B = cfg["batch"]
    ns, st = v
    dev = qkv.device
    wo, wl, cnt = _ax3q_ws(dev)
    out = torch.empty((B * 8, 16 * 64), dtype=torch.bfloat16, device=dev)
    _AX3Q["mod"].ax3q_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["positions"], data["page_table"],
                              data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8), cos_sin, wo, wl, cnt,
                              out, int(ns), int(st), float(cfg["sm_scale"] * 1.4426950408889634),
                              float(cfg["qk_scale_factor"]), 1e-5, 1 if cfg["rope_theta"] else 0,
                              int(AX3Q_TRIG if trig is None else trig), 1,
                              prof if prof is not None else torch.empty(0, dtype=torch.int64, device=dev))
    if v not in _AX3Q["seen"]:
        _AX3Q["seen"].add(v)
        print(f"[ax3q] first call {v} B={B}", flush=True)
    return out


def _ax3q_wanted(qkv, data, cfg, state):
    v = AX3Q_ROUTE.get(cfg["batch"]) if (AX3Q_ENABLE and cfg["q_len"] == 8) else None
    if v is None or tuple(v) not in (state.get("ax3q") or ()):
        return False
    pt = data["page_table"]
    cs = state.get("cos_sin")
    return (cfg["window"] == 2048 and cfg["rope_theta"] == ROPE_THETA and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0
            and cfg["head_dim"] == 64 and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2 and cfg["batch"] <= 64
            and cs is not None and cs.dtype == torch.float32 and cs.is_contiguous()
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.shape[0] == cfg["batch"] * 8
            and qkv.stride(0) % 8 == 0 and qkv.data_ptr() % 16 == 0
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and pt.dtype == torch.int32 and pt.dim() == 2 and pt.is_contiguous() and pt.shape[0] == cfg["batch"]
            and data["cache_seqlens"].dtype == torch.int32 and data["cache_seqlens"].is_contiguous()
            and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous()
            and data["positions"].dtype == torch.int64 and data["positions"].is_contiguous())


def _ax3q_try(qkv, data, cfg, state):
    try:
        return _ax3q_call(qkv, data, cfg, state["cos_sin"], tuple(AX3Q_ROUTE[cfg["batch"]]))
    except Exception as ex:  # noqa: BLE001
        print(f"[ax3q] call failed, route off: {repr(ex)[-400:]}", flush=True)
        state["ax3q"] = set()
        return None


def _ax3q_self_test(cos_sin, dev):
    wo, wl, cnt = _ax3q_ws(dev)
    ok = set()
    variants = sorted({tuple(v) for v in AX3Q_ROUTE.values()} | {tuple(v) for vs in AX3Q_ARMS.values() for v in vs})
    for v in variants:
        good, worst = True, 0.0
        try:
            for B, ctx, seed, row0 in AX3Q_SELF_TEST:
                ref = _k1fa4_reference(_k1fa4_case(B, ctx, 8, 2048, seed, row0, dev)).view(B * 8, -1)
                d1 = _k1fa4_case(B, ctx, 8, 2048, seed, row0, dev)
                o1 = _k1fa4_r1_path(d1).float().view(B * 8, -1)
                torch.cuda.synchronize(dev)
                e1 = float((o1 - ref).norm() / ref.norm())
                rows = d1["cache_loc"]
                k1b = d1["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                v1b = d1["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                dk = _k1fa4_case(B, ctx, 8, 2048, seed, row0, dev)
                qkv = dk["qkv"].view(B * 8, -1)
                o_a = _ax3q_call(qkv, dk, dk["config"], cos_sin, v).clone()
                o_b = _ax3q_call(qkv, dk, dk["config"], cos_sin, v).clone()
                torch.cuda.synchronize(dev)
                ek = float((o_a.float() - ref).norm() / ref.norm())
                kb = dk["k_cache"].view(-1, 128).view(torch.uint8)[rows]
                vb = dk["v_cache"].view(-1, 128).view(torch.uint8)[rows]
                nbad = int((kb != k1b).sum()) + int((vb != v1b).sum())
                clean = int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0 and float(wl.abs().sum()) == 0.0
                rep = float((o_a.float() - o_b.float()).abs().max()) <= 0.02 * float(o_a.float().abs().max()) + 1e-3
                c_ok = bool(torch.isfinite(o_a.float()).all()) and ek <= 1.5 * e1 + 5e-3 and nbad == 0 and clean and rep
                worst = max(worst, ek)
                print(f"[ax3q] self-test {v} B{B} ctx{ctx}: {'ok' if c_ok else 'FAIL'} err {ek:.3e} (r1 {e1:.3e}) "
                      f"cache != r1 {nbad} clean {clean} repeat {rep}", flush=True)
                if not c_ok:
                    good = False
                    cnt.zero_()
                    wo.zero_()
                    wl.zero_()
                    break
            if good:
                d = _k1fa4_case(16, 3000, 8, 2048, 939, False, dev)
                qkv = d["qkv"].view(128, -1)
                o1 = _ax3q_call(qkv, d, d["config"], cos_sin, v).clone()
                torch.cuda.synchronize(dev)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    og = _ax3q_call(qkv, d, d["config"], cos_sin, v)
                gr.replay()
                gr.replay()
                torch.cuda.synchronize(dev)
                good = (float((og.float() - o1.float()).abs().max()) <= 0.02 * float(o1.float().abs().max()) + 1e-3
                        and int(cnt.abs().sum()) == 0 and float(wo.abs().sum()) == 0.0)
                del gr, og, d
        except Exception as ex:  # noqa: BLE001
            good = False
            print(f"[ax3q] self-test {v} raised {repr(ex)[-600:]}", flush=True)
            cnt.zero_()
            wo.zero_()
            wl.zero_()
        print(f"[ax3q] variant {v}: {'ok' if good else 'FAILED'} (worst err {worst:.3e})", flush=True)
        if good:
            ok.add(v)
    return ok


def _ax3q_prepare(state, t0):
    state["ax3q"] = set()
    if not AX3Q_ENABLE:
        return
    if time.time() - t0 > AX3Q_PREPARE_BY_S:
        print(f"[ax3q] skipped: prepare already took {time.time() - t0:.1f}s: FA4 chain stays", flush=True)
        return
    dev = torch.empty(0, device="cuda").device
    t1 = time.time()
    try:
        mod = _ax3q_build()
        for st in sorted({v[1] for v in AX3Q_ROUTE.values()} | {v[1] for vs in AX3Q_ARMS.values() for v in vs}):
            print(f"[ax3q] ST{st}: regs / local / smem / CTAs per SM {list(mod.ax3q_info(st))}", flush=True)
        state["ax3q"] = _ax3q_self_test(state["cos_sin"], dev)
    except Exception as ex:  # noqa: BLE001
        print(f"[ax3q] build / self-test FAILED {repr(ex)[-1000:]}: FA4 chain stays", flush=True)
        state["ax3q"] = set()
    live = {k: tuple(v) for k, v in AX3Q_ROUTE.items() if tuple(v) in state["ax3q"]}
    print(f"[ax3q] route live {live} ({time.time() - t1:.1f}s; prepare so far {time.time() - t0:.1f}s)", flush=True)
    if AX3Q_SWEEP:
        try:
            _ax3q_sweep(state, dev)
        except Exception as ex:  # noqa: BLE001
            print(f"[ax3q] sweep FAILED {repr(ex)[-1000:]}", flush=True)
        torch.cuda.synchronize(dev)


'''

AX3Q_SWEEP_PY = r'''AX3Q_SWEEP_REPS = 40


def _ax3q_board_case(B, L, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    seq = (80000 * (1 + 0.075 * (2 * torch.rand((B,), generator=g, device=device) - 1))).long().clamp(1, 86000)
    max_pages = 86000 + L
    total = B * max_pages
    perm = torch.randperm(total, generator=g, device=device).to(torch.int32)
    pt = torch.full((B, max_pages), -1, dtype=torch.int32, device=device)
    off = 0
    for b in range(B):
        n = int(seq[b]) + L
        pt[b, :n] = perm[off:off + n]
        off += n
    f8 = torch.float8_e4m3fn
    k_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    v_cache = torch.empty((total, 1, 2, 64), dtype=f8, device=device)
    for c in (k_cache, v_cache):
        cv = c.view(-1, 128)
        for i in range(0, cv.shape[0], 1 << 20):
            cv[i:i + (1 << 20)] = torch.randn((min(1 << 20, cv.shape[0] - i), 128), generator=g, device=device).to(f8)
    q = (2 * torch.randn((B, L, 16, 64), generator=g, device=device)).to(torch.bfloat16)
    k = (2 * torch.randn((B, L, 2, 64), generator=g, device=device)).to(torch.bfloat16)
    v = torch.randn((B, L, 2, 64), generator=g, device=device).to(torch.bfloat16)
    gate = torch.randn((B, L, 1024), generator=g, device=device).to(torch.bfloat16)
    qkv = torch.cat([q.flatten(2), k.flatten(2), v.flatten(2), gate], -1)
    positions = (seq[:, None] + torch.arange(L, device=device)).reshape(-1)
    cache_loc = torch.gather(pt.long(), 1, positions.view(B, L)).reshape(-1)
    one = torch.tensor(1.0, dtype=torch.float32, device=device)
    return {"qkv": qkv, "positions": positions, "cache_loc": cache_loc, "cache_seqlens": (seq + L).to(torch.int32),
            "cu_seqlens_q": torch.arange(0, B * L + 1, L, dtype=torch.int32, device=device), "scheduler_metadata": None,
            "k_cache": k_cache, "v_cache": v_cache, "k_scale": one, "v_scale": one.clone(), "page_table": pt,
            "seq_lens": seq.to(torch.int32),
            "config": {"batch": B, "q_len": L, "num_q_heads": 16, "num_kv_heads": 2, "head_dim": 64, "page_size": 1,
                       "window": 2048, "rope_theta": 500000.0, "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125,
                       "kv_scale": 1.0}}


def _ax3q_sweep(state, device):
    import statistics
    cs = state["cos_sin"]
    flush = torch.ones((64 << 20,), dtype=torch.float32, device=device)
    sink = torch.empty((), dtype=torch.float32, device=device)
    stream = torch.cuda.Stream(device=device)
    cyc = int(300 * 1980)
    for _ in range(3):
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

    def timed(gr):
        with torch.cuda.stream(stream):
            torch.sum(flush, dim=0, out=sink)
            torch.cuda._sleep(cyc)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) * 1e3

    tiny = torch.zeros(1, device=device)
    g_empty, _ = capture(lambda: tiny.add_(1.0))
    for B in sorted(AX3Q_ARMS):
        d = _ax3q_board_case(B, 8, 7300 + B, device)
        cfg = d["config"]
        qkv = d["qkv"].view(B * 8, -1)
        arms = []
        if _v18_fa4_wanted(qkv, d, cfg, state):
            arms.append(("FA4 chain (ship)", lambda: _v18_fa4_core(qkv, d), None))
        for v in AX3Q_ARMS[B]:
            v = tuple(v)
            if v in state["ax3q"]:
                arms.append((f"AX3Q NS{v[0]} ST{v[1]}", lambda v=v: _ax3q_call(qkv, d, cfg, cs, v), v))
        graphs = []
        for name, fn, v in arms:
            gr, out = capture(fn)
            graphs.append((name, gr, out, v))
        ts = {name: [] for name, _, _, _ in graphs}
        te = []
        for _ in range(AX3Q_SWEEP_REPS):
            for name, gr, _, _ in graphs:
                ts[name].append(timed(gr))
            te.append(timed(g_empty))
        ovh = statistics.median(te)
        ref = graphs[0][2].float().reshape(-1, 1024)
        base = statistics.median(ts[graphs[0][0]])
        print(f"[ax3q] sweep B{B} q_len 8: replay overhead {ovh:.2f} us; raw / net median us:", flush=True)
        for name, gr, out, v in graphs:
            med = statistics.median(ts[name])
            err = float((out.float().reshape(-1, 1024) - ref).norm() / ref.norm())
            line = f"[ax3q]   {name:17s} raw {med:6.2f} net {med - ovh:6.2f} ({med - base:+.2f}) err {err:.2e}"
            if v is not None:
                prof = torch.zeros(8 * B * v[0], dtype=torch.int64, device=device)
                g2, _ = capture(lambda v=v, prof=prof: _ax3q_call(qkv, d, cfg, cs, v, prof=prof))
                reps = []
                for _ in range(6):
                    prof.zero_()
                    timed(g2)
                    pm = prof.view(-1, 8)[:, :7].double().cpu()
                    t0 = pm[:, 0].min()
                    reps.append(torch.where(pm > 0, (pm - t0) / 1000.0, torch.full_like(pm, float("nan"))))
                st = torch.stack(reps)
                med_m = st.nanmedian(dim=1).values.median(dim=0).values
                mx = torch.stack([torch.tensor([float(r[:, k][~torch.isnan(r[:, k])].max()) if (~torch.isnan(r[:, k])).any()
                                                else float("nan") for k in range(7)]) for r in reps]).median(dim=0).values
                line += " | " + " ".join(f"{k}:{med_m[k]:.2f}/{mx[k]:.2f}" for k in range(7))
                del g2, prof
            print(line, flush=True)
        del graphs, d, qkv
        torch.cuda.synchronize(device)
    del flush


'''

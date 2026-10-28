"""b1a_patches.py: B1A = one-launch FULL attention at B = 1 (decode1 full q_len 1, verify1 full q_len 8), notes/study/b1a.md.

Today B = 1 full attention is the FA4 route, three launches (Triton prologue + FA4 split-KV + Triton combine-gate): 14.4 us at
q_len 1, 17.4 us at q_len 8 in-layer; the best public layer runs one kernel at 11.0 / 13.9 us. VFA's flat kernel at B = 1 lost
twice (R-vfa-6 one-CTA tail, R-vfa-7 bounded cooperative finalize: the flat kernel's per-CTA fixed costs beat the two launches
saved). B1A is a different kernel, written for the once-per-launch path of one request:

* grid G = occupancy x SMs (264 at 2 CTAs / SM), 256 threads, no producer warps, no ring. CTA c owns the cached keys
  [T c / G, T (c + 1) / G) of T = n + NEWV virtual keys (n cached keys; the last CTA's NEWV slots are the new-key unit), so a
  CTA holds <= NT x 64 keys (NT 5: 320 keys, 80 KB of K + V) and loads them ALL at once: straight-line predicated cp.async of
  whole 128-B rows (both KV heads, 8 lanes per row, 128B swizzle), one mbarrier per 64-key tile (cp.async.mbarrier.arrive.noinc).
* SPEC: the key range comes from the page-table width (n = width x page - L, a shape the host knows), so the page-index loads
  do not wait for cache_seqlens; the true length is read in parallel and, if it differs, every CTA waits for its copies and
  re-issues with the true range (correct either way; a counter in bar[3] records misses).
* everything before griddepcontrol.wait reads only layer inputs (page table, lengths, old cache rows); qkv (the predecessor's
  output) is read after it. Q (16 L rows of 64 dims) is RMS-normalized in fp32, rounded to bf16 (stock qk-norm output),
  e4m3 (queries >= E4M3, = the FA4 prologue), straight into mma A-fragment registers. The last CTA computes the new K / V rows
  (k: RMSNorm -> bf16 -> / k_scale -> e4m3; v: / v_scale -> e4m3), writes them to the cache at cache_loc and attends them from
  shared memory (causal inside the block); no CTA reads cache rows >= n.
* math: FP8 mma.sync m16n8k32 (e4m3 x e4m3 -> f32). QK: A = Q registers, B = K rows by ldmatrix (no conversion). PV: A = P
  (e4m3, P = 2^(s - m + 8) <= 256 with the exact running row max), keys permuted into the A-fragment order straight from the QK
  accumulators; B = V by ldmatrix.trans + prmt 0x6420 / 0x7531 (two e4m3 dims per b16); row sums by one extra mma against e4m3
  ones (the same rounded P as the numerator). q_len 8: warp = (KV head, 16-row m-block = 2 tokens x 8 heads) over all keys;
  q_len 1: warp = (KV head, key quarter), 8 rows (8 padding rows skipped).
* merge: each CTA adds o * 2^(m - 8) and l * 2^(m - 8) (absolute frame, |s| bounded by the qk-norm) into workspace copy c % W
  with red.global.add (q_len 1 sums its 4 key quarters in shared memory first). Then a cross-CTA barrier: every CTA arrives
  once (count + generation pair, acq_rel); only the 2 L finalizer CTAs wait, bounded by %globaltimer (B1A_SPIN_MS, ~20 ms; on
  timeout a sticky flag bar[2] = 1 and exit, never a trap); the rest exit. Finalizer warp = one output row: sum the W copies,
  x v_scale / l, bf16, RMSNorm, x sigmoid(gate), bf16 out; it zeroes what it read, the last arriver resets the count.
* PDL in (programmatic stream serialization) and out (launch_dependents after the wait, B1A_TRIG); CUDA-graph capturable: no
  host syncs, allocations independent of input values, every branch on input values is inside the kernel.
* prepare(): build in the PT1 pool, occupancy check (blocks/SM x SMs >= grid for q_len 1 and 8), self-test vs the torch
  reference and vs the FA4 path (same e4m3 queries), new cache rows, workspace / barrier clean, graph replay on new inputs;
  any failure prints a loud FALLBACK line and FA4 stays. B1A_ENABLE / B1A_ROUTE are read at call time (A/B arms).

    python -I tools/b1a_patches.py --cuda                  CUDA source (device-only build) on stdout
    python -I tools/b1a_patches.py --ptxas [--nvcc=PATH] [--inc=DIR ...]   compile-only sm_90a check: regs / spills, C75xx
    python -I tools/b1a_patches.py --ptxas --full --inc=<torch/include> --inc=<torch/include/torch/csrc/api/include>
        --inc=<python include> --inc=<cuda include>     the same with the torch host wrapper (load_inline's .cu, C++20)
Generator: tools/gen_k3_b1a.py (k3_gc_1 + B1A -> kernels/lab/b1a_k3_gc_1_ab.py).
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)

B1A_BLOCK = r'''


# ---------------------------------------------------------------- B1A: one-launch B = 1 full attention (tools/b1a_patches.py)
B1A_ENABLE = True
B1A_ROUTE = {1: True, 8: True}
B1A_GRID = 0
B1A_NT = 5
B1A_COPIES = 8
B1A_NEWV = 32
B1A_SPEC = True
B1A_TRIG = 1
B1A_PDL = True
B1A_SYNC = 1
B1A_SPIN_MS = 20
B1A_NEED_FA4 = True
B1A_SELF_TEST = True
B1A_PREPARE_BY_S = 240.0
B1A_CASES = ((80000, 1, 1, 1.0, 0, 501), (80000, 8, 1, 1.0, 0, 502), (20000, 1, 1, 1.0, 0, 503), (20000, 8, 1, 1.0, 0, 504),
             (300, 1, 1, 1.0, 0, 505), (300, 8, 1, 1.0, 0, 506), (3, 8, 1, 1.0, 0, 507), (0, 1, 1, 1.0, 0, 508),
             (5000, 1, 1, 1.0, 7, 509), (5000, 8, 1, 1.0, 7, 510), (9000, 8, 64, 1.0, 0, 511), (9000, 1, 64, 1.0, 0, 512),
             (3000, 8, 1, 1.0 / 64, 0, 513), (3000, 1, 1, 1.0 / 64, 0, 514), (84000, 8, 1, 1.0, 0, 515))
_B1A = {"mod": None, "ws": {}, "grid": {}, "occ": {}, "nt": None}

B1A_CPP_SRC = r"""
#include <torch/extension.h>
void b1a_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor page_table,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o,
                 torch::Tensor ws_l, torch::Tensor bar, torch::Tensor out, int64_t L, int64_t G, int64_t W, int64_t page_shift,
                 int64_t spec, int64_t newv, int64_t trig, int64_t pdl, int64_t sync, int64_t nt, double qk_sm, double eps,
                 int64_t spin_ns);
int64_t b1a_occupancy(int64_t L, int64_t nt);
"""

B1A_CUDA_SRC = r"""
#ifndef B1A_DEVICE_ONLY
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define B1A_SM90 1
#else
#define B1A_SM90 0
#endif

namespace b1a {

constexpr int HQ = 16, HKV = 2, D = 64;          // tp8 shard: 16 q heads, 2 KV heads (8 q heads each), head dim 64
constexpr int ROWB = HKV * D;                    // 128 B per cached key row (both KV heads, e4m3)
constexpr int TK = 64;                           // keys per cp.async group (one mbarrier)
constexpr int TILEB = TK * ROWB;                 // 8 KB per K (and per V) tile
constexpr int NU = 32;                           // rows of the new-key unit (rows >= L are zero)
constexpr int NTHR = 256;
constexpr int MAXW = 8;                          // workspace copies
constexpr float LOG2E = 1.4426950408889634f;
constexpr float POFF = 8.f;                      // P = 2^(s - m + 8) in (0, 256]: 8 more octaves above e4m3's subnormals
constexpr uint32_t ONES = 0x38383838u;           // four e4m3 1.0: the row-sum mma operand
constexpr unsigned FULL = 0xffffffffu;

template <int L, int NT> struct Lay {            // shared memory (bytes from a 128-B aligned base)
  static constexpr int K = 0;                    // NT K tiles [64 keys][128 B], 16-B chunk c of row r at ((c ^ (r & 7)) << 4)
  static constexpr int V = NT * TILEB;           // NT V tiles, same layout
  static constexpr int NK = 2 * NT * TILEB;      // new-key unit: K [32][128 B], then V
  static constexpr int NV = NK + NU * ROWB;
  static constexpr int OS = NV + NU * ROWB;      // q_len 1: [4 key quarters][16 rows][64] f32 partials
  static constexpr int LS = OS + (L == 1 ? 4 * 16 * D * 4 : 0);
  static constexpr int BAR = LS + (L == 1 ? 4 * 16 * 4 : 0);
  static constexpr int MISC = BAR + NT * 8;
  static constexpr int BYTES = MISC + 16 + 128;  // + alignment slack
};

struct Params {
  const __nv_bfloat16* qkv;     // [L, ld_qkv] bf16: q (16 x 64) | k (2 x 64) | v (2 x 64) | gate (16 x 64)
  const int* cache_seqlens;     // [1] = cached keys n + L
  const long long* cache_loc;   // [L] flat cache rows of the block's new tokens
  const int* page_table;        // [1, pt_stride]
  uint8_t* k_cache;             // [rows][128] e4m3 (rows = pages x page size)
  uint8_t* v_cache;
  const float* k_scale;         // key = e4m3 x k_scale
  const float* v_scale;
  float* ws_o;                  // [W][16 L][64] f32 absolute-frame numerators (zero between calls)
  float* ws_l;                  // [W][16 L] denominators
  unsigned* bar;                // [0] arrivals, [1] generation, [2] sticky timeout flag, [3] speculation misses
  __nv_bfloat16* out;           // [L, 16 x 64]
  long long nrows;              // cache rows: bound for speculative rows
  long long spin_ns;            // finalizer wait bound
  int ld_qkv, pt_stride, page_shift, spec_n, newv, W, trig, sync;
  float qk_sm;                  // sm_scale x qk_scale_factor
  float eps;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float2 bf2f(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v)); }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel)); return r;
}
__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d) {   // 4 x e4m3 (RN, satfinite), a in the low byte
  uint32_t r;
  asm("{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\ncvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\nmov.b32 %0, {lo, hi};\n}"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
__device__ __forceinline__ uint16_t pk2(float a, float b) {                      // 2 x e4m3, a in the low byte
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(FULL, v, 1));
  return fmaxf(v, __shfl_xor_sync(FULL, v, 2));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}

#if B1A_SM90
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(b)), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile("{\n.reg .pred P;\nB1AW_%=:\nmbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n@!P bra B1AW_%=;\n}\n"
               ::"r"(sa(b)), "r"(parity) : "memory");
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint64_t* b) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sa(b)) : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void mma8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void red4(float* a, float x, float y, float z, float w) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(a), "f"(x), "f"(y), "f"(z), "f"(w) : "memory");
}
__device__ __forceinline__ void red1(float* a, float x) { asm volatile("red.global.add.f32 [%0], %1;" ::"l"(a), "f"(x) : "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* a) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ unsigned atom_add_acqrel(unsigned* a, unsigned v) {
  unsigned old; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory"); return old;
}
__device__ __forceinline__ void st_relaxed(unsigned* a, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ void st_release(unsigned* a, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// cached keys [a, a + kc) of CTA c: T = n + newv virtual keys split evenly, the last CTA's newv slots are the new-key unit
__device__ __forceinline__ void key_range(int n, int newv, int c, int G, int& a, int& kc) {
  const long long T = (long long)n + newv;
  const int v0 = (int)((T * c) / G), v1 = (int)((T * (c + 1)) / G);
  a = min(v0, n);
  kc = max(min(v1, n) - a, 0);
}

// thread q copies 16-B chunk q & 7 of the tile rows (q >> 3) and (q >> 3) + 32 of every tile: page indices first (registers) ...
template <int NT>
__device__ __forceinline__ void load_pages(const Params& p, int a, int kc, int tid, int (&pg)[NT][2]) {
  const int r0 = tid >> 3, lp = p.page_shift;
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int j = TK * i + r0 + 32 * k;
      pg[i][k] = (j < kc) ? __ldg(p.page_table + ((a + j) >> lp)) : -1;
    }
}

// ... then one straight-line burst of cp.async per tile (rows past the range or with a bad page: zero-filled), arrive on its barrier
template <int NT>
__device__ __forceinline__ void issue_copies(const Params& p, uint32_t kb, uint32_t vb, uint64_t* bars, int a, int kc, int tid,
                                             const int (&pg)[NT][2]) {
  const int ch = tid & 7, r0 = tid >> 3, lp = p.page_shift, pm = (1 << lp) - 1;
  const int ntl = (kc + TK - 1) / TK;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    if (i < ntl) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int jj = r0 + 32 * k;
        const long long row = ((long long)pg[i][k] << lp) + ((a + TK * i + jj) & pm);
        const bool ok = pg[i][k] >= 0 && row < p.nrows;
        const long long off = ok ? row * ROWB + ch * 16 : 0;
        const uint32_t d = (uint32_t)(i * TILEB + jj * ROWB) + ((uint32_t)(ch ^ (jj & 7)) << 4);
        cp_async16(kb + d, p.k_cache + off, ok ? 16 : 0);
        cp_async16(vb + d, p.v_cache + off, ok ? 16 : 0);
      }
      cp_async_mbar_arrive(bars + i);
    }
  }
}

// one unit of 8 NB keys for this warp's 16 rows (rows g: token tok0, rows g + 8: token tok1; R2 false = rows g + 8 unused)
template <int NB, bool NEW, bool R2>
__device__ __forceinline__ void unit(uint32_t kb, uint32_t vb, int cnt, int h, int lane, int tok0, int tok1,
                                     const uint32_t (&qa)[2][4], float (&o)[8][4], float (&la)[4], float (&m)[2], float cl2) {
  const int t = lane & 3;
  const uint32_t sw = (uint32_t)(lane & 7);
  float s[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {   // S = Q K^T: B fragment = 4 e4m3 dims of key 8 nb + g per register, straight from ldmatrix
    s[nb][0] = 0.f; s[nb][1] = 0.f; s[nb][2] = 0.f; s[nb][3] = 0.f;
    uint32_t r[4];
    ldsm_x4(r, kb + (uint32_t)(8 * nb + (lane & 7)) * ROWB + ((((uint32_t)(4 * h + (lane >> 3))) ^ sw) << 4));
    mma8(s[nb], qa[0], r[0], r[1]);
    mma8(s[nb], qa[1], r[2], r[3]);
  }
  if (NEW || cnt < 8 * NB) {   // keys past the range; new keys: causal inside the block
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = 8 * nb + 2 * t + e;
        if (!(key < cnt && (!NEW || key <= tok0))) s[nb][e] = neg_inf();
        if (!(key < cnt && (!NEW || key <= tok1))) s[nb][2 + e] = neg_inf();
      }
  }
  float x0 = neg_inf(), x1 = neg_inf();
#pragma unroll
  for (int nb = 0; nb < NB; ++nb) {
    x0 = fmaxf(x0, fmaxf(s[nb][0], s[nb][1]));
    if (R2) x1 = fmaxf(x1, fmaxf(s[nb][2], s[nb][3]));
  }
  x0 = quad_max(x0) * cl2;
  if (R2) x1 = quad_max(x1) * cl2;
  const float n0 = fmaxf(m[0], x0), n1 = R2 ? fmaxf(m[1], x1) : m[1];
  if (__any_sync(FULL, (n0 > m[0]) || (n1 > m[1]))) {   // the running max moved: rescale numerators and sums
    const float a0 = (n0 > m[0]) ? ex2(m[0] - n0) : 1.f;
    const float a1 = (n1 > m[1]) ? ex2(m[1] - n1) : 1.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j][0] *= a0; o[j][1] *= a0; o[j][2] *= a1; o[j][3] *= a1; }
    la[0] *= a0; la[1] *= a0; la[2] *= a1; la[3] *= a1;
    m[0] = n0; m[1] = n1;
  }
  const float b0 = (m[0] == neg_inf()) ? 0.f : m[0] - POFF;
  const float b1 = (m[1] == neg_inf()) ? 0.f : m[1] - POFF;
#pragma unroll
  for (int kq = 0; kq < NB / 4; ++kq) {   // 32 keys per k-step; A k index 4 t + e <-> key 2 t + (e & 1) + 8 (e >> 1) (+ 16)
    uint32_t pa[4];
    const int q = 4 * kq;
    pa[0] = pk4(ex2(fmaf(s[q][0], cl2, -b0)), ex2(fmaf(s[q][1], cl2, -b0)), ex2(fmaf(s[q + 1][0], cl2, -b0)), ex2(fmaf(s[q + 1][1], cl2, -b0)));
    pa[2] = pk4(ex2(fmaf(s[q + 2][0], cl2, -b0)), ex2(fmaf(s[q + 2][1], cl2, -b0)), ex2(fmaf(s[q + 3][0], cl2, -b0)), ex2(fmaf(s[q + 3][1], cl2, -b0)));
    if (R2) {
      pa[1] = pk4(ex2(fmaf(s[q][2], cl2, -b1)), ex2(fmaf(s[q][3], cl2, -b1)), ex2(fmaf(s[q + 1][2], cl2, -b1)), ex2(fmaf(s[q + 1][3], cl2, -b1)));
      pa[3] = pk4(ex2(fmaf(s[q + 2][2], cl2, -b1)), ex2(fmaf(s[q + 2][3], cl2, -b1)), ex2(fmaf(s[q + 3][2], cl2, -b1)), ex2(fmaf(s[q + 3][3], cl2, -b1)));
    } else {
      pa[1] = 0u;
      pa[3] = 0u;
    }
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {   // V chunk cc of this head: lane = key 32 kq + lane; trans -> 2 dims x 2 keys per b32
      uint32_t r[4];
      ldsm_x4_t(r, vb + (uint32_t)(32 * kq + lane) * ROWB + ((((uint32_t)(4 * h + cc)) ^ sw) << 4));
      mma8(o[2 * cc], pa, prmt(r[0], r[1], 0x6420u), prmt(r[2], r[3], 0x6420u));       // dims 16 cc + 2 g
      mma8(o[2 * cc + 1], pa, prmt(r[0], r[1], 0x7531u), prmt(r[2], r[3], 0x7531u));   // dims 16 cc + 2 g + 1
    }
    mma8(la, pa, ONES, ONES);
  }
}
#endif

template <int L, int NT>
__global__ void __launch_bounds__(NTHR, 2) b1a_kernel(const __grid_constant__ Params p) {
#if B1A_SM90
  using S = Lay<L, NT>;
  constexpr int ROWS = 16 * L;                   // output rows: token j x 16 heads (row = 16 j + head)
  constexpr int MB = (L == 8) ? 4 : 1;           // 16-row m-blocks per KV head
  constexpr int KS = 4 / MB;                     // warps per KV head that split the keys
  constexpr bool R2 = (L == 8);
  constexpr int NFIN = ROWS / 8;                 // finalizer CTAs (one output row per warp)
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((128u - (sa(smem_raw) & 127u)) & 127u);
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + S::BAR);
  int* misc = reinterpret_cast<int*>(smem + S::MISC);
  const uint32_t sb = sa(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x, G = gridDim.x;
  const bool tail = (c == G - 1);
  if (p.trig == 2) pdl_trigger();

  // 1. this CTA's keys (speculated from the page-table width), page indices, then every K / V row in one burst
  const int nmax = max((p.pt_stride << p.page_shift) - L, 0);
  const int nlen = __ldg(p.cache_seqlens);
  int n = (p.spec_n >= 0) ? min(p.spec_n, nmax) : min(max(nlen - L, 0), nmax);
  int a, kc;
  key_range(n, p.newv, c, G, a, kc);
  int pg[NT][2];
  load_pages<NT>(p, a, kc, tid, pg);
  if (tid < NT) mbar_init(bars + tid, NTHR);
  if (tail) {
#pragma unroll
    for (int i = tid; i < 2 * NU * ROWB / 16; i += NTHR) *reinterpret_cast<uint4*>(smem + S::NK + 16 * i) = make_uint4(0u, 0u, 0u, 0u);
  }
  mbar_fence_init();
  __syncthreads();
  issue_copies<NT>(p, sb + S::K, sb + S::V, bars, a, kc, tid, pg);
  int redo = 0;   // tiles [0, redo) completed one phase already (speculation missed): wait parity 1 there
  {
    const int nt_ = min(max(nlen - L, 0), nmax);
    if (nt_ != n) {   // CTA-uniform: the page table is wider than the cache: drain, re-issue with the true range
      const int ntl0 = (kc + TK - 1) / TK;
      for (int i = 0; i < ntl0; ++i) mbar_wait(bars + i, 0u);
      __syncthreads();
      if (c == 0 && tid == 0) atomicAdd(p.bar + 3, 1u);
      n = nt_;
      key_range(n, p.newv, c, G, a, kc);
      load_pages<NT>(p, a, kc, tid, pg);
      issue_copies<NT>(p, sb + S::K, sb + S::V, bars, a, kc, tid, pg);
      redo = ntl0;
    }
  }
  const int ntl = (kc + TK - 1) / TK;

  // 2. after the predecessor (qkv): Q fragments, the new K / V rows (last CTA), the finalizers' gate values
  pdl_wait();
  if (p.trig == 1) pdl_trigger();
  const int h = warp >> 2, wq = warp & 3, mb = wq % MB, ks = wq / MB;
  const int g = lane >> 2, t = lane & 3;
  const int tok0 = (L == 8) ? 2 * mb : 0, tok1 = (L == 8) ? 2 * mb + 1 : 0;
  const float ksc = __ldg(p.k_scale), vsc = __ldg(p.v_scale);
  uint32_t qa[2][4];
  {   // A rows g / g + 8 = query head 8 h + g of tokens tok0 / tok1; lane holds dims 16 cc + 4 t + e (cc 0..3, e 0..3)
    float x[2][16];
#pragma unroll
    for (int rr = 0; rr < (R2 ? 2 : 1); ++rr) {
      const __nv_bfloat16* qr = p.qkv + (size_t)(rr ? tok1 : tok0) * p.ld_qkv + (size_t)(h * 8 + g) * D + 4 * t;
      float ss = 0.f;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        const uint2 v = __ldcg(reinterpret_cast<const uint2*>(qr + 16 * cc));
        const float2 lo = bf2f(v.x), hi = bf2f(v.y);
        x[rr][4 * cc] = lo.x; x[rr][4 * cc + 1] = lo.y; x[rr][4 * cc + 2] = hi.x; x[rr][4 * cc + 3] = hi.y;
        ss += lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
      }
      ss += __shfl_xor_sync(FULL, ss, 1);
      ss += __shfl_xor_sync(FULL, ss, 2);
      const float rs = rsqrtf(ss * (1.f / 64.f) + p.eps);
#pragma unroll
      for (int e = 0; e < 16; ++e) x[rr][e] = bfr(x[rr][e] * rs);
    }
#pragma unroll
    for (int s2 = 0; s2 < 2; ++s2) {   // k-step s2 = dims 32 s2 .. + 31: regs (row g, lo 16), (row g + 8, lo), (g, hi), (g + 8, hi)
      qa[s2][0] = pk4(x[0][8 * s2], x[0][8 * s2 + 1], x[0][8 * s2 + 2], x[0][8 * s2 + 3]);
      qa[s2][2] = pk4(x[0][8 * s2 + 4], x[0][8 * s2 + 5], x[0][8 * s2 + 6], x[0][8 * s2 + 7]);
      qa[s2][1] = R2 ? pk4(x[1][8 * s2], x[1][8 * s2 + 1], x[1][8 * s2 + 2], x[1][8 * s2 + 3]) : 0u;
      qa[s2][3] = R2 ? pk4(x[1][8 * s2 + 4], x[1][8 * s2 + 5], x[1][8 * s2 + 6], x[1][8 * s2 + 7]) : 0u;
    }
  }
  if (tail) {   // new rows: job = (token jj, KV head hh, kind) per warp, lane = dims 2 lane, 2 lane + 1
    for (int job = warp; job < 4 * L; job += NTHR / 32) {
      const int kind = job & 1, hh = (job >> 1) & 1, jj = job >> 2;
      const uint32_t raw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)jj * p.ld_qkv + (size_t)(HQ + kind * HKV + hh) * D) + lane);
      float2 xv = bf2f(raw);
      if (kind == 0) {
        const float rs = rsqrtf(warp_sum(xv.x * xv.x + xv.y * xv.y) * (1.f / 64.f) + p.eps);
        xv.x = bfr(xv.x * rs);
        xv.y = bfr(xv.y * rs);
      }
      const float sc = kind ? vsc : ksc;
      const uint16_t e8 = pk2(xv.x / sc, xv.y / sc);
      const long long loc = __ldg(p.cache_loc + jj);
      *reinterpret_cast<uint16_t*>((kind ? p.v_cache : p.k_cache) + loc * ROWB + hh * D + 2 * lane) = e8;
      const int bb = hh * D + 2 * lane;
      *reinterpret_cast<uint16_t*>(smem + (kind ? S::NV : S::NK) + jj * ROWB + (((bb >> 4) ^ (jj & 7)) << 4) + (bb & 15)) = e8;
    }
  }
  uint32_t graw = 0u;
  if (c < NFIN) {   // this warp's output row in the finalize: row 8 c + warp = token r >> 4, head r & 15
    const int r = 8 * c + warp;
    graw = __ldcg(reinterpret_cast<const unsigned int*>(p.qkv + (size_t)(r >> 4) * p.ld_qkv + (size_t)(HQ + 2 * HKV + (r & 15)) * D) + lane);
  }
  __syncthreads();   // the new-key unit is in shared memory

  // 3. attention over this CTA's keys as the tiles land
  float o[8][4], la[4] = {0.f, 0.f, 0.f, 0.f}, m[2] = {neg_inf(), neg_inf()};
#pragma unroll
  for (int j = 0; j < 8; ++j) { o[j][0] = 0.f; o[j][1] = 0.f; o[j][2] = 0.f; o[j][3] = 0.f; }
  const float cl2 = p.qk_sm * ksc * LOG2E;   // raw e4m3 x e4m3 score -> log2 units
  if (L == 8) {
    for (int i = 0; i < ntl; ++i) {   // two 32-key units per tile (64-key units spill at the 128-register cap)
      mbar_wait(bars + i, i < redo ? 1u : 0u);
      unit<4, false, R2>(sb + S::K + i * TILEB, sb + S::V + i * TILEB, min(32, kc - TK * i), h, lane, tok0, tok1, qa, o, la, m, cl2);
      if (kc - TK * i > 32)
        unit<4, false, R2>(sb + S::K + i * TILEB + 32 * ROWB, sb + S::V + i * TILEB + 32 * ROWB, min(32, kc - TK * i - 32), h, lane,
                           tok0, tok1, qa, o, la, m, cl2);
    }
    if (tail) unit<4, true, R2>(sb + S::NK, sb + S::NV, L, h, lane, tok0, tok1, qa, o, la, m, cl2);
  } else {
    const int nu = (kc + 31) >> 5;               // 32-key units, unit u -> warp u % 4 of each head
    for (int u = ks; u < nu; u += KS) {
      const int i = u >> 1, off = (u & 1) * 32;
      mbar_wait(bars + i, i < redo ? 1u : 0u);
      unit<4, false, R2>(sb + S::K + i * TILEB + off * ROWB, sb + S::V + i * TILEB + off * ROWB, min(32, kc - TK * i - off), h, lane,
                         tok0, tok1, qa, o, la, m, cl2);
    }
    if (tail && ks == nu % KS) unit<4, true, R2>(sb + S::NK, sb + S::NV, L, h, lane, tok0, tok1, qa, o, la, m, cl2);
  }

  // 4. merge: o 2^(m - 8), l 2^(m - 8) into workspace copy c % W (lane holds dims 16 cc + 4 t .. + 3 of rows g / g + 8)
  const float f0 = (m[0] == neg_inf()) ? 0.f : ex2(fminf(m[0] - POFF, 126.f));
  const float f1 = (m[1] == neg_inf()) ? 0.f : ex2(fminf(m[1] - POFF, 126.f));
  const int cp = c % p.W;
  float* wo = p.ws_o + (size_t)cp * ROWS * D;
  float* wl = p.ws_l + (size_t)cp * ROWS;
  const bool any = (kc > 0) || tail;
  if (L == 8) {
    if (any) {
      const int R0 = tok0 * 16 + 8 * h + g, R1 = tok1 * 16 + 8 * h + g;
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        red4(wo + R0 * D + 16 * cc + 4 * t, o[2 * cc][0] * f0, o[2 * cc + 1][0] * f0, o[2 * cc][1] * f0, o[2 * cc + 1][1] * f0);
        red4(wo + R1 * D + 16 * cc + 4 * t, o[2 * cc][2] * f1, o[2 * cc + 1][2] * f1, o[2 * cc][3] * f1, o[2 * cc + 1][3] * f1);
      }
      if (t == 0) {
        red1(wl + R0, la[0] * f0);
        red1(wl + R1, la[2] * f1);
      }
    }
  } else {   // q_len 1: the 4 key quarters of each head summed in shared memory first (4x fewer atomics)
    float* os = reinterpret_cast<float*>(smem + S::OS);
    float* ls = reinterpret_cast<float*>(smem + S::LS);
    const int r = 8 * h + g;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc)
      *reinterpret_cast<float4*>(os + (ks * 16 + r) * D + 16 * cc + 4 * t) =
          make_float4(o[2 * cc][0] * f0, o[2 * cc + 1][0] * f0, o[2 * cc][1] * f0, o[2 * cc + 1][1] * f0);
    if (t == 0) ls[ks * 16 + r] = la[0] * f0;
    __syncthreads();
    if (any) {
      const int rr = tid >> 4, c4 = tid & 15;
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float4 v = *reinterpret_cast<const float4*>(os + (k * 16 + rr) * D + 4 * c4);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      }
      red4(wo + rr * D + 4 * c4, acc.x, acc.y, acc.z, acc.w);
      if (c4 == 0) red1(wl + rr, (ls[rr] + ls[16 + rr]) + (ls[32 + rr] + ls[48 + rr]));
    }
  }

  // 5. barrier: every CTA arrives once; only the NFIN finalizers wait (bounded by %globaltimer, sticky flag, no trap)
  if (p.sync == 0) __threadfence();
  __syncthreads();
  if (tid == 0) {
    int ok = 1;
    const unsigned g0 = ld_acquire(p.bar + 1);           // generation before this CTA's arrival (moves only after all arrive)
    const unsigned old = atom_add_acqrel(p.bar, 1u);
    if (old == (unsigned)G - 1u) {                         // last arrival: next call starts from 0, release the generation
      st_relaxed(p.bar, 0u);
      st_release(p.bar + 1, g0 + 1u);
    } else if (c < NFIN) {
      const long long t0 = gtime();
      while (ld_acquire(p.bar + 1) == g0) {
        if (gtime() - t0 > p.spin_ns) {
          atomicExch(p.bar + 2, 1u);
          ok = 0;
          break;
        }
        __nanosleep(32);
      }
    }
    misc[0] = ok;
  }
  if (c >= NFIN) return;
  __syncthreads();
  if (!misc[0]) return;
  if (p.sync == 0) __threadfence();

  // 6. finalize row 8 c + warp: sum the copies, / l x v_scale, bf16, RMSNorm x sigmoid(gate), bf16; zero what was read
  {
    const int r = 8 * c + warp;
    float2 acc = make_float2(0.f, 0.f);
    float lsum = 0.f;
#pragma unroll
    for (int k = 0; k < MAXW; ++k) {
      if (k < p.W) {
        float2* src = reinterpret_cast<float2*>(p.ws_o + ((size_t)k * ROWS + r) * D) + lane;
        const float2 v = __ldcg(src);
        acc.x += v.x;
        acc.y += v.y;
        lsum += __ldcg(p.ws_l + (size_t)k * ROWS + r);
      }
    }
    __syncwarp();   // every lane has read the shared denominator before lane 0 zeroes it
#pragma unroll
    for (int k = 0; k < MAXW; ++k) {
      if (k < p.W) {
        __stcg(reinterpret_cast<float2*>(p.ws_o + ((size_t)k * ROWS + r) * D) + lane, make_float2(0.f, 0.f));
        if (lane == 0) __stcg(p.ws_l + (size_t)k * ROWS + r, 0.f);
      }
    }
    const float inv = (lsum > 0.f) ? vsc / lsum : 0.f;
    const float y0 = bfr(acc.x * inv), y1 = bfr(acc.y * inv);
    const float rs = rsqrtf(warp_sum(y0 * y0 + y1 * y1) * (1.f / 64.f) + p.eps);
    const float2 gg = bf2f(graw);
    const __nv_bfloat162 o2 = __floats2bfloat162_rn(y0 * rs * (1.f / (1.f + __expf(-gg.x))), y1 * rs * (1.f / (1.f + __expf(-gg.y))));
    *reinterpret_cast<__nv_bfloat162*>(p.out + (size_t)(r >> 4) * (HQ * D) + (r & 15) * D + 2 * lane) = o2;
  }
#endif
}

}  // namespace b1a

#ifndef B1A_DEVICE_ONLY
template <int L, int NT>
static void b1a_go(const b1a::Params& p, int G, cudaStream_t stream, bool pdl) {
  constexpr int SMEM = b1a::Lay<L, NT>::BYTES;
  static bool init = false;
  if (!init) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(b1a::b1a_kernel<L, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)G);
  cfg.blockDim = dim3(b1a::NTHR);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, b1a::b1a_kernel<L, NT>, p));
}

template <int L, int NT>
static int64_t b1a_occ() {
  constexpr int SMEM = b1a::Lay<L, NT>::BYTES;
  int nb = 0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(b1a::b1a_kernel<L, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, b1a::b1a_kernel<L, NT>, b1a::NTHR, SMEM));
  return nb;
}

int64_t b1a_occupancy(int64_t L, int64_t nt) {
  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10), "b1a: L 1 / 8, nt 5 / 10");
  if (L == 1) return nt == 10 ? b1a_occ<1, 10>() : b1a_occ<1, 5>();
  return nt == 10 ? b1a_occ<8, 10>() : b1a_occ<8, 5>();
}

void b1a_forward(torch::Tensor qkv, torch::Tensor cache_seqlens, torch::Tensor cache_loc, torch::Tensor page_table,
                 torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor ws_o,
                 torch::Tensor ws_l, torch::Tensor bar, torch::Tensor out, int64_t L, int64_t G, int64_t W, int64_t page_shift,
                 int64_t spec, int64_t newv, int64_t trig, int64_t pdl, int64_t sync, int64_t nt, double qk_sm, double eps,
                 int64_t spin_ns) {
  TORCH_CHECK((L == 1 || L == 8) && (nt == 5 || nt == 10), "b1a: L 1 / 8, nt 5 / 10");
  TORCH_CHECK(W >= 1 && W <= b1a::MAXW && G >= 2 * L && newv >= 0 && newv <= 4096 && page_shift >= 0 && page_shift <= 12, "b1a: W / G / newv / page");
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kBFloat16 && qkv.dim() == 2 && qkv.stride(1) == 1 && qkv.size(0) == L &&
                  qkv.size(1) >= (2 * b1a::HQ + 2 * b1a::HKV) * b1a::D && qkv.stride(0) % 4 == 0 &&
                  reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 16 == 0, "b1a: qkv");
  TORCH_CHECK(cache_seqlens.scalar_type() == at::kInt && cache_seqlens.is_contiguous() && cache_seqlens.numel() >= 1, "b1a: cache_seqlens");
  TORCH_CHECK(cache_loc.scalar_type() == at::kLong && cache_loc.is_contiguous() && cache_loc.numel() >= L, "b1a: cache_loc");
  TORCH_CHECK(page_table.scalar_type() == at::kInt && page_table.is_contiguous() && page_table.dim() == 2 && page_table.size(0) == 1,
              "b1a: page_table");
  TORCH_CHECK(k_cache.scalar_type() == at::kByte && k_cache.is_contiguous() && v_cache.scalar_type() == at::kByte && v_cache.is_contiguous() &&
                  k_cache.numel() == v_cache.numel() && k_cache.numel() % b1a::ROWB == 0, "b1a: caches");
  TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat && k_scale.numel() >= 1 && v_scale.numel() >= 1, "b1a: scales");
  TORCH_CHECK(ws_o.scalar_type() == at::kFloat && ws_o.is_contiguous() && ws_o.numel() >= b1a::MAXW * 128 * b1a::D, "b1a: ws_o");
  TORCH_CHECK(ws_l.scalar_type() == at::kFloat && ws_l.is_contiguous() && ws_l.numel() >= b1a::MAXW * 128, "b1a: ws_l");
  TORCH_CHECK(bar.scalar_type() == at::kInt && bar.is_contiguous() && bar.numel() >= 4, "b1a: bar");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == L * b1a::HQ * b1a::D, "b1a: out");
  const long long nmax = (long long)page_table.size(1) << page_shift;
  TORCH_CHECK(nmax - L >= 0 && nmax < (1ll << 30), "b1a: page-table width");
  TORCH_CHECK((nmax - L + newv + G - 1) / G <= nt * b1a::TK, "b1a: keys per CTA exceed nt x 64 (route check)");
  const c10::cuda::OptionalCUDAGuard guard(qkv.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  b1a::Params p;
  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv.data_ptr());
  p.cache_seqlens = cache_seqlens.data_ptr<int>();
  p.cache_loc = reinterpret_cast<const long long*>(cache_loc.data_ptr<int64_t>());
  p.page_table = page_table.data_ptr<int>();
  p.k_cache = k_cache.data_ptr<uint8_t>();
  p.v_cache = v_cache.data_ptr<uint8_t>();
  p.k_scale = k_scale.data_ptr<float>();
  p.v_scale = v_scale.data_ptr<float>();
  p.ws_o = ws_o.data_ptr<float>();
  p.ws_l = ws_l.data_ptr<float>();
  p.bar = reinterpret_cast<unsigned*>(bar.data_ptr<int>());
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.nrows = k_cache.numel() / b1a::ROWB;
  p.spin_ns = spin_ns;
  p.ld_qkv = (int)qkv.stride(0);
  p.pt_stride = (int)page_table.size(1);
  p.page_shift = (int)page_shift;
  p.spec_n = spec ? (int)(nmax - L) : -1;
  p.newv = (int)newv;
  p.W = (int)W;
  p.trig = (int)trig;
  p.sync = (int)sync;
  p.qk_sm = (float)qk_sm;
  p.eps = (float)eps;
  if (L == 1) {
    if (nt == 10) b1a_go<1, 10>(p, (int)G, stream, pdl != 0); else b1a_go<1, 5>(p, (int)G, stream, pdl != 0);
  } else {
    if (nt == 10) b1a_go<8, 10>(p, (int)G, stream, pdl != 0); else b1a_go<8, 5>(p, (int)G, stream, pdl != 0);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#else
void* b1a_instances[] = {(void*)b1a::b1a_kernel<1, 5>, (void*)b1a::b1a_kernel<8, 5>, (void*)b1a::b1a_kernel<1, 10>,
                         (void*)b1a::b1a_kernel<8, 10>};
#endif
"""


def _b1a_build():
    if _B1A["mod"] is not None:
        return _B1A["mod"]
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((B1A_CPP_SRC + B1A_CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    _B1A["mod"] = load_inline(
        name=f"b1a_{tag}", cpp_sources=B1A_CPP_SRC, cuda_sources=B1A_CUDA_SRC, functions=["b1a_forward", "b1a_occupancy"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    _K3_BUILD_S["b1a"] = time.time() - t0
    return _B1A["mod"]


def _b1a_fits(data, cfg, G):
    n = data["page_table"].shape[1] * cfg["page_size"] - cfg["q_len"]
    return n >= 0 and (n + B1A_NEWV + G - 1) // G <= _B1A["nt"] * 64


def _b1a_wanted(qkv, data, cfg, state):
    L, page = cfg["q_len"], cfg["page_size"]
    if not (B1A_ENABLE and state.get("b1a_ok") and cfg["batch"] == 1 and B1A_ROUTE.get(L) and not cfg["window"]):
        return False
    G = _B1A["grid"].get((L, str(qkv.device)))
    return (G is not None and not cfg["rope_theta"] and cfg["num_q_heads"] == 16 and cfg["num_kv_heads"] == 2
            and cfg["head_dim"] == 64 and page > 0 and (page & (page - 1)) == 0 and str(qkv.device) in _B1A["ws"]
            and qkv.dtype == torch.bfloat16 and qkv.dim() == 2 and qkv.stride(1) == 1 and qkv.stride(0) % 4 == 0
            and qkv.shape[0] == L and qkv.shape[1] >= 36 * 64 and qkv.data_ptr() % 16 == 0
            and data["k_cache"].dtype == FP8 and data["v_cache"].dtype == FP8
            and data["k_cache"].is_contiguous() and data["v_cache"].is_contiguous()
            and data["page_table"].dtype == torch.int32 and data["page_table"].dim() == 2 and data["page_table"].shape[0] == 1
            and data["page_table"].is_contiguous() and data["cache_seqlens"].dtype == torch.int32
            and data["cache_seqlens"].is_contiguous() and data["cache_loc"].dtype == torch.int64 and data["cache_loc"].is_contiguous()
            and _b1a_fits(data, cfg, G))


def _b1a_run(qkv, data, cfg, pdl=None, spec=None, trig=None, copies=None):
    L = cfg["q_len"]
    dev = qkv.device
    ws_o, ws_l, bar = _B1A["ws"][str(dev)]
    out = torch.empty((L, 16 * 64), dtype=torch.bfloat16, device=dev)
    _B1A["mod"].b1a_forward(qkv, data["cache_seqlens"], data["cache_loc"], data["page_table"], data["k_cache"].view(torch.uint8),
                            data["v_cache"].view(torch.uint8), data["k_scale"], data["v_scale"], ws_o, ws_l, bar, out, L,
                            _B1A["grid"][(L, str(dev))], int(copies or B1A_COPIES), cfg["page_size"].bit_length() - 1,
                            1 if (B1A_SPEC if spec is None else spec) else 0, B1A_NEWV, int(B1A_TRIG if trig is None else trig),
                            1 if (B1A_PDL if pdl is None else pdl) else 0, int(B1A_SYNC), int(_B1A["nt"]),
                            float(cfg["qk_scale_factor"] * cfg["sm_scale"]), float(K1FA4_RMS_EPS), int(B1A_SPIN_MS * 1000000))
    return out


def _b1a_try(qkv, data, cfg, state):
    try:
        return _b1a_run(qkv, data, cfg)
    except Exception as ex:
        _k3_disable(state, "b1a_ok", False, "B1A", ex)
    return None


def _b1a_case(dev, ctx, L, page, kv_scale, extra, seed):
    """One request of ctx cached keys; page-table width = the pages needed + extra (extra 0 at page 1 = the judge's shape)."""
    g = torch.Generator().manual_seed(seed)
    hq, hkv = 16, 2
    need = (ctx + L + page - 1) // page
    width = need + extra
    total = width + 5
    perm = torch.randperm(total, generator=g).to(torch.int32)
    pt = torch.full((1, width), -1, dtype=torch.int32)
    pt[0, :need] = perm[:need]
    u = torch.randn((hkv, 64), generator=g)
    kf = 0.7 * u + 0.7 * torch.randn((total * page, hkv, 64), generator=g)
    vf = torch.randn((total * page, hkv, 64), generator=g)
    k_cache = (kf / kv_scale).clamp(-448, 448).to(FP8).view(total, page, hkv, 64)
    v_cache = (vf / kv_scale).clamp(-448, 448).to(FP8).view(total, page, hkv, 64)
    q = 2 * torch.randn((1, L, hq * 64), generator=g)
    kn = 2 * torch.randn((1, L, hkv * 64), generator=g)
    vn = torch.randn((1, L, hkv * 64), generator=g)
    gate = torch.randn((1, L, hq * 64), generator=g)
    qkv = torch.cat([q, kn, vn, gate], -1).to(torch.bfloat16)
    pos = ctx + torch.arange(L)
    cache_loc = (pt[0].long()[pos // page] * page + pos % page).reshape(-1)
    data = {
        "qkv": qkv, "positions": pos, "cache_loc": cache_loc, "cache_seqlens": torch.tensor([ctx + L], dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, L], dtype=torch.int32), "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(kv_scale, dtype=torch.float32), "v_scale": torch.tensor(kv_scale, dtype=torch.float32),
        "page_table": pt, "seq_lens": torch.tensor([ctx], dtype=torch.int32),
    }
    data = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data["config"] = {"batch": 1, "q_len": L, "num_q_heads": hq, "num_kv_heads": hkv, "head_dim": 64, "page_size": page,
                      "window": 0, "rope_theta": 0.0, "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": kv_scale}
    return data


def _b1a_ws_clean(dev):
    ws_o, ws_l, bar = _B1A["ws"][str(dev)]
    return not (float(ws_o.abs().sum()) or float(ws_l.abs().sum()) or int(bar[0]) or int(bar[2]))


def _b1a_self_test(dev, fa4):
    n_fa4 = 0
    for (ctx, L, page, kvs, extra, seed) in B1A_CASES:
        d = _b1a_case(dev, ctx, L, page, kvs, extra, seed)
        if not _b1a_fits(d, d["config"], _B1A["grid"][(L, str(dev))]):
            print(f"[b1a] self-test ctx={ctx} L={L}: beyond nt x 64 keys per CTA, the route check rejects it (ok)", flush=True)
            continue
        ref_kn, ref_vn = _k1_ref_new_kv(d)
        of = None
        if fa4 and page == 1 and kvs == 1.0 and ctx >= 64:   # FA4 first (it writes the same new rows; B1A never reads them)
            try:
                of = k1fa4_run(dict(d, qkv=d["qkv"].view(L, -1)), pdl=False).view(L, -1).clone()
            except Exception as ex:
                print(f"[b1a] (FA4 comparison unavailable at ctx={ctx} L={L}: {ex!r}"[:300] + ")", flush=True)
        for rep in range(2):
            out = _b1a_run(d["qkv"].view(L, -1), d, d["config"], pdl=bool(rep))
            torch.cuda.synchronize(dev)
            slots = d["cache_loc"].view(1, L)
            kn, vn = (cc.view(torch.uint8).view(-1, 2, 64)[slots] for cc in (d["k_cache"], d["v_cache"]))
            tot, row = _k1_rel_rows(out, _vfa_ref_attention(d, kn, vn))
            ek = _k1_rel_rows(kn.view(FP8).float().view(L, -1), ref_kn.view(L, -1))[0]
            ev = _k1_rel_rows(vn.view(FP8).float().view(L, -1), ref_vn.view(L, -1))[0]
            vf = _k1_rel_rows(out, of)[0] if of is not None else -1.0
            clean = _b1a_ws_clean(dev)
            print(f"[b1a] self-test ctx={ctx} L={L} page={page} kvs={kvs:g} width+{extra} #{rep}: rel {tot:.2e} row {row:.2e} "
                  f"vs FA4 {vf:.2e} new k {ek:.1e} v {ev:.1e} ws {'clean' if clean else 'DIRTY'}", flush=True)
            if not (bool(torch.isfinite(out.float()).all()) and clean and tot < 2e-2 and row < 6e-2 and vf < 2e-2
                    and ek < 3e-2 and ev < 3e-2):
                raise RuntimeError(f"self-test mismatch ctx={ctx} L={L} page={page} kvs={kvs} extra={extra}")
        n_fa4 += of is not None
        del d
    if fa4 and n_fa4 < 4:
        raise RuntimeError(f"only {n_fa4} cases were compared with FA4")
    for L in (1, 8):   # graph capture: replay = eager, then fresh inputs copied into the captured tensors
        d = _b1a_case(dev, 20000, L, 1, 1.0, 0, 531 + L)
        o1 = _b1a_run(d["qkv"].view(L, -1), d, d["config"]).clone()
        torch.cuda.synchronize(dev)
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            og = _b1a_run(d["qkv"].view(L, -1), d, d["config"])
        gph.replay()
        torch.cuda.synchronize(dev)
        same = _k1_rel_rows(og, o1)[0] < 2e-3
        f = _b1a_case(dev, 20000, L, 1, 1.0, 0, 541 + L)
        f2 = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in f.items()}
        o2 = _b1a_run(f["qkv"].view(L, -1), f, f["config"]).clone()
        for k, v in f2.items():
            if isinstance(v, torch.Tensor):
                d[k].copy_(v)
        og.fill_(float("nan"))
        gph.replay()
        torch.cuda.synchronize(dev)
        same2 = _k1_rel_rows(og, o2)[0] < 2e-3 and bool(torch.isfinite(og).all())
        print(f"[b1a] graph check L={L}: replay identical {same}, after new inputs {same2}, ws {'clean' if _b1a_ws_clean(dev) else 'DIRTY'}",
              flush=True)
        del gph
        if not (same and same2 and _b1a_ws_clean(dev)):
            raise RuntimeError(f"graph check failed L={L}")


def _b1a_prepare(state, t0):
    state["b1a_ok"] = False
    if not B1A_ENABLE:
        return
    dev = torch.empty(0, device="cuda").device
    try:
        if time.time() - t0 > B1A_PREPARE_BY_S:
            raise RuntimeError(f"prepare already took {time.time() - t0:.0f}s (> {B1A_PREPARE_BY_S:.0f}s)")
        _pt1_wait("b1a")
        if _PT1["errors"].get("b1a"):
            raise RuntimeError(f"build failed: {_PT1['errors']['b1a']}")
        mod = _b1a_build()
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        _B1A["nt"] = int(B1A_NT)   # grid and occupancy are proven for this NT: fixed from here on (B1A_NT is a prepare-time knob)
        for L in (1, 8):   # every CTA must be co-resident (the finalizers wait for all arrivals)
            occ = int(mod.b1a_occupancy(L, _B1A["nt"]))
            G = min(int(B1A_GRID), occ * sms) if B1A_GRID else occ * sms
            _B1A["occ"][(L, str(dev))] = occ
            if occ < 1 or occ * sms < G or G < 2 * L:
                raise RuntimeError(f"occupancy {occ} CTAs/SM x {sms} SMs < grid {G} at q_len {L}")
            _B1A["grid"][(L, str(dev))] = G
        _B1A["ws"][str(dev)] = (torch.zeros(8 * 128 * 64, dtype=torch.float32, device=dev),
                                torch.zeros(8 * 128, dtype=torch.float32, device=dev), torch.zeros(8, dtype=torch.int32, device=dev))
        fa4 = bool(state.get("fa4v6") and K1FA4_STATE.get("ready"))
        if B1A_NEED_FA4 and not fa4:
            raise RuntimeError("the FA4 route is not live, so B1A cannot be checked against it")
        if B1A_SELF_TEST:
            _b1a_self_test(dev, fa4)
        bar = _B1A["ws"][str(dev)][2]
        if int(bar[2]) != 0:
            raise RuntimeError("cooperative finalize timed out in the self-test")
        state["b1a_ok"] = True
        print(f"[b1a] LIVE: B = 1 full attention q_len {sorted(k for k, v in B1A_ROUTE.items() if v)} in one launch, grid "
              f"{ {k[0]: v for k, v in _B1A['grid'].items()} } (occupancy {_B1A['occ']}), nt {_B1A['nt']}, copies {B1A_COPIES}, "
              f"spec {B1A_SPEC} (misses in the self-test {int(bar[3])}), trig {B1A_TRIG}, pdl {B1A_PDL}, build "
              f"{_K3_BUILD_S.get('b1a', 0):.0f}s, prepare so far {time.time() - t0:.1f}s", flush=True)
    except Exception as ex:
        state["b1a_ok"] = False
        print(f"[b1a] FALLBACK: B = 1 full attention stays on the FA4 route: {ex!r}"[:1500], flush=True)
'''

ROUTE_OLD = "    if o is None and not _va1_b1(cfg, state) and _v18_fa4_wanted(qkv, data, cfg, state):" + NL
ROUTE_NEW = ("    if o is None and _b1a_wanted(qkv, data, cfg, state):" + NL +
             "        o = _b1a_try(qkv, data, cfg, state)" + NL +
             "        o = o.view(tokens, hq * d) if o is not None else None" + NL) + ROUTE_OLD
JOBS_OLD = "    if VFA_ENABLE:" + NL + '        jobs.append(("vfa", _vfa_build))' + NL
JOBS_NEW = JOBS_OLD + "    if B1A_ENABLE:" + NL + '        jobs.append(("b1a", _b1a_build))' + NL
PREP_OLD = "    _vfa_prepare(state, t0)" + NL
PREP_NEW = PREP_OLD + "    _b1a_prepare(state, t0)" + NL
KFN = NL + "def kernel_fn(data, state):" + NL


def apply(text, sets=()):
    """Insert the B1A block before kernel_fn, hook the PT1 build pool, prepare() (after FA4 and VFA) and the attention route
    (before the FA4 B = 1 branch). sets: (NAME, VALUE) pairs overriding the block's knobs."""
    for old in (ROUTE_OLD, JOBS_OLD, PREP_OLD, KFN):
        assert text.count(old) == 1, f"anchor found {text.count(old)} times: {old!r}"
    assert NL + "B1A_ENABLE = " not in text and "def _b1a_" not in text, "the base already carries B1A"
    block = B1A_BLOCK
    for k, v in sets:
        old = next(ln for ln in block.split(NL) if ln.startswith(f"{k} = "))
        assert block.count(NL + old + NL) == 1, k
        block = block.replace(NL + old + NL, NL + f"{k} = {v}" + NL)
    text = text.replace(KFN, block + KFN)
    return text.replace(ROUTE_OLD, ROUTE_NEW).replace(JOBS_OLD, JOBS_NEW).replace(PREP_OLD, PREP_NEW)


def cuda_source(device_only=True):
    a = B1A_BLOCK.index('B1A_CUDA_SRC = r"""') + len('B1A_CUDA_SRC = r"""')
    b = B1A_BLOCK.index('"""', a)
    src = B1A_BLOCK[a:b]
    return ("#define B1A_DEVICE_ONLY 1" + NL + src) if device_only else src


def ptxas_check(nvcc="nvcc", inc=(), work=None, full=False):
    """Compile the device-only source (full: with the torch host wrapper) for sm_90a (-Xptxas=-v): per instantiation
    registers and spills; flags C75xx."""
    import tempfile
    wd = pathlib.Path(work or tempfile.mkdtemp(prefix="b1a_"))
    wd.mkdir(parents=True, exist_ok=True)
    cu = wd / "b1a.cu"
    cu.write_text(cuda_source(not full), encoding="utf-8")
    cmd = [nvcc, "-c", str(cu), "-o", str(wd / "b1a.o"), "-O3", "-std=c++20" if full else "-std=c++17",
           "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "-lineinfo"] + [f"-I{d}" for d in inc]
    if full:
        cmd += ["-DTORCH_EXTENSION_NAME=b1achk", "-DTORCH_API_INCLUDE_EXTENSION_H", "--compiler-options", "-fPIC",
                "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr"]
    pr = subprocess.run(cmd, capture_output=True, text=True)
    log = pr.stdout + pr.stderr
    (wd / "b1a.log").write_text(log, encoding="utf-8")
    if pr.returncode:
        print(log[-6000:])
        print("[b1a] COMPILE FAILED")
        return 1
    bad = False
    for ln in log.splitlines():
        mm = re.search(r"Compiling entry function '(\S+)'", ln)
        if mm:
            k = re.search(r"b1a_kernelILi(\d+)ELi(\d+)E", mm.group(1))
            print(f"-- b1a_kernel<L {k.group(1)}, NT {k.group(2)}>" if k else f"-- {mm.group(1)}")
        if "registers" in ln or "spill" in ln or "smem" in ln:
            print("   " + ln.strip())
            sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if sp and (int(sp.group(1)) or int(sp.group(2))):
                bad = True
        if re.search(r"C75\d\d", ln):
            print("   WARNING " + ln.strip())
            bad = True
    print(f"[b1a] ptxas: {'FAIL (spills or C75xx)' if bad else 'OK: no spills, no C75xx'} (log {wd / 'b1a.log'})")
    return 1 if bad else 0


def main():
    args = sys.argv[1:]
    if "--cuda" in args:
        sys.stdout.write(cuda_source(True))
        return 0
    if "--ptxas" in args:
        nvcc = next((a[7:] for a in args if a.startswith("--nvcc=")), "nvcc")
        inc = [a[6:] for a in args if a.startswith("--inc=")]
        work = next((a[7:] for a in args if a.startswith("--work=")), None)
        return ptxas_check(nvcc, inc, work, full="--full" in args)
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())

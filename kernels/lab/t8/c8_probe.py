"""R-t8-1 probes (notes/memory/gpu_requests.md, notes/study/t8_design.md s4). LAB FILE: box runs only (tools/box/t8_probe.py).

c8b_probe<NSLOT, XR, MODE>: the C8-B work decomposition on the moe_w4a16 board routings. 132 CTAs (one per SM) x 576
threads: warp 0 producer (lane 0 issues cp.async.bulk, L2 evict_first), warp 1 control (activation staging, counter
waits), warps 2..17 consumers (split-K 16 inside each 16-row granule).
  - every mode: on-device routing prologue from topk ids (local experts by count desc / index asc, token and route lists),
    then the deal: W13 granules (e, rt) expert-major round-robin over 132 SMs, W2 row tiles s and s + 132 of every
    expert (expert-major inside the SM), 2 slots of 18,432 B per granule (T8 layout: a byte permutation of K2A's stages).
  - MODE 0 (c8b_s): null consumers (wait full, release empty), no waits, no epilogues, no tail.
  - MODE 1 (c8b_d): + x rows / inter rows staged per expert phase into a double-buffered activation region, W2 phases wait
    on per-expert W13 counters (cnt13[e] == 512), granule epilogues (16-warp SMEM reduction, SwiGLU -> inter stores +
    cnt13 release for W13, z / ss partials for W2), the tail: A moment + ss reduction -> cnt2 hop, m for own channels ->
    cnt_m hop, 64 unpooler SMs stream 8 x 12,288 B and add f moments -> cnt_f hop, final out / s2 writes; the last CTA
    resets counters and accumulators.
  - MODE 2 (c8b_e): MODE 1 + K2A's consumer instruction stream on the staged words (deq4, bmul2 with K2A's scale words,
    LDS.128 X fragments from the activation rows, mma.sync m16n8k16): 8 mma per warp per slot.
c8f_probe<NSLOT, AB, MODE>: the C8-F front at decode-layer sizes. Same thread roles, 24,576 B slots.
  - stream order per SM: qkv (K-chunk s % 8, row tiles u, u + nc, ...; q/k/v tiles 0..79 before gate tiles 80..143),
    o tiles s, s + 132, ..., its router column slice (63,488 B), pooler quarter s % 4 tiles u4, u4 + 33, ... (2 slots each).
  - MODE 0 (c8f_s): null consumers, no waits.
  - MODE 1 (c8f_d): + x chunk / o-input / h quarter staging, qkv red.add + per-group counters, 8 attention CTAs (real
    128-B KV row gathers into registers at entry, q/k/v reads from qacc after the counters, a calibrated compute spin,
    partial store + counter), merger (CTA 0, after its gate tiles: 8 partials + gate rows -> o-input + counter), o phase
    moments -> hop, boundary s1 / h writes -> hop, router red.add.v4 partial logits, pooler red.add partials.
Bounded waits everywhere (err flags, no trap). Marks (%globaltimer) per CTA when marks != nullptr.
"""
import hashlib
import time

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace c8p {
constexpr unsigned FULL = 0xffffffffu;
constexpr int SLOTB = 18432, SLOTF = 24576, NCW = 16, NT = 32 * (NCW + 2), G = 132;
constexpr int XROWB = 8192 + 16;         // activation row: 4096 bf16 + 16 B pad
constexpr int UST = 12288;               // unpooler stage: 16 rows x 512 K x 12 bit
constexpr int RSL = 63488;               // router column slice per SM: 256 x 62 FP32
constexpr long long BUDGET = 1ll << 31;  // clock64 cycles per wait (~1 s) -> err flag, no trap
enum { C13 = 0, C2 = 32, CM = 33, CF = 34, CEXIT = 35, NCB = 36 };
enum { FQ0 = 0, FQ1 = 1, FQG = 2, FATT = 3, FOIN = 4, FM1 = 5, FH = 6, FEXIT = 7, NCF = 8 };

__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int smid() { unsigned v; asm volatile("mov.u32 %0, %%smid;" : "=r"(v)); return (int)v; }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_tx(uint32_t b, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 1, 1); return; }
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ void bulk_n(uint32_t dst, const void* src, uint32_t n, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar) : "memory");
}
__device__ __forceinline__ uint64_t evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void spin_ge(const int* p, int target, int* err) {
  if (ld_acq(p) >= target) return;
  const long long t0 = clock64();
  while (ld_acq(p) < target) { __nanosleep(32); if (clock64() - t0 > BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; } }
}
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async;" ::: "memory"); }
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" :: "n"(32 * NCW) : "memory"); }   // consumer warps only
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // K2A BITPOS3 E2M1 -> bf16x2 (scaled 2^-8)
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d;
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ======================================== C8-B ========================================
struct PB {
  const uint8_t* w13; const uint8_t* w2; const uint8_t* unp;     // [32][512][2][SLOTB], [32][256][2][SLOTB], [64][8][UST]
  const __nv_bfloat16* xe; const int* ids; const float* wts;     // [T][4096], [T][8] global ids, [T][8]
  const float* s1;                                               // [T][8192]
  __nv_bfloat16* inter; __nv_bfloat16* mo;                       // [64 routes][4096], [T][4096]
  __nv_bfloat16* out; float* s2;                                 // [T][8192] each
  float* ss; float* mom;                                         // [64], [8][4]
  int* cnt; int* err; long long* marks; int* stages;             // [NCB], [4], [132][16], [132]
  int T, lo;
};

template <int NSLOT, int XR, int MODE>
__global__ void __launch_bounds__(NT, 1) c8b_probe(const __grid_constant__ PB p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* act = sm + NSLOT * SLOTB;                                       // [2][XR][XROWB]
  float* red = reinterpret_cast<float*>(act + 2 * XR * XROWB);             // [NCW][128]
  float* sumb = red + NCW * 128;                                           // [128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sumb + 128);                // full, empty, abar[2], afree[2]
  __shared__ int s_E, s_el[32], s_ec[32], s_tok[32][8], s_rt[32][8];
  __shared__ float s_w[32][8], s_z[32][64], s_ssp[64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x, T = p.T;
  long long* mk = p.marks ? p.marks + (size_t)s * 16 : nullptr;
  if (tid == 0 && mk) { mk[0] = gt(); mk[15] = smid(); }
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), abar = sa(bars + 2 * NSLOT), afree = sa(bars + 2 * NSLOT + 2);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    for (int i = 0; i < 2; ++i) { mbar_init(abar + 8 * i, 1); mbar_init(afree + 8 * i, NCW); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 32 * 64; i += NT) (&s_z[0][0])[i] = 0.f;
  for (int i = tid; i < 64; i += NT) s_ssp[i] = 0.f;
  if (warp == 0) {   // routing prologue: lane e collects local expert e; rank by (count desc, index asc)
    int c = 0;
    for (int i = 0; i < T * 8; ++i) {
      const int e = (p.ids[i] - p.lo) & 255;
      if (e == lane) {
        if (c < 8) { s_tok[lane][c] = i >> 3; s_rt[lane][c] = i; s_w[lane][c] = p.wts[i]; }
        ++c;
      }
    }
    int rank = 0;
    for (int j = 0; j < 32; ++j) {
      const int cj = __shfl_sync(FULL, c, j);
      if (cj > c || (cj == c && j < lane)) ++rank;
    }
    const int E = __popc(__ballot_sync(FULL, c > 0));
    int tk[8], rt[8];
    float w[8];
    __syncwarp();
    for (int k = 0; k < 8; ++k) { tk[k] = s_tok[lane][k]; rt[k] = s_rt[lane][k]; w[k] = s_w[lane][k]; }
    __syncwarp();
    for (int k = 0; k < 8; ++k) { s_tok[rank][k] = tk[k]; s_rt[rank][k] = rt[k]; s_w[rank][k] = w[k]; }
    s_el[rank] = lane;
    s_ec[rank] = c < 8 ? c : 8;
    if (lane == 0) s_E = E;
  }
  __syncthreads();
  const int E = s_E;
  const int n13 = (512 * E > s) ? (512 * E - 1 - s) / G + 1 : 0;
  const int nt2 = (s < 256 - G) ? 2 : 1, n2 = nt2 * E, ng = n13 + n2;
  const int nu = (MODE >= 1 && s < 64) ? 8 : 0;
  if (warp == 0) {   // ---------------- producer
    if (lane == 0) {
      const uint64_t pol = evict_first();
      long long wt = 0;
      int it = 0;
      for (int g = 0; g < ng + nu; ++g) {
        const uint8_t* src;
        int nsl;
        uint32_t bytes;
        if (g < n13) {
          const int j = s + G * g;
          src = p.w13 + ((size_t)s_el[j >> 9] * 512 + (j & 511)) * 2 * SLOTB; nsl = 2; bytes = SLOTB;
        } else if (g < ng) {
          const int j2 = g - n13, q = j2 / nt2, tile = (j2 % nt2 == 0) ? s : s + G;
          src = p.w2 + ((size_t)s_el[q] * 256 + tile) * 2 * SLOTB; nsl = 2; bytes = SLOTB;
        } else {
          src = p.unp + ((size_t)s * 8 + (g - ng)) * UST; nsl = 1; bytes = UST;
        }
        for (int h = 0; h < nsl; ++h, ++it) {
          const int slot = it % NSLOT;
          const long long t0 = mk ? gt() : 0;
          mbar_wait(empty + 8 * slot, ((it / NSLOT) & 1) ^ 1, p.err);
          if (mk) wt += gt() - t0;
          mbar_tx(full + 8 * slot, bytes);
          bulk(sa(ring + slot * SLOTB), src + (size_t)h * SLOTB, bytes, full + 8 * slot, pol);
          if (it == 0 && mk) mk[1] = gt();
        }
      }
      if (mk) mk[10] = wt;
    }
  } else if (warp == 1) {   // ---------------- control: one activation phase per expert (W13 x rows, W2 inter rows), tail m rows
    if (MODE >= 1 && lane == 0) {
      const int P = 2 * E + (nu ? 1 : 0);
      for (int ph = 0; ph < P; ++ph) {
        const int b = ph & 1;
        if (ph >= 2) mbar_wait(afree + 8 * b, ((ph - 2) >> 1) & 1, p.err);
        uint8_t* dst = act + (size_t)b * XR * XROWB;
        if (ph < E) {
          const int rows = s_ec[ph] < XR ? s_ec[ph] : XR;
          mbar_tx(abar + 8 * b, rows * 8192);
          for (int r = 0; r < rows; ++r) bulk_n(sa(dst + r * XROWB), p.xe + (size_t)s_tok[ph][r] * 4096, 8192, abar + 8 * b);
        } else if (ph < 2 * E) {
          const int q = ph - E;
          if (q == 0 && mk) mk[3] = gt();
          spin_ge(p.cnt + C13 + s_el[q], 512, p.err);
          if (q == 0 && mk) mk[4] = gt();
          fence_async();
          const int rows = s_ec[q] < XR ? s_ec[q] : XR;
          mbar_tx(abar + 8 * b, rows * 8192);
          for (int r = 0; r < rows; ++r) bulk_n(sa(dst + r * XROWB), p.inter + (size_t)s_rt[q][r] * 4096, 8192, abar + 8 * b);
        } else {
          spin_ge(p.cnt + CM, G, p.err);
          fence_async();
          const int rows = T < XR ? T : XR;
          mbar_tx(abar + 8 * b, rows * 8192);
          for (int r = 0; r < rows; ++r) bulk_n(sa(dst + r * XROWB), p.mo + (size_t)r * 4096, 8192, abar + 8 * b);
        }
      }
    }
  } else {   // ---------------- consumers
    const int cw = warp - 2, g8 = lane >> 2, t4 = lane & 3, ct = tid - 64;
    int it = 0, nst = 0;
    float acc[4];
    for (int g = 0; g < ng; ++g) {
      const bool w2 = g >= n13;
      int q, ph;
      bool first, last;
      if (!w2) {
        const int j = s + G * g;
        q = j >> 9; ph = q;
        first = (g == 0) || (((j - G) >> 9) != q);
        last = (g == n13 - 1) || (((j + G) >> 9) != q);
      } else {
        const int j2 = g - n13;
        q = j2 / nt2; ph = E + q;
        first = (j2 % nt2) == 0;
        last = (j2 % nt2) == nt2 - 1;
      }
      const int b = ph & 1;
      if (MODE >= 1 && first) mbar_wait(abar + 8 * b, (ph >> 1) & 1, p.err);
      acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
      for (int h = 0; h < 2; ++h, ++it) {
        const int slot = it % NSLOT;
        mbar_wait(full + 8 * slot, (it / NSLOT) & 1, p.err);
        if (MODE == 2) {
          const uint8_t* sl = ring + slot * SLOTB;
          const uint8_t* xb = act + (size_t)(b * XR + (g8 % XR)) * XROWB + (size_t)(h * 2048 + 16 * t4) * 2;
          const uint32_t sw2 = *reinterpret_cast<const uint32_t*>(sl + 16384 + cw * 128 + lane * 4);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const int kb = 2 * cw + i;
            const uint4 w = *reinterpret_cast<const uint4*>(sl + kb * 512 + lane * 16);
            const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * i) * 16u + 0x75807580u;
            const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * i) * 16u + 0x75807580u;
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
              uint32_t a[4], bb[4];
              deq4(hf ? w.y : w.x, a);
              deq4(hf ? w.w : w.z, bb);
#pragma unroll
              for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); bb[jj] = bmul2(bb[jj], S8); }
              const uint4 xv = *reinterpret_cast<const uint4*>(xb + (size_t)(kb * 64 + 8 * hf) * 2);
              mma(acc, a[0], bb[0], a[1], bb[1], xv.x, xv.y);
              mma(acc, a[2], bb[2], a[3], bb[3], xv.z, xv.w);
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + 8 * slot);
        ++nst;
      }
      if (MODE >= 1) {   // granule epilogue
        float* rw = red + cw * 128;
        rw[g8 * 8 + 2 * t4] = acc[0]; rw[g8 * 8 + 2 * t4 + 1] = acc[1];
        rw[(g8 + 8) * 8 + 2 * t4] = acc[2]; rw[(g8 + 8) * 8 + 2 * t4 + 1] = acc[3];
        cbar();
        if (ct < 128) {
          float v = 0.f;
#pragma unroll
          for (int k = 0; k < NCW; ++k) v += red[k * 128 + ct];
          sumb[ct] = v;
        }
        cbar();
        const int ec = s_ec[q];
        if (ct < 128) {
          const int row = ct >> 3, tk = ct & 7;
          if (!w2) {
            if (row < 8 && tk < ec) {
              const float gv = sumb[row * 8 + tk], uv = sumb[(row + 8) * 8 + tk];
              const int rt = (s + G * g) & 511;
              p.inter[(size_t)s_rt[q][tk] * 4096 + rt * 8 + row] = __float2bfloat16_rn(gv / (1.f + __expf(-gv)) * uv);
            }
          } else if (tk < ec) {
            const int ti = (g - n13) % nt2;
            const float z = __bfloat162float(__float2bfloat16_rn(sumb[ct])) * 1.0625f;
            s_z[ti * 16 + row][s_rt[q][tk]] = z;
            atomicAdd(&s_ssp[s_rt[q][tk]], z * z);
          }
        }
        if (last) { __syncwarp(); if (lane == 0) mbar_arrive(afree + 8 * b); }
        cbar();
        if (!w2 && ct == 0) { __threadfence(); red_rel(p.cnt + C13 + s_el[q], 1); }
      }
      if (g == n13 - 1 && ct == 0 && mk) mk[2] = gt();
    }
    if (ct == 0 && mk) mk[5] = gt();
    if (MODE >= 1) {   // ---------------- tail
      // A = sum alpha2^2 s1^2 over this SM's output slice (alpha2 = 0.9 here), then the per-route ss partials
      const int c0 = s * 62, c1 = (s == G - 1) ? 8192 : c0 + 62;
      if (ct < T) {
        float a = 0.f;
        for (int c = c0; c < c1; ++c) { const float v = 0.9f * p.s1[(size_t)ct * 8192 + c]; a += v * v; }
        atomicAdd(p.mom + ct * 4, a);
      }
      if (ct < 64) { const float v = s_ssp[ct]; if (v != 0.f) atomicAdd(p.ss + ct, v); }
      cbar();
      if (ct == 0) { __threadfence(); red_rel(p.cnt + C2, 1); spin_ge(p.cnt + C2, G, p.err); if (mk) mk[6] = gt(); }
      cbar();
      for (int i = ct; i < nt2 * 16 * T; i += 32 * NCW) {   // m for own channels
        const int ti = i / (16 * T), rem = i % (16 * T), row = rem / T, t = rem % T;
        float mv = 0.f;
        for (int k = 0; k < 8; ++k) {
          const int r = t * 8 + k;
          const float z = s_z[ti * 16 + row][r];
          if (z != 0.f) mv += p.wts[r] * z * rsqrtf(__ldcg(p.ss + r) * (1.f / 4096.f) + 1e-8f);
        }
        p.mo[(size_t)t * 4096 + (ti ? s + G : s) * 16 + row] = __float2bfloat16_rn(mv);
      }
      cbar();
      if (ct == 0) { __threadfence(); red_rel(p.cnt + CM, 1); }
      if (nu) {   // unpooler tile: m rows staged (phase 2E), 8 stages, null consumers, f moments
        const int ph = 2 * E, b = ph & 1;
        mbar_wait(abar + 8 * b, (ph >> 1) & 1, p.err);
        for (int u = 0; u < nu; ++u, ++it) {
          const int slot = it % NSLOT;
          mbar_wait(full + 8 * slot, (it / NSLOT) & 1, p.err);
          __syncwarp();
          if (lane == 0) mbar_arrive(empty + 8 * slot);
          ++nst;
        }
        cbar();
        if (ct < 3 * T) {
          const int t = ct / 3, j = ct % 3;
          const float f = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(act + (size_t)b * XR * XROWB + 2 * (s * 16 + j)));
          atomicAdd(p.mom + t * 4 + 1 + j, f * f + 1e-3f);
        }
        cbar();
        if (ct == 0) { __threadfence(); red_rel(p.cnt + CF, 1); }
      }
      if (ct == 0) { spin_ge(p.cnt + CF, 64, p.err); if (mk) mk[8] = gt(); }
      cbar();
      const int nc = c1 - c0;
      for (int i = ct; i < nc * T; i += 32 * NCW) {   // final writes of this SM's slice
        const int t = i / nc, c = c0 + i % nc;
        const float A = __ldcg(p.mom + t * 4), B = __ldcg(p.mom + t * 4 + 1), C = __ldcg(p.mom + t * 4 + 2), D = __ldcg(p.mom + t * 4 + 3);
        const float rf = rsqrtf(B * (1.f / 8192.f) + 1e-8f);
        const float rs = rsqrtf((A + 2.f * rf * C + rf * rf * D) * (1.f / 8192.f) + 1e-5f);
        const float sv = 0.9f * p.s1[(size_t)t * 8192 + c];
        p.s2[(size_t)t * 8192 + c] = sv;
        p.out[(size_t)t * 8192 + c] = __float2bfloat16_rn(sv * rs);
      }
    }
    if (ct == 0) { if (p.stages) p.stages[s] = nst; if (mk) mk[7] = gt(); }
  }
  __syncthreads();
  if (MODE >= 1 && tid == 0) {   // the last CTA out resets counters and accumulators
    __threadfence();
    const int old = atomicAdd(p.cnt + CEXIT, 1);
    if (old == G - 1) {
      __threadfence();
      for (int i = 0; i < CEXIT; ++i) p.cnt[i] = 0;
      for (int i = 0; i < 64; ++i) p.ss[i] = 0.f;
      for (int i = 0; i < 32; ++i) p.mom[i] = 0.f;
      __threadfence();
      p.cnt[CEXIT] = 0;
    }
  }
}

// ======================================== C8-F ========================================
struct PF {
  const uint8_t* qkv; const uint8_t* ow; const uint8_t* rw; const uint8_t* pw;   // [8][144][SLOTF], [512][SLOTF], [132][RSL], [4][256][2][SLOTF]
  const uint8_t* kc; const uint8_t* vc; const int* ptab;                          // [npages][128] each, [npages]
  const __nv_bfloat16* x;                                                         // [T][8192]
  float* qacc; float* part; __nv_bfloat16* oin; float* mom1;                      // [T][2304], [8][T][16][66], [T][1024], [8][4]
  float* s1; __nv_bfloat16* hb; float* logits; float* pacc;                       // [T][8192] x2, [T][256], [T][4096]
  int* cnt; int* err; long long* marks; int* stages; unsigned* sink;
  int T, seq; long long spin;
};

template <int NSLOT, int AB, int MODE>
__global__ void __launch_bounds__(NT, 1) c8f_probe(const __grid_constant__ PF p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  uint8_t* act = sm + NSLOT * SLOTF;
  uint64_t* bars = reinterpret_cast<uint64_t*>(act + AB);   // full, empty, abar, afree
  __shared__ float s_mom[8][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x, T = p.T;
  long long* mk = p.marks ? p.marks + (size_t)s * 16 : nullptr;
  if (tid == 0 && mk) { mk[0] = gt(); mk[15] = smid(); }
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), abar = sa(bars + 2 * NSLOT), afree = sa(bars + 2 * NSLOT + 1);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    mbar_init(abar, 1);
    mbar_init(afree, NCW);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (tid < 32) (&s_mom[0][0])[tid] = 0.f;
  __syncthreads();
  const int c = s & 7, nc = (c < 4) ? 17 : 16, u = s >> 3;
  const int nq = (143 - u) / nc + 1;                       // qkv tiles u, u + nc, ... (< 144)
  const int nqv = (79 - u) / nc + 1;                       // of which q/k/v tiles (< 80)
  const int no = (511 - s) / G + 1;                        // o tiles s, s + 132, ... (< 512)
  const int qq = s & 3, u4 = s >> 2;
  const int npl = (255 - u4) / 33 + 1;                     // pooler tiles u4, u4 + 33, ... (< 256)
  const int nsl = nq + no + 3 + 2 * npl;
  if (warp == 0) {   // ---------------- producer
    if (lane == 0) {
      const uint64_t pol = evict_first();
      long long wt = 0;
      for (int i = 0; i < nsl; ++i) {
        const uint8_t* src;
        uint32_t bytes = SLOTF;
        if (i < nq) src = p.qkv + ((size_t)c * 144 + u + nc * i) * SLOTF;
        else if (i < nq + no) src = p.ow + (size_t)(s + G * (i - nq)) * SLOTF;
        else if (i < nq + no + 3) { const int j = i - nq - no; src = p.rw + (size_t)s * RSL + (size_t)j * SLOTF; bytes = j < 2 ? SLOTF : RSL - 2 * SLOTF; }
        else { const int j = i - nq - no - 3; src = p.pw + (((size_t)qq * 256 + u4 + 33 * (j >> 1)) * 2 + (j & 1)) * SLOTF; }
        const int slot = i % NSLOT;
        const long long t0 = mk ? gt() : 0;
        mbar_wait(empty + 8 * slot, ((i / NSLOT) & 1) ^ 1, p.err);
        if (mk) wt += gt() - t0;
        mbar_tx(full + 8 * slot, bytes);
        bulk(sa(ring + slot * SLOTF), src, bytes, full + 8 * slot, pol);
        if (i == 0 && mk) mk[1] = gt();
      }
      if (mk) mk[11] = wt;
    }
  } else if (warp == 1) {   // ---------------- control: x chunk, then o-input (after cnt_oin), then h quarter (after cnt_h)
    if (MODE >= 1 && lane == 0) {
      mbar_tx(abar, T * 2048);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.x + (size_t)t * 8192 + c * 1024, 2048, abar);
      spin_ge(p.cnt + FOIN, 1, p.err);
      if (mk) mk[6] = gt();
      for (int i = s; i < T * 2304; i += G) p.qacc[i] = 0.f;   // qacc is dead after the o-input: zero this SM's share
      mbar_wait(afree, 0, p.err);
      fence_async();
      mbar_tx(abar, T * 2048);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 2048), p.oin + (size_t)t * 1024, 2048, abar);
      spin_ge(p.cnt + FH, G, p.err);
      if (mk) mk[9] = gt();
      mbar_wait(afree, 1, p.err);
      fence_async();
      mbar_tx(abar, T * 4096);
      for (int t = 0; t < T; ++t) bulk_n(sa(act + t * 4096), p.hb + (size_t)t * 8192 + qq * 2048, 4096, abar);
    }
  } else {   // ---------------- consumers
    const int ct = tid - 64;
    uint4 kv[8];
    const bool att = MODE >= 1 && s < 8;
    if (att) {   // KV window rows for this split: 256 keys, K (even threads) or V (odd threads), 128 B rows (both heads)
      const int key = ct >> 1, pos = p.seq - 2048 + s * 256 + key;
      const int pg = p.ptab[pos < 0 ? 0 : pos];
      const uint4* row = reinterpret_cast<const uint4*>(((ct & 1) ? p.vc : p.kc) + (size_t)pg * 128);
#pragma unroll
      for (int k = 0; k < 8; ++k) kv[k] = __ldg(row + k);
    }
    int nst = 0;
    unsigned sink = 0u;
    for (int i = 0; i < nsl; ++i) {
      const int slot = i % NSLOT;
      if (MODE >= 1) {
        if (i == 0) mbar_wait(abar, 0, p.err);                       // x chunk
        if (i == nq) mbar_wait(abar, 1, p.err);                      // o-input
        if (i == nq + no + 3) mbar_wait(abar, 0, p.err);             // h quarter (third completion)
      }
      mbar_wait(full + 8 * slot, (i / NSLOT) & 1, p.err);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty + 8 * slot);
      ++nst;
      if (MODE == 0) continue;
      if (i < nq) {   // qkv granule epilogue: reduction barrier, red.add 16 x T partials, group counter
        const int tile = u + nc * i;
        cbar();
        if (ct < 16 * T) atomicAdd(p.qacc + (size_t)(ct / 16) * 2304 + tile * 16 + (ct & 15), 1e-30f);
        cbar();
        if (ct == 0) { __threadfence(); red_rel(p.cnt + (tile < 40 ? FQ0 : tile < 80 ? FQ1 : FQG), 1); }
        if (att && i == nqv - 1) {   // attention after this SM's last q/k/v tile
          if (ct == 0) { if (mk) mk[2] = gt(); spin_ge(p.cnt + FQ0, 320, p.err); spin_ge(p.cnt + FQ1, 320, p.err); if (mk) mk[3] = gt(); }
          cbar();
          float qs = 0.f;
          for (int k = ct; k < 1280 * T; k += 32 * NCW) qs += __ldcg(p.qacc + (k / 1280) * 2304 + (k % 1280));
#pragma unroll
          for (int k = 0; k < 8; ++k) sink ^= kv[k].x ^ kv[k].y ^ kv[k].z ^ kv[k].w;
          const long long t0 = clock64();
          while (clock64() - t0 < p.spin) {}
          float* pp = p.part + (size_t)s * T * 16 * 66;
          for (int k = ct; k < T * 16 * 66; k += 32 * NCW) pp[k] = qs * 1e-30f + (float)k;
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + FATT, 1); if (mk) mk[4] = gt(); }
        }
        if (att && s == 0 && i == nq - 1) {   // merger: after its gate tiles, the 8 partials and the gate rows -> o-input
          if (ct == 0) { spin_ge(p.cnt + FATT, 8, p.err); spin_ge(p.cnt + FQG, 512, p.err); }
          cbar();
          float acc = 0.f;
          for (int k = ct; k < 8 * T * 16 * 66; k += 32 * NCW) acc += __ldcg(p.part + k);
          for (int k = ct; k < T * 1024; k += 32 * NCW) {
            const float gv = __ldcg(p.qacc + (k / 1024) * 2304 + 1280 + (k % 1024));
            p.oin[k] = __float2bfloat16_rn(acc * 1e-30f + gv);
          }
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + FOIN, 1); if (mk) mk[5] = gt(); }
        }
        if (i == nq - 1) { __syncwarp(); if (lane == 0) mbar_arrive(afree); }   // x chunk released
      } else if (i < nq + no) {   // o granule: moment partials (16 rows x T) into s_mom
        cbar();
        if (ct < 4 * T) atomicAdd(&s_mom[ct >> 2][ct & 3], 1e-3f);
        if (i == nq + no - 1) {   // end of the o phase: moments -> hop, then s1 / h for own channels -> hop
          cbar();
          for (int k = ct; k < T * 256; k += 32 * NCW) if ((k % G) == s) p.logits[k] = 0.f;   // distributed zeroing
          for (int k = s + G * ct; k < T * 4096; k += G * 32 * NCW) p.pacc[k] = 0.f;
          if (ct < 4 * T) atomicAdd(p.mom1 + ct, s_mom[ct >> 2][ct & 3]);
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + FM1, 1); if (mk) mk[7] = gt(); spin_ge(p.cnt + FM1, G, p.err); if (mk) mk[8] = gt(); }
          cbar();
          for (int k = ct; k < no * 16 * T; k += 32 * NCW) {
            const int t = k % T, ch = (s + G * (k / (16 * T))) * 16 + (k / T) % 16;
            const float m0 = __ldcg(p.mom1 + t * 4), m3 = __ldcg(p.mom1 + t * 4 + 3);
            const float v = 0.9f * __bfloat162float(p.x[(size_t)t * 8192 + ch]) + rsqrtf(m0 + m3 + 1e-5f);
            p.s1[(size_t)t * 8192 + ch] = v;
            p.hb[(size_t)t * 8192 + ch] = __float2bfloat16_rn(v);
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(afree);   // o-input released
          cbar();
          if (ct == 0) { __threadfence(); red_rel(p.cnt + FH, 1); }
        }
      } else if (i < nq + no + 3) {   // router slice: partial logits after the last router stage
        if (i == nq + no + 2) {
          cbar();
          if (ct < 64 * T) atomicAdd(reinterpret_cast<float4*>(p.logits) + ct, make_float4(1e-3f, 1e-3f, 1e-3f, 1e-3f));
        }
      } else {   // pooler granule (2 slots): reduction barrier + red.add of 16 x T partials
        const int j = i - nq - no - 3;
        if (j & 1) {
          cbar();
          const int tile = u4 + 33 * (j >> 1);
          if (ct < 16 * T) atomicAdd(p.pacc + (size_t)(ct / 16) * 4096 + tile * 16 + (ct & 15), 1e-3f);
        }
      }
    }
    if (att && sink == 0x12345678u) p.sink[0] = sink;
    if (ct == 0) { if (p.stages) p.stages[s] = nst; if (mk) mk[10] = gt(); }
  }
  __syncthreads();
  if (MODE >= 1 && tid == 0) {
    __threadfence();
    const int old = atomicAdd(p.cnt + FEXIT, 1);
    if (old == G - 1) {
      __threadfence();
      for (int i = 0; i < FEXIT; ++i) p.cnt[i] = 0;
      for (int i = 0; i < 32; ++i) p.mom1[i] = 0.f;
      __threadfence();
      p.cnt[FEXIT] = 0;
    }
  }
}

template <typename K> static void set_smem(K k, int bytes) {
  C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
template <int NSLOT, int XR> constexpr int smem_b() { return NSLOT * SLOTB + 2 * XR * XROWB + NCW * 512 + 512 + (2 * NSLOT + 4) * 8; }
template <int NSLOT, int AB> constexpr int smem_f() { return NSLOT * SLOTF + AB + (2 * NSLOT + 2) * 8; }

template <int NSLOT, int XR, int MODE> static void launch_b(const PB& pb, cudaStream_t st) {
  constexpr int SM = smem_b<NSLOT, XR>();
  static bool once = false;
  if (!once) { set_smem(c8b_probe<NSLOT, XR, MODE>, SM); once = true; }
  c8b_probe<NSLOT, XR, MODE><<<G, NT, SM, st>>>(pb);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NSLOT, int AB, int MODE> static void launch_f(const PF& pf, cudaStream_t st) {
  constexpr int SM = smem_f<NSLOT, AB>();
  static bool once = false;
  if (!once) { set_smem(c8f_probe<NSLOT, AB, MODE>, SM); once = true; }
  c8f_probe<NSLOT, AB, MODE><<<G, NT, SM, st>>>(pf);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <typename K> static void attrs_of(K k, int smem, std::vector<int64_t>& v) {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, k));
  set_smem(k, smem);
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, NT, smem));
  v.push_back(a.numRegs); v.push_back((int64_t)a.localSizeBytes); v.push_back((int64_t)a.sharedSizeBytes + smem); v.push_back(occ);
}
}  // namespace c8p

using namespace c8p;
#define U8(t) reinterpret_cast<const uint8_t*>(t.data_ptr())

void c8b_run(int64_t T, int64_t mode, torch::Tensor w13, torch::Tensor w2, torch::Tensor unp, torch::Tensor xe, torch::Tensor ids,
             torch::Tensor wts, torch::Tensor s1, torch::Tensor inter, torch::Tensor mo, torch::Tensor out, torch::Tensor s2,
             torch::Tensor ss, torch::Tensor mom, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks, torch::Tensor stages,
             int64_t lo, int64_t use_marks) {
  PB pb;
  pb.w13 = U8(w13); pb.w2 = U8(w2); pb.unp = U8(unp);
  pb.xe = reinterpret_cast<const __nv_bfloat16*>(xe.data_ptr()); pb.ids = ids.data_ptr<int>(); pb.wts = wts.data_ptr<float>();
  pb.s1 = s1.data_ptr<float>();
  pb.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); pb.mo = reinterpret_cast<__nv_bfloat16*>(mo.data_ptr());
  pb.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); pb.s2 = s2.data_ptr<float>();
  pb.ss = ss.data_ptr<float>(); pb.mom = mom.data_ptr<float>(); pb.cnt = cnt.data_ptr<int>(); pb.err = err.data_ptr<int>();
  pb.marks = use_marks ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  pb.stages = use_marks ? stages.data_ptr<int>() : nullptr;
  pb.T = (int)T; pb.lo = (int)lo;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (T <= 1) {
    if (mode == 0) launch_b<10, 1, 0>(pb, st); else if (mode == 1) launch_b<10, 1, 1>(pb, st); else launch_b<10, 1, 2>(pb, st);
  } else {
    if (mode == 0) launch_b<7, 4, 0>(pb, st); else if (mode == 1) launch_b<7, 4, 1>(pb, st); else launch_b<7, 4, 2>(pb, st);
  }
}

void c8f_run(int64_t T, int64_t mode, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw, torch::Tensor kc,
             torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor part, torch::Tensor oin,
             torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc, torch::Tensor cnt,
             torch::Tensor err, torch::Tensor marks, torch::Tensor stages, torch::Tensor sink, int64_t seq, int64_t spin,
             int64_t use_marks) {
  PF pf;
  pf.qkv = U8(qkv); pf.ow = U8(ow); pf.rw = U8(rw); pf.pw = U8(pw); pf.kc = U8(kc); pf.vc = U8(vc); pf.ptab = ptab.data_ptr<int>();
  pf.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  pf.qacc = qacc.data_ptr<float>(); pf.part = part.data_ptr<float>(); pf.oin = reinterpret_cast<__nv_bfloat16*>(oin.data_ptr());
  pf.mom1 = mom1.data_ptr<float>(); pf.s1 = s1.data_ptr<float>(); pf.hb = reinterpret_cast<__nv_bfloat16*>(hb.data_ptr());
  pf.logits = logits.data_ptr<float>(); pf.pacc = pacc.data_ptr<float>();
  pf.cnt = cnt.data_ptr<int>(); pf.err = err.data_ptr<int>();
  pf.marks = use_marks ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  pf.stages = use_marks ? stages.data_ptr<int>() : nullptr;
  pf.sink = reinterpret_cast<unsigned*>(sink.data_ptr());
  pf.T = (int)T; pf.seq = (int)seq; pf.spin = spin;
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  if (T <= 1) { if (mode == 0) launch_f<8, 4096, 0>(pf, st); else launch_f<8, 4096, 1>(pf, st); }
  else { if (mode == 0) launch_f<7, 32768, 0>(pf, st); else launch_f<7, 32768, 1>(pf, st); }
}

std::vector<int64_t> c8_attrs() {
  std::vector<int64_t> v;
  attrs_of(c8b_probe<10, 1, 0>, smem_b<10, 1>(), v); attrs_of(c8b_probe<10, 1, 1>, smem_b<10, 1>(), v); attrs_of(c8b_probe<10, 1, 2>, smem_b<10, 1>(), v);
  attrs_of(c8b_probe<7, 4, 0>, smem_b<7, 4>(), v); attrs_of(c8b_probe<7, 4, 1>, smem_b<7, 4>(), v); attrs_of(c8b_probe<7, 4, 2>, smem_b<7, 4>(), v);
  attrs_of(c8f_probe<8, 4096, 0>, smem_f<8, 4096>(), v); attrs_of(c8f_probe<8, 4096, 1>, smem_f<8, 4096>(), v);
  attrs_of(c8f_probe<7, 32768, 0>, smem_f<7, 32768>(), v); attrs_of(c8f_probe<7, 32768, 1>, smem_f<7, 32768>(), v);
  return v;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void c8b_run(int64_t T, int64_t mode, torch::Tensor w13, torch::Tensor w2, torch::Tensor unp, torch::Tensor xe, torch::Tensor ids,
             torch::Tensor wts, torch::Tensor s1, torch::Tensor inter, torch::Tensor mo, torch::Tensor out, torch::Tensor s2,
             torch::Tensor ss, torch::Tensor mom, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks, torch::Tensor stages,
             int64_t lo, int64_t use_marks);
void c8f_run(int64_t T, int64_t mode, torch::Tensor qkv, torch::Tensor ow, torch::Tensor rw, torch::Tensor pw, torch::Tensor kc,
             torch::Tensor vc, torch::Tensor ptab, torch::Tensor x, torch::Tensor qacc, torch::Tensor part, torch::Tensor oin,
             torch::Tensor mom1, torch::Tensor s1, torch::Tensor hb, torch::Tensor logits, torch::Tensor pacc, torch::Tensor cnt,
             torch::Tensor err, torch::Tensor marks, torch::Tensor stages, torch::Tensor sink, int64_t seq, int64_t spin,
             int64_t use_marks);
std::vector<int64_t> c8_attrs();
"""

ATTR_NAMES = ["c8b T1 m0", "c8b T1 m1", "c8b T1 m2", "c8b T8 m0", "c8b T8 m1", "c8b T8 m2",
              "c8f T1 m0", "c8f T1 m1", "c8f T8 m0", "c8f T8 m1"]
CFLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def build(name="c8_probe", verbose=False):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((CPP_SRC + CUDA_SRC).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                      functions=["c8b_run", "c8f_run", "c8_attrs"], extra_cuda_cflags=CFLAGS, verbose=verbose)
    return mod, time.time() - t0


def t8_repack(ws, n_tiles):
    """K2A stage layout [32][items][16 stages][18432] -> C8 granule layout [32][16-row tiles][2 halves][18432]: per half,
    the 8 stages' 2 KB word pieces of the tile (16 KB) then their 256 B scale pieces (2 KB). Pure byte permutation."""
    import torch
    e, items = ws.shape[0], ws.shape[1]
    w = ws.view(e, items, 16, 18432)
    wp = w[..., :16384].reshape(e, items, 16, 8, 2048).permute(0, 1, 3, 2, 4).reshape(e, items, 8, 2, 8 * 2048)
    sp = w[..., 16384:].reshape(e, items, 16, 8, 256).permute(0, 1, 3, 2, 4).reshape(e, items, 8, 2, 8 * 256)
    out = torch.cat([wp, sp], dim=-1).reshape(e, items * 8, 2, 18432).contiguous()
    assert out.shape[1] == n_tiles
    return out


def b_expected_stages(E, mode, s, G=132):
    n13 = len(range(s, 512 * E, G))
    nt2 = 2 if s < 256 - G else 1
    nu = 8 if (mode >= 1 and s < 64) else 0
    return 2 * (n13 + nt2 * E) + nu


def f_expected_stages(s, G=132):
    c, u = s & 7, s >> 3
    nc = 17 if c < 4 else 16
    nq = len(range(u, 144, nc))
    no = len(range(s, 512, G))
    npl = len(range(s >> 2, 256, 33))
    return nq + no + 3 + 2 * npl


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "c8_probe_gen.cu")
    out.write_text(CUDA_SRC, encoding="utf-8")
    print(f"wrote {out} ({len(CUDA_SRC)} chars)")

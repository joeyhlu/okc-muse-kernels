// C8-B core compile check (notes/study/t8_design.md s2.7). NOT the R-t8-1 probe: register / spill / SMEM / SASS budget only.
// Build: MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" nvcc -arch=sm_90a -O3 -std=c++17 -c kernels/lab/t8/c8b_core_check.cu -o /tmp/c8b.o -Xptxas -v
// 132 CTAs x 576 threads: warp 0 producer, warp 1 control, warps 2..17 consumers (split-K 16 per 16-row granule).
#include <cuda_bf16.h>
#include <stdint.h>

namespace c8 {
constexpr int SLOT = 18432;            // 16 rows x 2048 K E2M1 (16 KB) + scales (2 KB)
constexpr int NCW = 16;                // consumer warps
constexpr int NT = 32 * (NCW + 2);     // 576
constexpr int XROW = 4096 * 2 + 64;    // staged activation row (bf16 + pad: rows land on different banks)
constexpr long long BUDGET = 1ll << 31;

struct P {
  const uint8_t* w13;  const uint8_t* w2;   // T8 layout: [32][512 tiles][2 slots][SLOT], [32][256 tiles][2][SLOT]
  const __nv_bfloat16* xe;                  // [T][4096]
  __nv_bfloat16* inter;                     // [64 routes][4096]
  float* y;                                 // [64 routes][4096]
  float* ss;                                // [64]
  int* cnt13; int* cnt2; int* err;          // [32], [1], [4]
  const int* elist; const int* etok;        // local experts [E], tokens per expert [E]
  int E, T;
};

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_tx(uint32_t b, uint32_t n) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(n) : "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory"); }
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n" : "=r"(ok) : "r"(b), "r"(par) : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) if (clock64() - t0 > BUDGET) { atomicExch(err, 1); return; }   // bounded, no trap (R9)
}
__device__ __forceinline__ void bulk(uint32_t dst, const void* src, uint32_t n, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
               :: "r"(dst), "l"(src), "r"(n), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t evict_first() { uint64_t p; asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p)); return p; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rel(int* p, int v) { asm volatile("red.release.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void spin_ge(const int* p, int target, int* err) {
  long long n = 0;
  while (ld_acq(p) < target) { __nanosleep(64); if (++n > (1ll << 24)) { atomicExch(err + 1, 1); return; } }
}
// K2A's BITPOS3 E2M1 -> bf16x2 decode and the bf16x2 scale multiply
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d; asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u)); return d;
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// granule i of SM s: W13 round-robin over E*512 (expert-major), then W2 row tiles r = s, s+132 (< 256) of every expert
__device__ __forceinline__ int n_w13(int s, int E) { const int G = E * 512; return G / 132 + (s < G % 132 ? 1 : 0); }
__device__ __forceinline__ int n_w2(int s, int E) { return E * ((s < 256 - 132) ? 2 : 1); }

template <int NSLOT, int MODE>   // MODE 0: null consumers, 1: + dependencies / epilogues, 2: + decode + mma
__global__ void __launch_bounds__(NT, 1) c8b_core(const __grid_constant__ P p) {
  extern __shared__ __align__(128) uint8_t sm[];
  uint8_t* ring = sm;
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(sm + NSLOT * SLOT);              // [8][XROW/2] activations
  float* red = reinterpret_cast<float*>(sm + NSLOT * SLOT + 8 * XROW);                   // [16 warps][128]
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + NSLOT * SLOT + 8 * XROW + NCW * 512); // full[NSLOT], empty[NSLOT], xready
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, s = blockIdx.x;
  const int E = p.E, n13 = n_w13(s, E), n2 = n_w2(s, E), ng = n13 + n2;
  const uint32_t full = sa(bars), empty = sa(bars + NSLOT), xready = sa(bars + 2 * NSLOT);
  if (tid == 0) {
    for (int i = 0; i < NSLOT; ++i) { mbar_init(full + 8 * i, 1); mbar_init(empty + 8 * i, NCW); }
    mbar_init(xready, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 0) {   // ---- producer: never waits on activations, only on ring space
    if (lane == 0) {
      const uint64_t pol = evict_first();
      for (int g = 0, it = 0; g < ng; ++g) {
        const uint8_t* src;
        if (g < n13) { const int gi = s + 132 * g, e = p.elist[gi / 512], rt = gi % 512; src = p.w13 + ((size_t)e * 512 + rt) * 2 * SLOT; }
        else { const int j = g - n13, rt = s + 132 * (j / E), e = p.elist[j % E]; src = p.w2 + ((size_t)e * 256 + rt) * 2 * SLOT; }
        for (int h = 0; h < 2; ++h, ++it) {
          const int slot = it % NSLOT;
          mbar_wait(empty + 8 * slot, ((it / NSLOT) & 1) ^ 1, p.err);
          mbar_tx(full + 8 * slot, SLOT);
          bulk(sa(ring + slot * SLOT), src + h * SLOT, SLOT, full + 8 * slot, pol);
        }
      }
    }
  } else if (warp == 1) {   // ---- control: W13 -> W2 per-expert dependency, then inter staging (MODE >= 1)
    if (MODE >= 1 && lane == 0) {
      for (int j = 0; j < E && n2 > 0; ++j) {
        spin_ge(p.cnt13 + p.elist[j], 512, p.err);
        // (probe) bulk-copy inter rows of expert j into xs, arrive on xready; here: one arrive per expert
        mbar_arrive(xready);
      }
    }
  } else {   // ---- 16 consumer warps, split-K 16 inside each 16-row granule
    const int cw = warp - 2, g8 = lane >> 2, t4 = lane & 3;
    float acc[4];
    for (int g = 0, it = 0; g < ng; ++g) {
      const bool w2 = g >= n13;
      const int tok = MODE >= 1 ? (w2 ? p.etok[(g - n13) % E] : p.etok[(s + 132 * g) / 512]) : 1;
      if (MODE >= 1 && w2 && (g - n13) % E == 0 && g - n13 < E) mbar_wait(xready, 0, p.err);
      acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
      for (int h = 0; h < 2; ++h, ++it) {
        const int slot = it % NSLOT;
        mbar_wait(full + 8 * slot, (it / NSLOT) & 1, p.err);
        if (MODE == 2) {
          const uint8_t* sl = ring + slot * SLOT;
#pragma unroll
          for (int kb = 0; kb < 2; ++kb) {
            const int k = 2 * cw + kb;                                    // k-block (64 K) of this warp
            const uint4 wv = *reinterpret_cast<const uint4*>(sl + k * 512 + lane * 16);
            const uint32_t sw = *reinterpret_cast<const uint32_t*>(sl + 16384 + (k >> 1) * 128 + lane * 4);
            const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
              uint32_t a[4];
              deq4(ww[kt], a);
              const uint32_t sc = __byte_perm(sw, 0u, (k & 1) ? 0x3232u : 0x1010u);
#pragma unroll
              for (int q = 0; q < 4; ++q) a[q] = bmul2(a[q], sc);
              const int kk = h * 2048 + k * 64 + kt * 16 + 2 * t4;
              uint32_t b0 = 0u, b1 = 0u;
              if (g8 < tok) {
                const __nv_bfloat16* xr = xs + (size_t)g8 * (XROW / 2);
                b0 = *reinterpret_cast<const uint32_t*>(xr + kk);
                b1 = *reinterpret_cast<const uint32_t*>(xr + kk + 8);
              }
              mma(acc, a, b0, b1);
            }
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty + 8 * slot);
      }
      if (MODE >= 1) {   // granule epilogue: cross-warp reduction, then W13 (SwiGLU -> inter, cnt13) or W2 (y, ss)
        float* rw = red + cw * 128;
        rw[g8 * 8 + 2 * t4] = acc[0]; rw[g8 * 8 + 2 * t4 + 1] = acc[1];
        rw[(g8 + 8) * 8 + 2 * t4] = acc[2]; rw[(g8 + 8) * 8 + 2 * t4 + 1] = acc[3];
        asm volatile("bar.sync 1, %0;" :: "n"(32 * NCW) : "memory");
        const int ct = tid - 64;
        if (ct < 128) {
          float v = 0.f;
#pragma unroll
          for (int q = 0; q < NCW; ++q) v += red[q * 128 + ct];
          const int row = ct >> 3, tk = ct & 7;
          if (!w2) {
            const float u = __shfl_down_sync(0xffffffffu, v, 0);   // (probe) gate/up pairing of rows r and r+8
            if (row < 8 && tk < tok) {
              const float gl = v, sw = gl / (1.f + __expf(-gl)) * u;
              const int gi = s + 132 * g, rt = gi % 512;
              p.inter[(size_t)tk * 4096 + rt * 8 + row] = __float2bfloat16_rn(sw);
            }
          } else if (tk < tok) {
            const int j = g - n13, rt = s + 132 * (j / E);
            const float yb = __bfloat162float(__float2bfloat16_rn(v));
            p.y[(size_t)tk * 4096 + rt * 16 + row] = v;
            atomicAdd(p.ss + (j % E) * 8 + tk, yb * yb);
          }
        }
        asm volatile("bar.sync 1, %0;" :: "n"(32 * NCW) : "memory");
        if (ct == 0) {
          __threadfence();
          if (!w2) red_rel(p.cnt13 + p.elist[(s + 132 * g) / 512], 1);
          else if (g == ng - 1) red_rel(p.cnt2, 1);
        }
      }
    }
  }
}
template __global__ void c8b_core<10, 0>(const __grid_constant__ P);
template __global__ void c8b_core<10, 1>(const __grid_constant__ P);
template __global__ void c8b_core<10, 2>(const __grid_constant__ P);
template __global__ void c8b_core<6, 2>(const __grid_constant__ P);
template __global__ void c8b_core<7, 2>(const __grid_constant__ P);
}  // namespace c8

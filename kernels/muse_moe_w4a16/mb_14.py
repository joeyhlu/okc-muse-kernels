# okc: problem=muse_moe_w4a16
"""mb_14 (mb_13 + tools/vsh_patches.py on the lv source: expert-strided VMS shares arms lvsh / lvshf1 / lvsh8; lvf4/5/7 dropped). mb_13 (tools/mb_gen13.py: mb_12 + layer-VMS VMS_FX arms lvf1/4/5/7, w72rh dropped). mb_12 (tools/mb_gen12.py: mb_11 + VM_RH release-helper arm w72rh). mb_11 (tools/mb_gen11.py: mb_10 + board VML cost-table arms Lx<pct>). mb_10 (tools/mb_gen10.py: mb_9 + k3_b23_1 VML + layer VMS arm lv + VMS MAXT-32 probe s32 at bucket 32). mb_9 (tools/mb_gen_gc7.py: mb_8 + k3_gc_7 VML bf2 / w72 + UNP L2 stage layout): MoE board loop file (tools/mb_gen.py; notes/study/moe_board_loop.md). VMS (one-launch mma.sync, jobs of
<= 8 tokens) + VML (one-launch RS-wgmma stream-K) with an in-prepare variant sweep per bucket on the judge GPU;
our own implementations of the public vjeux structure (ideas only). Check only, never scored."""
import hashlib
import time

import torch

# ======== K2A v3.1 prepack helpers (VML's ws13 / ws2 stage layouts; verbatim from vml_2.py)
K2A_BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))

K2A31_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (28, 29, 30, 5), (12, 13, 14, 21),
           (25, 26, 27, 2), (9, 10, 11, 18), (4, 0, 1, 3), (20, 16, 17, 19))

_K2A_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def _k2a_reencode_scales(sf_u8: torch.Tensor):
    """E4M3 bytes -> (b' uint8, negative mask, zero mask, has_nan). b' = (e+10)<<3 | f for |sf| = 2^e (1+f/8);
    every finite E4M3 magnitude (normal and subnormal) is exact. Zero scales get b' = 24 (their codes are
    zeroed), negative scales have their codes' signs flipped by the caller. (Unchanged from v0/v1.)"""
    b = sf_u8.to(torch.int32)
    sign, ex, m = b >> 7, (b >> 3) & 15, b & 7
    nan = (ex == 15) & (m == 7)
    zero = (ex == 0) & (m == 0)
    sub = (ex == 0) & (m != 0)
    lut_e5 = torch.tensor([0, 1, 2, 2, 3, 3, 3, 3], dtype=torch.int32, device=b.device)
    lut_f = torch.tensor([0, 0, 0, 4, 0, 2, 4, 6], dtype=torch.int32, device=b.device)
    bp = ((ex + 3) << 3) | m
    bp = torch.where(sub, (lut_e5[m.long()] << 3) | lut_f[m.long()], bp)
    bp = torch.where(zero, torch.full_like(bp, 24), bp)
    return bp.to(torch.uint8), (sign == 1) & ~zero, zero, bool(nan.any())

def _k2a_tile_row_perm(n_rows: int, gate_up: bool, device=None) -> torch.Tensor:
    """prepacked row m (= 16*tile + rho) -> original row. W13: rho < 8 -> gate 8*tile + rho, rho >= 8 -> up."""
    m = torch.arange(n_rows, device=device)
    if not gate_up:
        return m
    tile, rho = m // 16, m % 16
    return (rho // 8) * (n_rows // 2) + 8 * tile + (rho % 8)

def _k2a31_pack_words(codes: torch.Tensor, bitpos=K2A_BITPOS) -> torch.Tensor:
    """codes [..., 8] (0..15) -> int32 [...] per bitpos (K2A_BITPOS: v2.2, K2A31_BITPOS3: v3.1)."""
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(bitpos):
        ci = c[..., i]
        w |= ((ci & 1) << p0) | (((ci >> 1) & 1) << p1) | (((ci >> 2) & 1) << p2) | (((ci >> 3) & 1) << ps)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)

def _k2a31_prepack_expert(w: torch.Tensor, sf: torch.Tensor, gate_up: bool, bitpos=K2A_BITPOS):
    """w uint8 [N, K/2] (E2M1, low nibble first), sf e4m3 [N, K/16] ->
    (wq int32 [N/16, K/64, 32, 4], sq uint8 [N/16, K/128, 32, 4])."""
    n, k2 = w.shape
    k = 2 * k2
    rt, nkb = n // 16, k // 64
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = _k2a_reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale in NVFP4 weights")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(n, k // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(n, k)
    perm = _k2a_tile_row_perm(n, gate_up, device=w.device)
    codes = codes[perm]
    bp = bp[perm]
    # codes[16*r + 8*hi + g][64*kb + 16*t + 8*half + i] -> (r, kb, g, t, hi, half, i)
    c = codes.view(rt, 2, 8, nkb, 4, 2, 8).permute(0, 3, 2, 4, 1, 5, 6).reshape(rt, nkb, 32, 4, 8)
    wq = _k2a31_pack_words(c, bitpos)
    # bp[16*r + 8*hi + g][group 8*kp + 4*par + t] -> (r, kp, g, t, par, hi): byte 2*par + hi
    sq = bp.view(rt, 2, 8, nkb // 2, 2, 4).permute(0, 3, 2, 5, 4, 1).reshape(rt, nkb // 2, 32, 4)
    return wq.contiguous(), sq.contiguous()

def _k2a33_stage_repack(wq: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
    """v2.2 (wq int32 [rt, 64, 32, 4], sq uint8 [rt, 32, 32, 4]) -> uint8 [rt/8, 16, 18432]: stage (rb, s) =
    [tile w 8][kb i 4][lane][16 B] = wq[8rb+w][4s+i] then [w 8][kp 2][lane][4 B] = sq[8rb+w][2s+kp]. Lossless."""
    rt = wq.shape[0]
    nrb = rt // 8
    wb = wq.contiguous().view(torch.uint8).view(nrb, 8, 16, 4, 512).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 16384)
    sb = sq.contiguous().view(nrb, 8, 16, 2, 128).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 2048)
    return torch.cat((wb, sb), dim=-1).contiguous()

def _k2a_dequant(packed, sf, gs):
    grid = torch.tensor(_K2A_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)

def _k2a_rel_err(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row

# ======== VMS (kernels/lab/vms_moe_lab.py CUDA, + VMS_MAXT)
MB_VMS_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef VMS_NS
#define VMS_NS 3
#endif
#ifndef VMS_G
#define VMS_G 132
#endif
#ifndef VMS_NW
#define VMS_NW 8
#endif
#ifndef VMS_FL
#define VMS_FL 0
#endif
#ifndef VMS_MAXT
#define VMS_MAXT 16
#endif
#ifndef VMS_PAD
#define VMS_PAD 0
#endif
#define VMS_X2S ((VMS_FL & 1024) != 0)   // mb: FL bit 1024 = smem second x tile (with FL 128)
#ifndef VMS_WT
#define VMS_WT 0   // mb: > 0 = cost-weighted CTA unit ranges, a 2-tile unit weighs VMS_WT / 4
#endif
#define VMS_DW ((VMS_FL & 64) != 0)
#define VMS_SKT ((VMS_FL & 128) ? 16 : 8)
#define VMS_CLK ((VMS_FL & 256) != 0)
#define VMS_WSBL ((VMS_FL & 512) != 0 && !VMS_DW && VMS_SKT == 8)
#define VMS_STAG ((VMS_FL & 16) ? 1 : (VMS_FL & 32) ? 2 : VMS_NS)

namespace vms {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = VMS_MAXT * 8, MAXJ = 64, NRT = VMS_MAXT / 4;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = VMS_NW, NCH = 64 / VMS_NW, NT = NW * 32, NTL = NT + 32, NRB = 4, NS = VMS_NS, G = VMS_G;
constexpr int WAIT_MAX = 1 << 22;

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* tw;
  const unsigned char* wA; const unsigned char* wB;
  const float* gsA; const float* gsB; const float* nw;
  __nv_bfloat16* inter; float* ybuf; float* ss; int* ctr;
  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;
  int fl;   // unused (flags are the compile-time VMS_FL: 1 L2 probe source, 2 null consumer, 4 / 8 next-job L1 prefetch)
};

struct __align__(16) Smem {
  uint64_t full[NS];
  uint64_t pfull[NRB];
  uint64_t pempty[NRB];
  int njobs, gen;
  int base[NRT][NE];
  int jE[MAXJ], jS[MAXJ], jC[MAXJ];
  int ent[MAXP], pos[MAXP];
  float4 red[NRB][NW][32];
#if VMS_SKT == 16
  float4 red2[NRB][NW][32];
#endif
#if VMS_PAD > 0
  unsigned char pad[VMS_PAD];   // mb probe: smem size only
#endif
#if VMS_SKT == 16 && VMS_X2S
  uint4 x2s[NW][2 * NCH][32];   // mb VMS_X2S: second x tile per lane
#endif
#if VMS_WT
  int jW[MAXJ + 1];   // mb VMS_WT: per-job weight prefix (quarter units per unit)
  int rng[4];   // mb VMS_WT: this CTA's a0, a1, b0, b1
#endif
};
constexpr int RING_OFF = (int)((sizeof(Smem) + 127) / 128 * 128);
constexpr int SMEM_BYTES = RING_OFF + NS * UB;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(su32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#endif
}
__device__ __forceinline__ int ld_acq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int ld_rlx(const int* p) {
  int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void cta_sync_compute() { asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory"); }

// unit q of this CTA's sequence (phase-A units a0.., then phase-B units b0..) -> global source
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  if (VMS_FL & 1) return p.wA + (size_t)(blockIdx.x & 7) * UB;
  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}

// sm_90: thread 0 only (bulk copy + expect_tx); sm_86: every compute thread (cp.async + arrive.noinc, count NT)
__device__ __forceinline__ void issue(const unsigned char* src, unsigned char* dst, uint64_t* bar, int tid, int nthr = NT) {
#if __CUDA_ARCH__ >= 900
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
#else
  for (int i = tid; i < UB / 16; i += nthr)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}

// e4m3 byte b (0..3) of w -> bf16x2(scale * 2^64) in both halves; exact for bytes 0x08..0x7E (prepare checks)
__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {
  return __byte_perm(w, 0u, sel) * k16 + 0x5C005C00u;
}
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}


// one unit for one warp: dequant each word once, 1 or 2 n-tiles (TWO: tokens 8..15 from x2)
template <bool TWO>
__device__ __forceinline__ void consume(const unsigned char* slot, const uint4 sc, const uint4 (&xr)[2 * NCH],
                                        const uint4 (&x2)[2 * NCH], float (&acc)[4][4], float (&acc2)[4][4],
                                        uint32_t k16, int warp, int lane) {
#pragma unroll
  for (int cc = 0; cc < NCH; ++cc) {
    const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + cc) * 512 + lane * 16);
    const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
    const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
    const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                            xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t qw = ww[s];
      const uint32_t M = 0x81C081C0u;
      const uint32_t r0 = qw & M;
      const uint32_t r1 = (qw * 16u) & M;
      const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
      const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
      const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);
      mma(acc[TWO ? (s & 1) : s], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);   // TWO: 2 chains per tile (register budget)
      if (TWO) {
        const uint32_t b0 = s < 2 ? (s == 0 ? x2[2 * cc].x : x2[2 * cc].z) : (s == 2 ? x2[2 * cc + 1].x : x2[2 * cc + 1].z);
        const uint32_t b1 = s < 2 ? (s == 0 ? x2[2 * cc].y : x2[2 * cc].w) : (s == 2 ? x2[2 * cc + 1].y : x2[2 * cc + 1].w);
        mma(acc2[s & 1], A0, A1, A2, A3, b0, b1);
      }
    }
  }
}


#if VMS_X2S
// mb VMS_X2S: one unit for one warp, 2 n-tiles: tokens 0-7 from x registers, tokens 8-15 from this lane's smem slot
__device__ __forceinline__ void consume2s(const unsigned char* slot, const uint4 sc, const uint4 (&xr)[2 * NCH],
                                          const uint4* x2p, float (&acc)[4][4], uint32_t k16, int warp, int lane) {
#pragma unroll
  for (int cc = 0; cc < NCH; ++cc) {
    const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + cc) * 512 + lane * 16);
    const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
    const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
    const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                            xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
    const uint4 y0 = x2p[(2 * cc) * 32], y1 = x2p[(2 * cc + 1) * 32];
    const uint32_t b2[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t qw = ww[s];
      const uint32_t M = 0x81C081C0u;
      const uint32_t r0 = qw & M;
      const uint32_t r1 = (qw * 16u) & M;
      const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
      const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
      const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);
      mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);
      mma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);
    }
  }
}
#endif

#if VMS_WT
// mb VMS_WT: the unit index where CTA boundary c (0..G) falls when a unit of a 1-tile job weighs 4 and a unit of a 2-tile
// job VMS_WT quarter units (S.jW = per-job weight prefix); R = units per job in this phase (512 A / 256 B). Boundaries are
// monotone in c and the same function for c and c + 1, so the CTA ranges partition [0, njobs x R) exactly.
__device__ __forceinline__ int wt_inv(const Smem& S, int c, int R) {
  const int nj = S.njobs;
  if (c >= G) return nj * R;
  const int tg = (S.jW[nj] * R * c) / G;
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (S.jW[mid] * R <= tg) lo = mid; else hi = mid; }
  const int w = S.jW[lo + 1] - S.jW[lo], off = tg - S.jW[lo] * R;
  const int u = (w == 4) ? ((off + 3) >> 2) : ((off + VMS_WT - 1) / VMS_WT);
  return min(lo * R + u, (lo + 1) * R);
}
#endif
#ifndef VMS_LB
#define VMS_LB 1
#endif
#if VMS_LB
__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {
#else
__global__ void vms_kernel(Params p) {   // mb VMS_LB 0: no launch bounds, so -maxrregcount caps the registers
#endif
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const bool mk = p.marks != nullptr;
  long long* mrow = p.marks + (size_t)c * 64;
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], VMS_DW ? 32 : NT);
#endif
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], VMS_DW ? NW : 1); mbar_init(&S.pempty[i], 1); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  // ---- routing (warp 0, redundant per CTA): pairs sorted by expert, jobs of <= 8 pairs
  if (warp == 0) {
    const int m8 = p.m * 8;
    int idv[NRT]; unsigned mm[NRT];
#pragma unroll
    for (int r = 0; r < NRT; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }
    if (tid == 0) S.gen = ld_rlx(p.ctr);
#pragma unroll
    for (int r = 0; r < NRT; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < NRT; ++r) {
      mm[r] = __match_any_sync(0xffffffffu, idv[r]);
      if (idv[r] >= 0 && lane == __ffs(mm[r]) - 1) S.base[r][idv[r]] = __popc(mm[r]);
    }
    __syncwarp();
    int cr[NRT], tot = 0;
#pragma unroll
    for (int r = 0; r < NRT; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }
    int inc = tot, nj = (tot + VMS_SKT - 1) / VMS_SKT, jinc = nj;
#if VMS_WT
    const int wsum = nj ? (nj - 1) * (VMS_SKT > 8 ? VMS_WT : 4) + ((tot - VMS_SKT * (nj - 1)) > 8 ? VMS_WT : 4) : 0;
    int winc = wsum;
#endif
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
#if VMS_WT
      const int cw = __shfl_up_sync(0xffffffffu, winc, d);
      if (lane >= d) winc += cw;
#endif
      if (lane >= d) { inc += a; jinc += b; }
    }
    const int off = inc - tot, joff = jinc - nj;
    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + VMS_SKT * k; S.jC[joff + k] = min(VMS_SKT, tot - VMS_SKT * k); }
#if VMS_WT
    { int wr = winc - wsum; for (int k = 0; k < nj; ++k) { wr += (min(VMS_SKT, tot - VMS_SKT * k) > 8) ? VMS_WT : 4; S.jW[joff + k + 1] = wr; } if (lane == 0) S.jW[0] = 0; }
#endif
    int run = off;
#pragma unroll
    for (int r = 0; r < NRT; ++r) { S.base[r][lane] = run; run += cr[r]; }
    if (lane == 31) S.njobs = jinc;
    __syncwarp();
#if VMS_WT
    if (lane < 4) S.rng[lane] = wt_inv(S, c + (lane & 1), (lane < 2) ? RBA : RBB);   // a0, a1, b0, b1
#endif
    const unsigned lt = (1u << lane) - 1u;
#pragma unroll
    for (int r = 0; r < NRT; ++r) {
      const int i = r * 32 + lane;
      if (i < m8) {
        if (idv[r] >= 0) { const int pr = S.base[r][idv[r]] + __popc(mm[r] & lt); S.ent[pr] = i; S.pos[i] = pr; }
        else S.pos[i] = -1;
      }
    }
  }
  __syncthreads();
  const int njobs = S.njobs;
  const int NA = njobs * RBA, NN = njobs * (RBA + RBB);
#if VMS_WT
  const int a0 = S.rng[0], a1 = S.rng[1];
  const int b0 = S.rng[2], b1 = S.rng[3];
  const int nA = a1 - a0, N = nA + (b1 - b0);
  (void)NN;
#else
  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;
  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;
  const int b0 = t0 - a0;
  const int nA = a1 - a0, N = t1 - t0;
#endif
  if (mk && tid == 0) { mrow[1] = gt(); mrow[10] = N; }
  if (warp < NW) {
    // ================= compute warps
#if !(VMS_DW && __CUDA_ARCH__ < 900)
    for (int q = 0; q < VMS_STAG && q < N; ++q) {
#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);
    }
#endif
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[2 * NCH];
#if VMS_SKT == 16 && VMS_X2S
    uint4* x2p = &S.x2s[warp][0][lane];
#define X2(k) x2p[(k) * 32]
    bool two = false;
#elif VMS_SKT == 16
    uint4 x2[2 * NCH];
#define X2(k) x2[k]
    bool two = false;
#endif
    int curj = -1;
    int sl = 0; uint32_t par = 0;
#if VMS_CLK
    long long ck[5] = {0, 0, 0, 0, 0};
#endif
    for (int q = 0; q < N; ++q) {
      const bool PB = q >= nA;
      const int v = PB ? b0 + (q - nA) : a0 + q;
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
#if VMS_CLK
        const long long tsw = clock64();
#endif
        curj = j;
        const int jj = PB ? j - 4096 : j;
        const int cnt = S.jC[jj], st = S.jS[jj];
        if (PB) {
          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();
          if (lane == 0) {
            int n = 0;
            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();
        }
        if (g < cnt) {
          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)
                                : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD);
          src += warp * (8 * NCH) + 2 * t;
#pragma unroll
          for (int cc = 0; cc < NCH; ++cc) {
            xr[2 * cc] = PB ? ((VMS_FL & 8) ? __ldca(src + cc * 8) : __ldcg(src + cc * 8)) : __ldg(src + cc * 8);
            xr[2 * cc + 1] = PB ? ((VMS_FL & 8) ? __ldca(src + cc * 8 + 1) : __ldcg(src + cc * 8 + 1)) : __ldg(src + cc * 8 + 1);
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 2 * NCH; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
#if VMS_SKT == 16
        two = cnt > 8;
        if (two) {
          if (g + 8 < cnt) {
            const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + 8 + g) * ID)
                                  : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + 8 + g] >> 3) * HD);
            src += warp * (8 * NCH) + 2 * t;
#pragma unroll
            for (int cc = 0; cc < NCH; ++cc) {
              X2(2 * cc) = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);
              X2(2 * cc + 1) = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);
            }
          } else {
#pragma unroll
            for (int cc = 0; cc < 2 * NCH; ++cc) X2(cc) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
#endif
#if VMS_CLK
        ck[4] += clock64() - tsw;
#endif
      }
      if ((VMS_FL & 12) && q + 1 < N) {   // the next unit starts a new job: pull its x / inter fragments into L1 now
        const bool PB1 = q + 1 >= nA;
        const int v1 = PB1 ? b0 + (q + 1 - nA) : a0 + q + 1;
        const int j1 = PB1 ? (v1 >> 8) + 4096 : (v1 >> 9);
        if (j1 != curj) {
          const int jj1 = PB1 ? j1 - 4096 : j1;
          bool okp = g < S.jC[jj1];
          if (PB1) okp = okp && (VMS_FL & 8) && ld_acq(p.ctr + 8 + jj1) >= RBA;
          else okp = okp && (VMS_FL & 4);
          if (okp) {
            const unsigned char* row = PB1 ? reinterpret_cast<const unsigned char*>(p.inter + (size_t)(S.jS[jj1] + g) * ID)
                                           : reinterpret_cast<const unsigned char*>(p.x + (size_t)(S.ent[S.jS[jj1] + g] >> 3) * HD);
            row += warp * (128 * NCH) + t * (32 * NCH);
            asm volatile("prefetch.global.L1 [%0];\n" :: "l"(row));
            if (NCH == 8) asm volatile("prefetch.global.L1 [%0];\n" :: "l"(row + 128));
          }
        }
      }
#if VMS_CLK
      const long long tq0 = clock64();
#endif
      mbar_wait(&S.full[sl], par);
#if VMS_CLK
      const long long tq1 = clock64();
      ck[0] += tq1 - tq0;
#endif
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
#if VMS_STAG < VMS_NS && !(VMS_DW && __CUDA_ARCH__ < 900)
      if (q == 0) {
        for (int q2 = VMS_STAG; q2 < NS && q2 < N; ++q2) {
#if __CUDA_ARCH__ >= 900
          if (tid == 0)
#endif
          issue(unit_src(p, S, q2, nA, a0, b0), ring + q2 * UB, &S.full[q2], tid);
        }
      }
#endif
      const unsigned char* slot = ring + sl * UB;
#if VMS_NW == 16
      const uint2 sc2 = *reinterpret_cast<const uint2*>(slot + UC + (warp >> 1) * 512 + lane * 16 + (warp & 1) * 8);
      const uint4 sc = make_uint4(sc2.x, sc2.y, 0u, 0u);
#else
      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);
#endif
      float acc[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#if VMS_SKT == 16 && !VMS_X2S
      float acc2[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc2[s][0] = 0.f; acc2[s][1] = 0.f; acc2[s][2] = 0.f; acc2[s][3] = 0.f; }
      if (two) consume<true>(slot, sc, xr, x2, acc, acc2, k16, warp, lane);
      else consume<false>(slot, sc, xr, x2, acc, acc2, k16, warp, lane);
#elif !(VMS_FL & 2)
#if VMS_X2S
      if (two) consume2s(slot, sc, xr, x2p, acc, k16, warp, lane);
      else {
#endif
#pragma unroll
      for (int cc = 0; cc < NCH; ++cc) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + cc) * 512 + lane * 16);
        const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
        const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
        const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                                xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t qw = ww[s];
          const uint32_t M = 0x81C081C0u;
          const uint32_t r0 = qw & M;
          const uint32_t r1 = (qw * 16u) & M;
          const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
          const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
#if VMS_X2S
      }
#endif
#endif
      float4 o;
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
#if VMS_X2S
      if (two) { o.x = acc[0][0] + acc[1][0]; o.y = acc[0][1] + acc[1][1]; o.z = acc[0][2] + acc[1][2]; o.w = acc[0][3] + acc[1][3]; }
#endif
      const int bq = q & (NRB - 1);
#if VMS_CLK
      const long long tq2 = clock64();
      ck[1] += tq2 - tq1;
#endif
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
#if VMS_CLK
      const long long tq3 = clock64();
      ck[2] += tq3 - tq2;
#endif
      S.red[bq][warp][lane] = o;
#if VMS_SKT == 16
      if (two) {
        float4 o2;
#if VMS_X2S
        o2.x = acc[2][0] + acc[3][0]; o2.y = acc[2][1] + acc[3][1]; o2.z = acc[2][2] + acc[3][2]; o2.w = acc[2][3] + acc[3][3];
#else
        o2.x = acc2[0][0] + acc2[1][0] + acc2[2][0] + acc2[3][0];
        o2.y = acc2[0][1] + acc2[1][1] + acc2[2][1] + acc2[3][1];
        o2.z = acc2[0][2] + acc2[1][2] + acc2[2][2] + acc2[3][2];
        o2.w = acc2[0][3] + acc2[1][3] + acc2[2][3] + acc2[3][3];
#endif
        S.red2[bq][warp][lane] = o2;
      }
#endif
#if VMS_DW
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);   // the helper refills the slot once all NW warps are in
#else
      cta_sync_compute();
#if VMS_CLK
      ck[3] += clock64() - tq3;
#endif
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);
      }
#if VMS_WSBL
      if (q == N - 1 && PB) {
        if (warp == 0) {
          const int g = lane >> 2, t = lane & 3;
          float4 sm = S.red[bq][0][lane];
#pragma unroll
          for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
          const int vv = b0 + (q - nA), jb = vv >> 8, rb = vv & 255;
          const int cnt = S.jC[jb], st = S.jS[jb];
          const float gs = __ldg(p.gsB + S.jE[jb]);
          const int n0 = rb * 16 + g;
          const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
          float* ssc = p.ss + (S.gen & 1) * MAXP;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int tk = 2 * t + h;
            const float y0 = (h ? sm.y : sm.x) * w0, y1 = (h ? sm.w : sm.z) * w1;
            float sq = y0 * y0 + y1 * y1;
            if (tk < cnt) {
              p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
              p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
            }
            sq += __shfl_xor_sync(0xffffffffu, sq, 4);
            sq += __shfl_xor_sync(0xffffffffu, sq, 8);
            sq += __shfl_xor_sync(0xffffffffu, sq, 16);
            if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
          }
          __threadfence();
        }
      } else if (tid == 0) mbar_arrive(&S.pfull[bq]);
#else
      if (tid == 0) mbar_arrive(&S.pfull[bq]);
#endif
#endif
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    if (mk && tid == 0) mrow[5] = gt();
#if VMS_CLK
    if (mk && lane == 0) for (int k = 0; k < 5; ++k) mrow[16 + warp * 5 + k] = ck[k];
#endif
  } else {
    // ================= helper warp: per-unit K reduction + epilogue
    const int g = lane >> 2, t = lane & 3;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
#if VMS_DW && __CUDA_ARCH__ < 900
    for (int q = 0; q < NS && q < N; ++q) issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], lane, 32);
#endif
    int hsl = 0;
#if VMS_WSBL
    const int NH = (N > nA) ? N - 1 : N;
#else
    const int NH = N;
#endif
#if VMS_CLK
    long long hk[3] = {0, 0, 0};
    long long th0 = clock64();
#endif
    for (int q = 0; q < NH; ++q) {
      const int bq = q & (NRB - 1);
#if VMS_CLK
      const long long th1 = clock64();
      hk[1] += th1 - th0;
#endif
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
#if VMS_CLK
      th0 = clock64();
      hk[0] += th0 - th1;
#endif
#if VMS_DW
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (lane == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + hsl * UB, &S.full[hsl], lane, 32);
      }
#endif
      if (++hsl == NS) hsl = 0;
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
#if VMS_SKT == 16
      const int jq = q < nA ? ((a0 + q) >> 9) : ((b0 + (q - nA)) >> 8);
      const int ntile = S.jC[jq] > 8 ? 2 : 1;
      float4 sm2 = make_float4(0.f, 0.f, 0.f, 0.f);
      if (ntile == 2) {
#pragma unroll
        for (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }
      }
#else
      const int ntile = 1;
      const float4 sm2 = sm;
#endif
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (q < nA) {
        const int v = a0 + q, j = v >> 9, rb = v & 511;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsA + S.jE[j]);
        for (int tile = 0; tile < ntile; ++tile)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 8 * tile + 2 * t + h;
          const float4 ms = tile ? sm2 : sm;
          if (tk < cnt) {
            const float ga = (h ? ms.y : ms.x) * gs, up = (h ? ms.w : ms.z) * gs;
            const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
          }
        }
        if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units
          __threadfence();
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));
          if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
        }
      } else {
        const int v = b0 + (q - nA), j = v >> 8, rb = v & 255;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsB + S.jE[j]);
        const int n0 = rb * 16 + g;
        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
        for (int tile = 0; tile < ntile; ++tile)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 8 * tile + 2 * t + h;
          const float4 ms = tile ? sm2 : sm;
          const float y0 = (h ? ms.y : ms.x) * w0, y1 = (h ? ms.w : ms.z) * w1;
          float sq = y0 * y0 + y1 * y1;
          if (tk < cnt) {
            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
        }
      }
    }
#if VMS_CLK
    const long long th2 = clock64();
    hk[1] += th2 - th0;
#endif
    __threadfence();
#if VMS_CLK
    hk[2] = clock64() - th2;
    if (mk && lane == 0) for (int k = 0; k < 3; ++k) mrow[56 + k] = hk[k];
#endif
    if (mk && lane == 0) mrow[7] = gt();
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const int gen = S.gen;
  if (tid == 0) {
    __threadfence();
    atomicAdd(p.ctr + 1, 1);
    const int target = (gen + 1) * G;
    int n = 0;
    while (ld_acq(p.ctr + 1) < target) {
      if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 2); break; }
      __nanosleep(32);
    }
    if (mk) mrow[8] = gt();
  }
  __syncthreads();
  if (c == 0) {   // reset for the next launch: every CTA has passed its doneA waits and read gen
    if (tid < MAXJ) p.ctr[8 + tid] = 0;
    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;
    if (tid == 0) atomicAdd(p.ctr, 1);
  }
  // ---- final pass: out[t][col] = sum_s w * rsqrt(ss / HD + eps) * y ; item = (token, quarter of the columns)
  const float* ssc = p.ss + (gen & 1) * MAXP;
  for (int it = c; it < p.m * 4; it += G) {
    const int tk = it >> 2, col = (it & 3) * 1024 + (tid & 255) * 4;
    if (tid >= 256) break;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int pr = S.pos[tk * 8 + s];
      if (pr >= 0) {
        const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(__ldcg(ssc + pr) * (1.f / HD) + p.eps);
        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + col));
        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y), o1 = __floats2bfloat162_rn(a.z, a.w);
    uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = pk;
  }
  if (mk && tid == 0) mrow[9] = gt();
}
}  // namespace vms

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo, int64_t fl) {
  TORCH_CHECK(m >= 1 && m <= VMS_MAXT, "vms: 1 <= m <= VMS_MAXT");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vms::vms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vms::SMEM_BYTES));
    attr = true;
  }
  vms::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.ids = ids.data_ptr<int>(); p.tw = tw.data_ptr<float>();
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo; p.fl = (int)fl;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vms::G); lc.blockDim = dim3(vms::NTL); lc.dynamicSmemBytes = vms::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vms::vms_kernel, p));
}

std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, vms::vms_kernel));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)vms::SMEM_BYTES, vms::G, vms::NS};
}
'''
MB_VMS_CPP = r'''
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo, int64_t fl);
std::vector<int64_t> attrs();
'''
MB_VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((2, 3, 4, 11), (18, 19, 20, 27)), ((28, 29, 30, 5), (12, 13, 14, 21)), ((0, 1, 26, 9), (16, 17, 10, 25)))

# ======== VML (verbatim k3_gc_7 slice, tools/mb_gen_gc7.py)
_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096
NB_VML_REP = ()


def _nb_vml_rep(m):
    for lo, hi, rep in NB_VML_REP:
        if lo <= m <= hi:
            return rep
    return _k2a_bucket_of(m)


_vml_MMAX = 320
_vml_JMAX = 80
_vml_RMAX = _vml_MMAX * _vml_TOPK
_vml_SLOTS = 136
_vml_FIXW = 8 * 128 * 4
_vml_COST = (128, 135, 147, 160, 171, 182, 193, 204)


def _vml_mag(d):
    return -(-(1 << 40) // d)


def _vml_wg_spec64(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, %d, 1, 1, 0;\\n"
    s = ""
    for nm, sd, cons in (("wgmma_acc64", 1, plus), ("wgmma_zero64", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint64_t desc) {\n  asm volatile(\"" % (nm, n)
              + body % (n, outs, a, na + 4, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc));\n}\n" % cons)
    return s


def _vml_wg_spec(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = ("{\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nwgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 "
            "{%s}, {%s}, dd, %d, 1, 1, 0;\\n}\\n")
    s = ""
    for nm, sd, cons in (("wgmma_acc", 1, plus), ("wgmma_zero", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint32_t lo, uint32_t hi) {\n  asm volatile(\"" % (nm, n)
              + body % (na + 4, na + 5, n, outs, a, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}\n" % cons)
    return s


_vml_CUDA_T = r"""#ifndef SFX
#define SFX 0
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace vm {
constexpr int KDIM = 4096, E_LOC = 32, TOPK = 8;
constexpr int SW = 18432;
constexpr int US = 2 * SW;
#ifndef VM_JN
#define VM_JN 64
#endif
#ifndef VM_NTM
#define VM_NTM (VM_JN / 8)
#endif
#ifndef VM_NX
#define VM_NX 3
#endif
#ifndef VM_XSN
#define VM_XSN 64
#endif
#ifndef VM_MKW
#define VM_MKW 32
#endif
static_assert(VM_NTM * 8 == VM_JN && VM_JN <= VM_XSN && VM_NTM <= 16 && (VM_MKW == 32 || VM_MKW == 64), "m32e config");
constexpr int NS = 3, NX = VM_NX;
constexpr int XS = VM_XSN * 512;
constexpr int MKW = VM_MKW, MKB = (VM_MKW == 64) ? 24 : 16, MKX = (VM_MKW == 64) ? 40 : 24, MKP = (VM_MKW == 64) ? 45 : 29;
constexpr int NCW = 16;
#ifndef VM_WG5
#define VM_WG5 0
#endif
#ifndef VM_PREG
#define VM_PREG 56
#endif
#ifndef VM_CREG
#define VM_CREG 112
#endif
#define VM_S2(x) #x
#define VM_S(x) VM_S2(x)
constexpr int NTH = 32 * NCW + (VM_WG5 ? 128 : 64);
#if VM_WG5 >= 2
#define VM_RSYNC() bar_sync(8, 32 * NCW)
#define VM_RNT (32 * NCW)
#else
#define VM_RSYNC() __syncthreads()
#define VM_RNT NTH
#endif
#ifndef VM_W5B6
#define VM_W5B6 1   // 0 = the R-m32e-14..17 barrier-6 count (race on ent; stress-test arm only)
#endif
#define VM_B6 ((VM_WG5 >= 2 && VM_W5B6) ? 32 * NCW + 96 : 32 * NCW + 32)
constexpr int JMAX = @JMAX@, GMAX = 136, SLOTS = @SLOTS@, FIXW = (@FIXW@ > VM_NTM * 512) ? @FIXW@ : VM_NTM * 512;
#ifndef VM_DP
#define VM_DP 0
#endif
#ifndef VM_DPI
#define VM_DPI 4
#endif
constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);
#ifndef VM_BRED
#define VM_BRED 0
#endif
#ifndef VM_DEFER2
#define VM_DEFER2 0
#endif
#ifndef VM_PS
#define VM_PS 0
#endif
#ifndef VM_AD
#define VM_AD 0
#endif
#ifndef VM_GSM
#define VM_GSM 0
#endif
#ifndef VM_UD
#define VM_UD 0
#endif
#ifndef VM_SYNC7
#define VM_SYNC7 0
#endif
#ifndef VM_NOB9
#define VM_NOB9 0
#endif
#ifndef VM_LD
#define VM_LD 0
#endif
#ifndef VM_FT
#define VM_FT 0
#endif
#ifndef VM_FTR
#define VM_FTR 0
#endif
#ifndef VM_TA
#define VM_TA 0
#endif
#if VM_NOB9
#define VM_F9(a) (0)
#else
#define VM_F9(a) ((a).flags & 512)
#endif
#ifndef VM_FSW
#define VM_FSW 0
#endif
#if VM_MKW == 64
#define SEGMK(...) __VA_ARGS__
#else
#define SEGMK(...)
#endif
static_assert(!(VM_BRED && VM_DP), "m32e: VM_BRED and VM_DP use different split paths");
#ifndef VM_SC
#define VM_SC 0
#endif
static_assert(!(VM_SC && VM_DP), "m32e: VM_SC and VM_DP are separate plans");
constexpr int BAR_OFF = NS * US + NX * XS;
constexpr int TAB_OFF = BAR_OFF + 8 * 2 * (NS + NX);
constexpr long long WAIT_BUDGET = 1ll << 31;

struct Tab {
int nj, R, Rpad, pad0;
int wlast[4];
int jE[JMAX], jC[JMAX], jS[JMAX + 1], jrow[JMAX + 1];
uint32_t cpA[JMAX + 1], cpB[JMAX + 1];
int bA[GMAX + 1], bB[GMAX + 1];
int est[E_LOC];
#if VM_GSM
float gs13[E_LOC], gs2[E_LOC];
#endif
int bready, pad1;                 // bit 9: B stage orders below are written (x-producer lanes 1-31, during phase A)
uint32_t brd[16];                 // bit 9 scratch: predicted readiness of one job's 16 B stages
uint8_t bpos[16];                 // bit 9 scratch: order position -> stage
#if VM_DP
int sA, sB, qpub, qpad;
uint32_t cA, cB;
int q[16];
#endif
#if VM_RH
#define VM_RHQ 32
int rhq[4][VM_RHQ];               // R-vrh: per-WG pending doneA indices (-2 = phase A done)
int rhn[4];                       // R-vrh: per-WG push count (st.release.cta / ld.acquire.cta)
#endif
unsigned long long bperm[JMAX];   // bit 9: per job, 16 x 4 bits: B order position -> stage
};
constexpr int SMEM = TAB_OFF + (int)sizeof(Tab);

struct Args {
const __nv_bfloat16* x; const int* ids; const float* wts; int M; int ls;
const uint8_t* ws13; const uint8_t* ws2; const float* gsc13; const float* gsc2; const float* nw;
__nv_bfloat16* xA; __nv_bfloat16* xB; float* y; float* fix; int* cnt; int* xready; int* doneA; int* rmap; int* gbar;
__nv_bfloat16* out; int* err; long long* marks; unsigned long long magG; int flags;
uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];
#if VM_DP
int poolA, poolB;
#endif
#if VM_SC
int segA, segB;
#endif
#if VM_PS
int pdel;
#endif
#if VM_MKW == 64
int G0;
#endif
};
__device__ __forceinline__ int split_point(const Args& A, int pb, int g0, int g1) {
const int ue = ((g1 - 1) >> 4) << 4;
return ((A.flags & 1) && pb == 0 && (g1 & 15) != 0 && ue > g0) ? ue : g0;
}

__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}
__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // UNP L2 layout (tools/unp_patches.py)
o[0] = w & 0x81C081C0u;
o[1] = __funnelshift_l(w, w, 10) & 0x81C081C0u;
o[2] = __funnelshift_l(w, w, 13) & 0x81C081C0u;
o[3] = ((w & 0x001B001Bu) * unp_ok32(0x1084u)) & 0x81C081C0u;   // UNP L2: carry-free bit gather
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
uint32_t d;
asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
return d;
}
#if VM_FSW
__device__ __forceinline__ float swiglu(float g, float u) { return __fdividef(g, 1.f + __expf(-g)) * u; }
#else
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }
#endif
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
uint32_t ok;
asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
: "=r"(ok) : "r"(b), "r"(par) : "memory");
return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err) {
if (mbar_try(b, par)) return;
const long long t0 = clock64();
int n = 0;
while (!mbar_try(b, par)) {
if ((++n & 255) == 0) {
if (*reinterpret_cast<volatile int*>(err) != 0) return;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); return; }
}
}
}
__device__ __forceinline__ int ld_acquire(const int* p) {
int v;
asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
return v;
}
__device__ __forceinline__ void wait_flag(const int* p, int target, int* err) {
if (ld_acquire(p) >= target) return;
const long long t0 = clock64();
while (ld_acquire(p) < target) {
__nanosleep(64);
if (*reinterpret_cast<volatile int*>(err) != 0) return;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 2); return; }
}
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" :: "l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int atom_acqrel_add(int* p, int v) {
int o;
asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(o) : "l"(p), "r"(v) : "memory");
return o;
}
__device__ __forceinline__ void red_add_v4(float* p, float a, float b, float c, float d) {
asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.global;\n" ::: "memory"); }
__device__ __forceinline__ void fence_acq_rel() { asm volatile("fence.acq_rel.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
:: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_ef(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
:: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
uint64_t pol;
asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
return pol;
}
__device__ __forceinline__ long long gtime() {
long long t;
asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
return t;
}
__device__ __noinline__ long long lat_probe(const uint8_t* p, int flags) {   // marks only: cycles per dependent L2 load
const unsigned z = (unsigned)flags >> 30;   // 0, opaque to the compiler
unsigned v;
asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
const long long t0 = clock64();
#pragma unroll 1
for (int k = 0; k < 8; ++k) asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(v) : "l"(p + (v & z)) : "memory");
asm volatile("add.u32 %0, %0, 0;\n" : "+r"(v));
return (clock64() - t0 + (long long)(v & z)) / 7;
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
template <int N> __device__ __forceinline__ void wgmma_acc(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <int N> __device__ __forceinline__ void wgmma_acc64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
template <int N> __device__ __forceinline__ void wgmma_zero64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);
@WGSPECS@
__device__ __forceinline__ uint32_t mdiv(uint32_t a, unsigned long long m) {
const unsigned long long lo = (unsigned long long)a * (m & 0xFFFFFFFFull);
const unsigned long long hi = (unsigned long long)a * (m >> 32);
return (uint32_t)((hi + (lo >> 32)) >> 8);
}
__device__ __forceinline__ int excl_scan(int v, int lane) {
int s = v;
#pragma unroll
for (int o = 1; o < 32; o <<= 1) { const int n = __shfl_up_sync(0xffffffffu, s, o); if (lane >= o) s += n; }
return s - v;
}

__device__ __forceinline__ int stage_at(const Args& A, const Tab& T, int pb, int c, int G, unsigned long long magG) {
const int nj = T.nj;
if (nj == 0) return 0;
#if VM_DP
if (c >= G) return pb ? T.sB : T.sA;
const uint32_t* cp = pb ? T.cpB : T.cpA;
const uint32_t x = mdiv((pb ? T.cB : T.cA) * (uint32_t)c, magG);
#else
if (c >= G) return nj << (9 - pb);
const uint32_t* cp = pb ? T.cpB : T.cpA;
const uint32_t x = mdiv(cp[nj] * (uint32_t)c, magG);
#endif
int lo = 0, hi = nj;
while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (cp[mid] <= x) lo = mid; else hi = mid; }
const int nt = (T.jC[lo] + 7) >> 3;
#if VM_SC
const uint32_t xr = x - cp[lo], cs = A.cost[VM_NTM * pb + nt - 1], sg = (uint32_t)(pb ? A.segB : A.segA);
const uint32_t uc = 16u * cs + sg, uu = xr / uc, rr = xr - uu * uc;
uint32_t st = 16u * uu + (rr > sg ? (rr - sg) / cs : 0u);
#else
uint32_t st = mdiv(x - cp[lo], A.mag[VM_NTM * pb + nt - 1]);
#endif
const uint32_t ns = 512u >> pb;
if (st > ns - 1) st = ns - 1;
#if VM_SC
if (A.flags & 16384) {
const uint32_t ks = (pb ? (uint32_t)A.segB : (uint32_t)A.segA) / A.cost[VM_NTM * pb + nt - 1], w = st & 15u;
if (w <= ks) st -= w;
else if (16u - w <= ks) st += 16u - w;
}
#endif
return (lo << (9 - pb)) + (int)st;
}
__device__ __forceinline__ int cta_of(const Tab& T, int pb, int s, int G) {
const int* b = pb ? T.bB : T.bA;
int lo = 0, hi = G;
while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (b[mid] <= s) lo = mid; else hi = mid; }
return lo;
}
#ifndef VM_RT
#define VM_RT 0
#endif
#if VM_RT
static_assert(!VM_DP && !VM_SC && NCW >= 11 && GMAX + 1 <= 160 && 2 * VM_NTM <= 32, "VM_RT: static plan, warps 0-4 / 5-9 cover bA / bB");
__device__ __forceinline__ unsigned rt_eq(unsigned v, const unsigned* bb, int x) {   // lanes whose 5-bit id equals x
unsigned m = v;
#pragma unroll
for (int b = 0; b < 5; ++b) m &= ((x >> b) & 1) ? bb[b] : ~bb[b];
return m;
}
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {   // R-rt-1: two barriers
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const int E8 = A.M * TOPK;
const int C = (((E8 + 15) >> 4) + 31) & ~31;
const int e0 = warp * C, e1 = min(E8, e0 + C);
int xv[5];
#pragma unroll
for (int k = 0; k < 5; ++k) xv[k] = __ldg(A.ids + min(e0 + 32 * k + lane, E8 - 1));   // unpredicated: all 5 loads in flight at once
#pragma unroll
for (int k = 0; k < 5; ++k) {
const int e = e0 + 32 * k + lane, id = xv[k] - A.ls;
xv[k] = (warp < NCW && 32 * k < C && e < e1 && (unsigned)id < (unsigned)E_LOC) ? id : -1;
}
uint32_t* scost = reinterpret_cast<uint32_t*>(hist + 768);   // cost / mag tables in smem: no indexed param (LDC) chains
unsigned long long* smag = reinterpret_cast<unsigned long long*>(hist + 832);
const int tq = min(tid, 2 * VM_NTM - 1);   // issued with the ids loads, stored after the ballots (LDC latency overlapped)
const uint32_t cq = A.cost[tq];
const unsigned long long mq = A.mag[tq];
int kb[5], kr[5];   // per chunk: entries of expert `lane` in this warp's earlier chunks; rank of this lane's entry in the chunk
if (warp < NCW) {
int run = 0;
#pragma unroll
for (int k = 0; k < 5; ++k) {
kb[k] = run;
kr[k] = 0;
if (e0 + 32 * k < e1) {
const int x = xv[k];
unsigned bb[5];
const unsigned v = __ballot_sync(0xffffffffu, x >= 0);
#pragma unroll
for (int b = 0; b < 5; ++b) bb[b] = __ballot_sync(0xffffffffu, (x >> b) & 1);
kr[k] = __popc(rt_eq(v, bb, x) & ((1u << lane) - 1u));
run += __popc(rt_eq(v, bb, lane));
}
}
hist[warp * 33 + lane] = run;
}
if (tid < 2 * VM_NTM) { scost[tid] = cq; smag[tid] = mq; }
VM_RSYNC();
if (warp < NCW) {
int tot = 0, pre = 0;
#pragma unroll
for (int w = 0; w < NCW; ++w) { const int h = hist[w * 33 + lane]; pre += (w < warp) ? h : 0; tot += h; }
const int est = excl_scan(tot, lane);
#pragma unroll
for (int k = 0; k < 5; ++k) {   // route rows: r = est[x] + earlier warps' x entries + this warp's earlier chunks + chunk rank
if (e0 + 32 * k >= e1) break;
const int e = e0 + 32 * k + lane, x = xv[k];
const int rb = __shfl_sync(0xffffffffu, est + pre + kb[k], x & 31);
const int r = (x >= 0) ? rb + kr[k] : -1;
if (x >= 0) ent[r] = (uint16_t)e;
if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;
}
int* pl = hist + 528;   // plan handoff (last warp -> warps 0-9): 6 x 32 per-expert values + 3 scalars
if (warp == NCW - 1) {   // the plan, once: job split (the same closed form), padded rows, per-expert costs and their scans
#if VM_JN == 64
const int nj = (tot + 63) >> 6;
int base = 64;
#if VM_TA
if (nj) base = ((((tot + 7) >> 3) + nj - 1) / nj) << 3;   // VTR: tile-aligned split
#else
if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));
#endif
#else
const int nj = (tot + VM_JN - 1) / VM_JN;
#if VM_TA
const int base = nj ? ((((tot + 7) >> 3) + nj - 1) / nj) << 3 : VM_JN;   // VTR: tile-aligned split
#else
const int base = nj ? (tot + nj - 1) / nj : VM_JN;
#endif
#endif
int prow = 0, ca = 0, cb = 0;
for (int k = 0; k < nj; ++k) {
const int sz = min(base, tot - k * base), nt = ((sz + 7) >> 3) - 1;
prow += (sz + 7) & ~7;
ca += (int)scost[nt];
cb += (int)scost[VM_NTM + nt];
}
const int pjr = excl_scan(nj | (prow << 16), lane);   // jobs <= 80, padded rows <= 3200: two 16-bit fields, no carry
const int jst = pjr & 0xFFFF, rst = pjr >> 16, sca = excl_scan(ca, lane), scb = excl_scan(cb, lane);
const uint32_t csA = 512u * (uint32_t)sca, csB = 256u * (uint32_t)scb;
const int njT = __shfl_sync(0xffffffffu, jst + nj, 31);
const uint32_t totA = __shfl_sync(0xffffffffu, csA + 512u * (uint32_t)ca, 31), totB = __shfl_sync(0xffffffffu, csB + 256u * (uint32_t)cb, 31);
pl[lane] = (int)csA; pl[32 + lane] = (int)csB; pl[64 + lane] = nj; pl[96 + lane] = base; pl[128 + lane] = tot; pl[160 + lane] = jst;
if (lane == 0) { pl[192] = njT; pl[193] = (int)totA; pl[194] = (int)totB; }
bar_arrive(10, 32 * 11);
int r = rst;
uint32_t pa = csA, pq = csB;
for (int k = 0; k < nj; ++k) {
const int j = jst + k, sz = min(base, tot - k * base), nt = ((sz + 7) >> 3) - 1;
T.jE[j] = lane; T.jC[j] = sz; T.jS[j] = est + k * base; T.jrow[j] = r; T.cpA[j] = pa; T.cpB[j] = pq;
r += (sz + 7) & ~7;
pa += 512u * scost[nt];
pq += 256u * scost[VM_NTM + nt];
}
T.est[lane] = est;
#if VM_GSM
T.gs13[lane] = __ldg(A.gsc13 + lane); T.gs2[lane] = __ldg(A.gsc2 + lane);
#endif
if (lane == 31) {
T.nj = njT; T.R = est + tot; T.Rpad = rst + prow;
T.jS[njT] = est + tot; T.jrow[njT] = rst + prow; T.cpA[njT] = totA; T.cpB[njT] = totB;
}
} else if (warp < 10) {   // stage_at(pb, c) for c = 0..G: warps 0-4 phase A, 5-9 phase B
bar_sync(10, 32 * 11);
const int G = gridDim.x, pb = (warp >= 5) ? 1 : 0, c = tid - 160 * pb;
const uint32_t cs = (uint32_t)pl[32 * pb + lane];
const int nj = pl[64 + lane], base = pl[96 + lane], tq2 = pl[128 + lane], jst = pl[160 + lane];
const int njT = pl[192];
const uint32_t totP = (uint32_t)pl[193 + pb];
const int lastne = 31 - __clz(__ballot_sync(0xffffffffu, nj > 0));   // the last expert with jobs (-1: none)
const uint32_t x = mdiv(totP * (uint32_t)c, A.magG);
int lo = 0;
#pragma unroll
for (int s = 16; s >= 1; s >>= 1) { const uint32_t v = __shfl_sync(0xffffffffu, cs, (lo + s) & 31); if (lo + s <= lastne && v <= x) lo += s; }
const int njE = __shfl_sync(0xffffffffu, nj, lo), bE = __shfl_sync(0xffffffffu, base, lo), tE = __shfl_sync(0xffffffffu, tq2, lo);
const int jE0 = __shfl_sync(0xffffffffu, jst, lo);
uint32_t cp = __shfl_sync(0xffffffffu, cs, lo);
const uint32_t sc = 512u >> pb;
int k = 0, sz = min(bE, tE);
while (k + 1 < njE) {
const uint32_t w = sc * scost[VM_NTM * pb + max(1, (sz + 7) >> 3) - 1];
if (cp + w > x) break;
cp += w;
++k;
sz = min(bE, tE - k * bE);
}
const int nt = max(1, (sz + 7) >> 3);
uint32_t st = mdiv(x - cp, smag[VM_NTM * pb + nt - 1]);
const uint32_t ns = 512u >> pb;
if (st > ns - 1) st = ns - 1;
const int res = (njT == 0) ? 0 : (c >= G) ? (njT << (9 - pb)) : (((jE0 + k) << (9 - pb)) + (int)st);
if (c <= G) { if (pb) T.bB[c] = res; else T.bA[c] = res; }
}
}
if (tid == 0) T.bready = 0;
VM_RSYNC();
}
#else
__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const int E8 = A.M * TOPK;
const int C = (((E8 + 15) >> 4) + 31) & ~31;
const int e0 = warp * C, e1 = min(E8, e0 + C);
int xv[5];
#pragma unroll
for (int k = 0; k < 5; ++k) {
const int e = e0 + 32 * k + lane;
xv[k] = -1;
if (warp < NCW && 32 * k < C && e < e1) { const int id = __ldg(A.ids + e) - A.ls; xv[k] = ((unsigned)id < (unsigned)E_LOC) ? id : -1; }
}
for (int i = tid; i < 16 * 33; i += VM_RNT) hist[i] = 0;
VM_RSYNC();
if (warp < NCW) {
#pragma unroll
for (int k = 0; k < 5; ++k) {
if (e0 + 32 * k >= e1) break;
const int x = xv[k];
const unsigned m = __match_any_sync(0xffffffffu, x);
if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
__syncwarp();
}
}
VM_RSYNC();
if (warp == 0) {
int tot = 0;
for (int w = 0; w < NCW; ++w) { const int h = hist[w * 33 + lane]; hist[w * 33 + lane] = tot; tot += h; }
const int est = excl_scan(tot, lane);
#if VM_JN == 64 && !VM_DP
const int nj = (tot + 63) >> 6;
int base = 64;
if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));   // == the countdown loop for every (base, nj, tot) (tools/gen_k3_mb.py asserts it); mb bit 13
#else
const int nj = (tot + VM_JN - 1) / VM_JN;
#if VM_TA
const int base = nj ? ((((tot + 7) >> 3) + nj - 1) / nj) << 3 : VM_JN;   // VTR: tile-aligned split
#else
const int base = nj ? (tot + nj - 1) / nj : VM_JN;
#endif
#endif
int prow = 0;
for (int k = 0; k < nj; ++k) { const int sz = min(base, tot - k * base); prow += (sz + 7) & ~7; }
#if VM_DP
int rank = lane;
if (A.flags & 8192) { rank = 0; for (int k = 0; k < 32; ++k) { const int t2 = __shfl_sync(0xffffffffu, tot, k); rank += (t2 > tot) || (t2 == tot && k < lane); } }
fin[rank] = (uint32_t)nj; fin[32 + rank] = (uint32_t)prow;
__syncwarp();
const int jsr = excl_scan((int)fin[lane], lane), rsr = excl_scan((int)fin[32 + lane], lane);
fin[64 + lane] = (uint32_t)jsr; fin[96 + lane] = (uint32_t)rsr;
__syncwarp();
const int jst = (int)fin[64 + rank], rst = (int)fin[96 + rank];
#else
const int jst = excl_scan(nj, lane), rst = excl_scan(prow, lane);
#endif
int r = rst;
for (int k = 0; k < nj; ++k) {
const int j = jst + k, sz = min(base, tot - k * base);
T.jE[j] = lane; T.jC[j] = sz; T.jS[j] = est + k * base; T.jrow[j] = r;
r += (sz + 7) & ~7;
}
T.est[lane] = est;
#if VM_GSM
T.gs13[lane] = __ldg(A.gsc13 + lane); T.gs2[lane] = __ldg(A.gsc2 + lane);
#endif
#if VM_DP
{
const int njT = __reduce_add_sync(0xffffffffu, nj), RT = __reduce_add_sync(0xffffffffu, tot), RpT = __reduce_add_sync(0xffffffffu, prow);
if (lane == 0) { T.nj = njT; T.R = RT; T.Rpad = RpT; T.jS[njT] = RT; T.jrow[njT] = RpT; }
}
#else
if (lane == 31) {
T.nj = jst + nj; T.R = est + tot; T.Rpad = rst + prow;
T.jS[jst + nj] = est + tot; T.jrow[jst + nj] = rst + prow;
}
#endif
}
VM_RSYNC();
if (warp < NCW) {
#pragma unroll
for (int k = 0; k < 5; ++k) {
if (e0 + 32 * k >= e1) break;
const int e = e0 + 32 * k + lane, x = xv[k];
const unsigned m = __match_any_sync(0xffffffffu, x);
const int rank = __popc(m & ((1u << lane) - 1u));
int r = -1;
if (x >= 0) { r = T.est[x] + hist[warp * 33 + x] + rank; ent[r] = (uint16_t)e; }
if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;
__syncwarp();
if (x >= 0 && lane == __ffs(m) - 1) hist[warp * 33 + x] += __popc(m);
__syncwarp();
}
}
VM_RSYNC();
if (warp == 0) {
uint32_t accA = 0, accB = 0;
for (int j0 = 0; j0 < T.nj; j0 += 32) {
const int j = j0 + lane;
const int nt = (j < T.nj) ? ((T.jC[j] + 7) >> 3) - 1 : 0;
const int ca = (j < T.nj) ? (int)A.cost[nt] : 0, cb = (j < T.nj) ? (int)A.cost[VM_NTM + nt] : 0;
#if VM_SC
const int ja = (j < T.nj) ? 512 * ca + 32 * A.segA : 0, jb = (j < T.nj) ? 256 * cb + 16 * A.segB : 0;
const int sc = excl_scan(ja, lane), sd = excl_scan(jb, lane);
if (j < T.nj) { T.cpA[j] = accA + (uint32_t)sc; T.cpB[j] = accB + (uint32_t)sd; }
accA += (uint32_t)__shfl_sync(0xffffffffu, sc + ja, 31);
accB += (uint32_t)__shfl_sync(0xffffffffu, sd + jb, 31);
#else
const int sc = excl_scan(ca, lane), sd = excl_scan(cb, lane);
if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }
accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);
accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);
#endif
}
if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }
#if VM_DP
if (lane == 0) {
const int SA = T.nj << 9, SB = T.nj << 8;
const int pa = min(DPMAX, (int)(((long long)(SA >> 4) * A.poolA) >> 8)), pq = min(DPMAX, (int)(((long long)(SB >> 4) * A.poolB) >> 8));
T.sA = SA - (pa << 4); T.sB = SB - (pq << 4);
const int ja = T.sA >> 9, jb = T.sB >> 8;
T.cA = (ja < T.nj) ? T.cpA[ja] + (uint32_t)(T.sA - (ja << 9)) * A.cost[((T.jC[ja] + 7) >> 3) - 1] : T.cpA[T.nj];
T.cB = (jb < T.nj) ? T.cpB[jb] + (uint32_t)(T.sB - (jb << 8)) * A.cost[VM_NTM + ((T.jC[jb] + 7) >> 3) - 1] : T.cpB[T.nj];
T.qpub = 0;
}
#endif
}
VM_RSYNC();
const int G = gridDim.x;
if (tid <= G) { T.bA[tid] = stage_at(A, T, 0, tid, G, A.magG); T.bB[tid] = stage_at(A, T, 1, tid, G, A.magG); }
if (tid == 0) T.bready = 0;
VM_RSYNC();
}

#endif   // VM_RT
__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {
const int tid = threadIdx.x, G = gridDim.x, c = blockIdx.x;
const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int s = tid >> 5, kk = (tid & 31) * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
constexpr int GB = 8;
for (int qq = q0; qq < q1; qq += GB) {
uint4 v[GB];
int off[GB];
#pragma unroll
for (int k = 0; k < GB; ++k) {
const int q = qq + k;
off[k] = -1;
if (q < q1) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r < cn) {
const int npad = (cn + 7) & ~7;
off[k] = T.jrow[j] * 2048 + s * npad * 128 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
v[k] = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + tid);
}
}
}
#pragma unroll
for (int k = 0; k < GB; ++k)
if (off[k] >= 0) {
uint32_t* d = xa + off[k];
d[0] = v[k].x; d[32] = v[k].y; d[64] = v[k].z; d[96] = v[k].w;
}
}
fence_proxy_async();
bar_sync(5, 32 * NCW);
if (tid == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}

__device__ void gather_warp(const Args& A, const Tab& T, const uint16_t* ent, int lane) {
const int G = gridDim.x, c = blockIdx.x;
const int q0 = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1 = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
for (int q = q0; q < q1; ++q) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r >= cn) continue;
const int npad = (cn + 7) & ~7;
const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
uint4 v[16];
#pragma unroll
for (int m = 0; m < 16; ++m) v[m] = __ldg(src + 32 * m);
uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
for (int m = 0; m < 16; ++m) {
uint32_t* dm = d + m * npad * 128;
dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
}
}
fence_proxy_async();
__syncwarp();
if (lane == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}
#if VM_WG5
#ifndef VM_GV
#define VM_GV 8
#endif
__device__ void gather_w5(const Args& A, const Tab& T, const uint16_t* ent, int lane, int part) {
const int G = gridDim.x, c = blockIdx.x;
const int q0c = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1c = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int n = q1c - q0c, q0 = q0c + (n * part) / 3, q1 = q0c + (n * (part + 1)) / 3;
const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
for (int q = q0; q < q1; ++q) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r >= cn) continue;
const int npad = (cn + 7) & ~7;
const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
for (int h = 0; h < 16 / VM_GV; ++h) {
uint4 v[VM_GV];
#pragma unroll
for (int m = 0; m < VM_GV; ++m) v[m] = __ldg(src + 32 * (VM_GV * h + m));
#pragma unroll
for (int m = 0; m < VM_GV; ++m) {
uint32_t* dm = d + (VM_GV * h + m) * npad * 128;
dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
}
}
}
fence_proxy_async();
__syncwarp();
if (lane == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}
#endif
__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {
return (A.flags & 64) ? min(NX - 1, T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) : 0;
}
__device__ void self_fill(const Args& A, const Tab& T, uint8_t* smem, uint32_t bars, const uint16_t* ent, int nf) {
const int tid = threadIdx.x, c = blockIdx.x;
const int g0 = T.bA[c], g1 = T.bA[c + 1], sp = split_point(A, 0, g0, g1);
for (int f = 0; f < nf; ++f) {
const int gs = (sp + f < g1) ? sp + f : sp + f - (g1 - g0);
const int j = gs >> 9, st = gs & 15, cn = T.jC[j];
uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);
for (int w = tid; w < cn * 32; w += 32 * NCW) {
const int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;
const uint4 v = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch);
uint32_t* d = xs + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
d[0] = v.x; d[32] = v.y; d[64] = v.z; d[96] = v.w;
}
}
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
bar_sync(5, 32 * NCW);
if (tid == 0)
for (int f = 0; f < nf; ++f) mbar_arrive(bars + 8 * (2 * NS + f));
}

__device__ __forceinline__ uint32_t afin(const Args& A, const Tab& T, int U, int G) {
const int a = U << 4, b = a + 16;
int lo = 0, hi = G - 1;
while (lo < hi) { const int m = (lo + hi) >> 1; if (T.bA[m + 1] > a) hi = m; else lo = m + 1; }
uint32_t f = 0;
for (int c = lo; c < G && T.bA[c] < b; ++c) {
const int g0 = T.bA[c], g1 = T.bA[c + 1];
if (g1 <= g0) continue;
const int sp = split_point(A, 0, g0, g1), l2 = max(a, g0), h2 = min(b, g1);
if (h2 <= l2) continue;
const int pos = (l2 < sp) ? (g1 - sp) + (min(h2, sp) - 1 - g0) : h2 - 1 - sp;
const uint32_t fc = (uint32_t)(((unsigned long long)(pos + 1) << 16) / (unsigned)(g1 - g0));
f = fc > f ? fc : f;
}
return f;
}
__device__ void bperm_lanes(const Args& A, Tab& T, int lane) {
const unsigned mask = 0xfffffffeu;
const int c = blockIdx.x, G = gridDim.x, g0 = T.bB[c], g1 = T.bB[c + 1];
if (g1 > g0) {
for (int j = g0 >> 8; j <= (g1 - 1) >> 8; ++j) {
for (int s = lane - 1; s < 16; s += 31) {
const uint32_t f0 = afin(A, T, 32 * j + 2 * s, G), f1 = afin(A, T, 32 * j + 2 * s + 1, G);
T.brd[s] = f0 > f1 ? f0 : f1;
}
__syncwarp(mask);
for (int s = lane - 1; s < 16; s += 31) {
const uint32_t r = T.brd[s];
int rank = 0;
for (int t = 0; t < 16; ++t) { const uint32_t q = T.brd[t]; rank += (q < r) || (q == r && t < s); }
T.bpos[rank] = (uint8_t)s;
}
__syncwarp(mask);
if (lane == 1) {
unsigned long long pm = 0;
for (int r = 0; r < 16; ++r) pm |= (unsigned long long)T.bpos[r] << (4 * r);
T.bperm[j] = pm;
}
__syncwarp(mask);
}
}
if (lane == 1) { __threadfence_block(); *reinterpret_cast<volatile int*>(&T.bready) = 1; }
}
__device__ __forceinline__ void bperm_wait(const Args& A, const Tab& T) {
if (*reinterpret_cast<const volatile int*>(&T.bready) != 0) { __threadfence_block(); return; }
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.bready) == 0) {
__nanosleep(64);
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 3); break; }
}
__threadfence_block();
}
__device__ __forceinline__ int bstage(const Args& A, const Tab& T, int j, int s) {
return VM_F9(A) ? (int)((T.bperm[j] >> (4 * s)) & 15ull) : s;
}

__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {
const uint64_t pol = policy_evict_first();
#if VM_PS
const long long tps = gtime();
#endif
const int c = blockIdx.x;
int it = 0;
for (int pb = 0; pb < 2; ++pb) {
const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
if (pb && VM_F9(A)) bperm_wait(A, T);
for (int i = 0; i < g1 - g0; ++i, ++it) {
const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
const int e = T.jE[j];
const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
#if VM_PS
else if (it > 0 && A.pdel > 0) { const long long tw = tps + (long long)A.pdel * it; while (gtime() < tw) __nanosleep(32); }
#endif
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
#if VM_DP
{
Tab& TT = const_cast<Tab&>(T);
const int S0 = pb ? T.sB : T.sA, nit = ((pb ? (T.nj << 8) : (T.nj << 9)) - S0) / VM_DPI;
for (;;) {
const int i0 = (nit > 0) ? atomicAdd(A.gbar + pb, 1) : 0;
const int v = (i0 < nit) ? i0 : -1;
const int k = TT.qpub;
TT.q[k & 15] = v;
__threadfence_block();
*reinterpret_cast<volatile int*>(&TT.qpub) = k + 1;
if (v < 0) break;
for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int s = gs & 15;
const int e = T.jE[j];
const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
}
}
#endif
}
}
__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {
const int c = blockIdx.x, nf = nfill_of(A, T);
int it = 0;
long long wA = 0, wB = 0, tB0 = 0, tB1 = 0;   // marks only (v9): readiness waits of this CTA's x producer
int nB = 0;
#if VM_DP
int kq = 0;
#endif
for (int pb = 0; pb < 2; ++pb) {
const int g0 = pb ? T.bB[c] : T.bA[c], g1 = pb ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, pb, g0, g1);
if (pb && VM_F9(A)) bperm_wait(A, T);
int lastj = -1;
for (int i = 0; i < g1 - g0; ++i, ++it) {
const int gs = (sp + i < g1) ? sp + i : sp + i - (g1 - g0);
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5);
const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);
const int npad = (T.jC[j] + 7) & ~7;
if (!pb && i < nf) continue;
if (!pb) {
if (j != lastj) {
if (A.marks && ld_acquire(A.xready + j) < npad) {
const long long t = gtime();
wait_flag(A.xready + j, npad, A.err);
wA += gtime() - t;
} else {
wait_flag(A.xready + j, npad, A.err);
}
fence_proxy_async(); lastj = j;
}
} else {
if (A.marks && ld_acquire(A.doneA + j * 16 + s) < 8) {
const long long t = gtime();
wait_flag(A.doneA + j * 16 + s, 8, A.err);
wB += gtime() - t; ++nB;
} else {
wait_flag(A.doneA + j * 16 + s, 8, A.err);
}
fence_proxy_async();
if (A.marks) { tB1 = gtime(); if (tB0 == 0) tB0 = tB1; }
}
const uint32_t bytes = (uint32_t)npad * 512u;
const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
#if VM_DP
{
const int S0 = pb ? T.sB : T.sA;
for (;;) {
if (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 4); break; }
}
}
__threadfence_block();
const int v = T.q[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5);
const int s = gs & 15;
const int npad = (T.jC[j] + 7) & ~7;
if (!pb) wait_flag(A.xready + j, npad, A.err);
else wait_flag(A.doneA + j * 16 + s, 8, A.err);
fence_proxy_async();
const uint32_t bytes = (uint32_t)npad * 512u;
const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
}
}
#endif
}
if (A.marks) {   // cols 24..28: B doneA wait ns, B stages that waited, A xready wait ns, first / last B x acquire (gtime)
long long* m = A.marks + (size_t)c * MKW;
m[MKX] = wB; m[MKX + 1] = nB; m[MKX + 2] = wA; m[MKX + 3] = tB0; m[MKX + 4] = tB1;
}
}

__device__ __forceinline__ void dec_half(uint32_t w0, uint32_t w8, uint32_t Sg, uint32_t S8, uint32_t* hh) {
deq4(w0, hh); deq4(w8, hh + 4);
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2(hh[j], Sg); hh[4 + j] = bmul2(hh[4 + j], S8); }
}
template <int N, bool Z>
__device__ __forceinline__ void mma2(float* d, const uint32_t* hh, uint32_t dlo, uint32_t dhi) {
if constexpr (Z) wgmma_zero<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
else wgmma_acc<N>(d, hh[0], hh[4], hh[1], hh[5], dlo, dhi);
wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);
}
#if VM_UD
template <int N, bool Z>
__device__ __forceinline__ void mma2u(float* d, const uint32_t* hh, uint64_t desc) {
if constexpr (Z) wgmma_zero64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);
else wgmma_acc64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);
wgmma_acc64<N>(d, hh[2], hh[6], hh[3], hh[7], desc + 16);
}
#endif
#ifndef VM_RH
#define VM_RH 0   // R-vrh: 1 = doneA releases issued by producer-WG warp NCW+2 (VM_WG5 >= 2)
#endif
#if VM_RH && VM_WG5 < 2
#error "VM_RH needs the WG5 producer warpgroup (spare warp NCW+2)"
#endif
#ifndef VM_DEFER
#define VM_DEFER 0
#endif
#ifndef VM_TRICKS
#define VM_TRICKS 1
#endif
#ifndef VM_TRICKS_DQ
#define VM_TRICKS_DQ 1   // 0: VM_TRICKS without the opaque-multiply dequant (2 registers fewer)
#endif
__device__ __forceinline__ void deq4_dq(uint32_t w, uint32_t* o, uint32_t k64, uint32_t k4) {   // UNP L2: == deq4
(void)k64; (void)k4;
deq4(w, o);
}
__device__ __forceinline__ uint32_t bmul2l(uint32_t a, uint32_t b) {   // a x (b.lo, b.lo): HMUL2 .H0_H0
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __low2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t bmul2h(uint32_t a, uint32_t b) {   // a x (b.hi, b.hi): HMUL2 .H1_H1
__nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), __high2bfloat162(*reinterpret_cast<__nv_bfloat162*>(&b)));
return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void dec_half2(uint32_t w0, uint32_t w8, uint32_t sw2, uint32_t par, uint32_t* hh, uint32_t k16,
uint32_t k64, uint32_t k4) {
#if VM_TRICKS_DQ
deq4_dq(w0, hh, k64, k4); deq4_dq(w8, hh + 4, k64, k4);
#else
deq4(w0, hh); deq4(w8, hh + 4);
#endif
const uint32_t sb = 2u * par;
const uint32_t sgp = __byte_perm(sw2, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16 + 0x75807580u;
#pragma unroll
for (int j = 0; j < 4; ++j) { hh[j] = bmul2l(hh[j], sgp); hh[4 + j] = bmul2h(hh[4 + j], sgp); }
}
#if VM_AD
struct Pend { int i; int su, sslot, snt, scnt, son, stag; };
#else
struct Pend { int i; };
#endif

#if VM_RH
__device__ __forceinline__ int rh_ld_acq(const int* p) {
int v;
asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];\n" : "=r"(v) : "r"(sa(p)) : "memory");
return v;
}
__device__ __forceinline__ void rh_st_rel(int* p, int v) {
asm volatile("st.release.cta.shared::cta.b32 [%0], %1;\n" :: "r"(sa(p)), "r"(v) : "memory");
}
__device__ __forceinline__ void rh_push(int q, int v) {   // thread 0 of WG q only (single writer of rhn[q])
extern __shared__ __align__(1024) uint8_t smem[];
Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
const int k = T.rhn[q];
T.rhq[q][k & (VM_RHQ - 1)] = v;
rh_st_rel(&T.rhn[q], k + 1);
}
__device__ void rh_loop(const Args& A, Tab& T) {   // producer-WG warp NCW+2, lane 0
int seen0 = 0, seen1 = 0, seen2 = 0, seen3 = 0, done = 0;
const long long t0 = clock64();
while (done < 4) {
bool any = false;
#pragma unroll
for (int q = 0; q < 4; ++q) {
int& sn = q == 0 ? seen0 : q == 1 ? seen1 : q == 2 ? seen2 : seen3;
const int n = rh_ld_acq(&T.rhn[q]);
while (sn < n) {
const int idx = *reinterpret_cast<volatile int*>(&T.rhq[q][sn & (VM_RHQ - 1)]);
++sn;
any = true;
if (idx < 0) ++done;
else red_release_add(A.doneA + idx, 1);
}
}
if (!any) {
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 3); return; }
__nanosleep(64);
}
}
}
#endif
__device__ __forceinline__ void pend_flush(const Args& A, Pend& pd, int q, int wp, int lane) {
if (pd.i >= 0) {
fence_proxy_async();
bar_sync(1 + q, 128);
#if VM_RH
if (((wp << 5) | lane) == 0) rh_push(q, pd.i);   // R-vrh: the release is issued by the helper warp
#else
if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);
#endif
pd.i = -1;
}
}
#ifndef VCP_NB
#define VCP_NB 2
#endif
#ifndef VCP_W
#define VCP_W 1
#endif
#ifndef VCP_G2
#define VCP_G2 0
#endif
#ifndef VCP_NMIN
#define VCP_NMIN 24
#endif
#ifndef VCP_NMAX
#define VCP_NMAX 128
#endif
#ifndef VCP_PK
#define VCP_PK 0
#endif
#ifndef VCP_PRE
#define VCP_PRE 0
#endif
#ifndef VCP_EA
#define VCP_EA 0
#endif
#ifndef VCP_EAN
#define VCP_EAN 48
#endif
#ifndef VCP_M1
#define VCP_M1 0
#endif
#ifndef VCP_LDA
#define VCP_LDA 0
#endif
#ifndef VCP_DPI
#define VCP_DPI 1
#endif
#ifndef VCP_DPH
#define VCP_DPH 1
#endif
template <int N> struct VcpCfg {   // R-vcp: stage pipeline shape per N (defaults = the shipped 2 buffers, wait 1, 8 groups)
static constexpr bool on = N >= VCP_NMIN && N <= VCP_NMAX;
static constexpr int NB = on ? VCP_NB : 2;
static constexpr int W = on ? VCP_W : 1;
static constexpr bool G2 = on && VCP_G2;
static constexpr bool PRE = on && VCP_PRE;
static_assert(!PRE || (NB == 4 && W == 1 && !G2), "VCP_PRE needs VCP_NB 4, VCP_W 1");
static constexpr int EA = VCP_EA ? (N <= VCP_EAN ? 1 : 2) : 0;   // 1: up-front codes, 2: c[1..3] after group 0
static_assert(NB == 2 || NB == 4, "VCP_NB");
static_assert(W >= 1 && W <= (G2 ? NB / 2 - 1 : NB - 1), "VCP_W too deep for VCP_NB");
};
template <int N>
__device__ __forceinline__ void stage(const Args& A, uint8_t* smem, uint32_t bars, int& it, int& prevx, float* d,
uint32_t (&hb)[VcpCfg<N>::NB][8], int q, int wp, int lane, bool first, Pend& pd) {
using VC = VcpCfg<N>;
#if VM_UD
static_assert(VC::NB == 2 && VC::W == 1 && !VC::G2, "VCP needs VM_UD 0");
#endif
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
#if VM_TRICKS
c[0] = cp[0];
const uint32_t sw0 = sp[0], sw1 = sp[32];
if constexpr (VC::EA == 1) {   // VCP (a): every code word up front, the weight slot freed before the x wait
#pragma unroll
for (int i = 1; i < 4; ++i) c[i] = cp[32 * i];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
}
#else
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
const uint32_t sw0 = sp[0], sw1 = sp[32];
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
#endif
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
#if VM_TRICKS
const uint32_t zo = (uint32_t)A.flags >> 30;   // 0 (flags < 2^30), opaque to the compiler
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint64_t d0 = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);
#if VM_UD
const uint64_t dsc = d0 + (uint64_t)((uint32_t)xs * (uint32_t)(XS >> 4));
#endif
#else
const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);
const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
#if VM_UD
const uint64_t dsc = d0;
#endif
#endif
#if VM_TRICKS
if constexpr (VC::PRE) {   // VCP (2'): group g + 1 unpacked between commit(g) and wait(g - 1): 2 groups in flight during it
#pragma unroll
for (int g = 0; g < 8; ++g) {
const int i = g >> 1, hf = g & 1;
if (g == 0) {
c[1] = cp[32];
dec_half2(c[0].x, c[0].z, sw0, 0u, hb[0], k16, k64, k4);
}
wg_fence();
if (first && g == 0) mma2<N, true>(d, hb[g % VC::NB], dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(d, hb[g % VC::NB], dlo + 64 * i + 32 * hf, dhi);
wg_commit();
if (g < 7) {
const int g1 = g + 1, i1 = g1 >> 1, h1 = g1 & 1;
if (h1 == 0 && i1 < 3) c[i1 + 1] = cp[32 * (i1 + 1)];   // codes one k64 step ahead of their unpack
dec_half2(h1 ? c[i1].y : c[i1].x, h1 ? c[i1].w : c[i1].z, (i1 < 2) ? sw0 : sw1, (uint32_t)(i1 & 1), hb[g1 % VC::NB], k16, k64, k4);
if (g1 == 6) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
}
wg_wait<1>();
if (g == 0 && prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && hf == VCP_DPH) pend_flush(A, pd, q, wp, lane);
}
}
} else
#endif
{
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
#if VM_TRICKS
if (VC::EA == 0 && i < 3) c[i + 1] = cp[32 * (i + 1)];   // the next 4-step group's codes, one group ahead
#else
const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#endif
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t* hh = hb[(2 * i + hf) % VC::NB];
#if VM_TRICKS
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16, k64, k4);
if (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // all codes in registers
if (VC::EA == 2 && i == 0 && hf == 1) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }   // VCP (a), N > VCP_EAN
#else
dec_half(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, Sg, S8, hh);
#endif
if (!VC::G2 || hf == 1) {
wg_fence();
if constexpr (VC::G2) {   // VCP (3): one commit group per k64 step (both k32 halves)
uint32_t* h0 = hb[(2 * i) % VC::NB];
if (first && i == 0) mma2<N, true>(d, h0, dlo + 64 * i, dhi);
else mma2<N, false>(d, h0, dlo + 64 * i, dhi);
mma2<N, false>(d, hh, dlo + 64 * i + 32, dhi);
} else {
if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);
else mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);
}
wg_commit();
if (VC::EA == 2 && i == 0 && hf == 0) {
#pragma unroll
for (int k = 1; k < 4; ++k) c[k] = cp[32 * k];
}
wg_wait<VC::W>();
if ((VC::G2 ? i : 2 * i + hf) == VC::W - 1 && prevx >= 0) {   // the previous stage's last group is complete
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && (VC::G2 || hf == VCP_DPH)) pend_flush(A, pd, q, wp, lane);   // shipped point: behind commit group 3 (G2: the only commit of step i)
}
}
}
}
}
prevx = xs;
++it;
}
__device__ __forceinline__ void mma16816(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
: "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
: "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it, float (&d2)[2][N / 2], int q,
int wp, int lane) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const int tile = (q & 1) * 4 + wp;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
uint4 c[4];
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
const uint32_t sw0 = sp[0], sw1 = sp[32];
#if SFX   // R-sfx: weight slot released after its LDS data returned
const uint32_t zom = (uint32_t)A.flags >> 30;   // opaque 0
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws) + ((c[0].x ^ c[1].x ^ c[2].x ^ c[3].x ^ sw0 ^ sw1) & zom));
__syncwarp();
uint32_t xdep = 0u;
#else
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
#endif
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
#pragma unroll
for (int i = 0; i < 4; ++i) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);   // UNP SMQ: opaque 16 (flags < 2^30), as stage<N>
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
uint32_t hh[8];
#if VCP_LDA || VCP_M1
uint32_t bb[N / 8][4];
#endif
#if VCP_LDA   // VCP (b): x fragments in flight during the unpack
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), bb[nt][0], bb[nt][1], bb[nt][2], bb[nt][3]);
#endif
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);   // UNP SMQ (tools/unp_patches.py)
#if VCP_LDA || VCP_M1
#if !VCP_LDA
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), bb[nt][0], bb[nt][1], bb[nt][2], bb[nt][3]);
#endif
if constexpr (VCP_M1 && N >= 16) {   // VCP (b): one accumulator, the nt chains interleaved
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], bb[nt][0], bb[nt][1]);
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) mma16816(&d2[0][4 * nt], hh[2], hh[6], hh[3], hh[7], bb[nt][2], bb[nt][3]);
} else {
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], bb[nt][0], bb[nt][1]);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], bb[nt][2], bb[nt][3]);
}
}
#else
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
#if SFX
if (i == 3) xdep ^= b0;
#endif
}
#endif
}
}
__syncwarp();
#if SFX   // R-sfx: x slot released after the last k64 step's ldmatrix data returned
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs) + (xdep & zom));
#else
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
#endif
++it;
}
template <bool PB, int N>
__device__ __forceinline__ void epilogue(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d,
Pend& pd) {
const int g = lane >> 2, t = lane & 3, wt = (wp << 5) | lane;
const int e = T.jE[j], cn = T.jC[j];
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
if constexpr (!PB) {
SEGMK(const long long t_a0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs13[e];
#else
const float gs = __ldg(A.gsc13 + e);
#endif
const int npad = (cn + 7) & ~7;
const int col = 8 * tg + g, s2 = col >> 8, kk = col & 255;
const int i = kk >> 6, tt = (kk >> 4) & 3, qq = (kk >> 1) & 7, b = kk & 1;
__nv_bfloat16* xb = A.xB + (size_t)T.jrow[j] * KDIM + (size_t)s2 * npad * 256;
#if VCP_PK   // VCP: lanes g, g ^ 1 (lane ^ 4) hold adjacent bf16 of xB: one 4-B store per pair (same values)
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const __nv_bfloat162 v2 = __floats2bfloat162_rn(swiglu(d[4 * nt] * gs, d[4 * nt + 2] * gs), swiglu(d[4 * nt + 1] * gs, d[4 * nt + 3] * gs));
const uint32_t vs = *reinterpret_cast<const uint32_t*>(&v2);
const uint32_t vo = __shfl_xor_sync(0xffffffffu, vs, 4);
const uint32_t w2 = b ? __byte_perm(vo, vs, 0x7632) : __byte_perm(vs, vo, 0x5410);
const int n = 8 * nt + 2 * t + b;
if (n < cn) *reinterpret_cast<uint32_t*>(xb + ((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + b)) * 4 + tt) * 2) = w2;
}
if (false)
#endif
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn)
xb[((((nt * 4 + i) * 8 + qq) * 8 + (2 * t + hh)) * 4 + tt) * 2 + b] =
__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));
}
SEGMK(if (A.marks && wt == 0 && q == 0) { A.marks[(size_t)blockIdx.x * MKW + 52] += 1; A.marks[(size_t)blockIdx.x * MKW + 53] += gtime() - t_a0; })
if constexpr (VM_DEFER || VM_DEFER2) {
pd.i = j * 16 + (ub >> 1);
} else {
SEGMK(const long long t_f = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
fence_proxy_async();
bar_sync(1 + q, 128);
if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);
SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 63] += gtime() - t_f;)
}
} else {
SEGMK(const long long t_b0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
yb[(size_t)n * KDIM + h0] = d[4 * nt + hh] * gs;
yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;
}
}
SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_b0;)
}
}
#if VM_BRED
template <int N>
__device__ __forceinline__ void bred(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d) {
const int g = lane >> 2, t = lane & 3;
const int e = T.jE[j], cn = T.jC[j];
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
SEGMK(const long long t_r0 = (A.marks && wp == 0 && lane == 0 && q == 0) ? gtime() : 0;)
#if VM_GSM
const float gs = T.gs2[e];
#else
const float gs = __ldg(A.gsc2 + e);
#endif
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0), "f"(d[4 * nt + hh] * gs) : "memory");
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0 + 8), "f"(d[4 * nt + 2 + hh] * gs) : "memory");
}
}
SEGMK(if (A.marks && wp == 0 && lane == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_r0;)
}
#endif
#if VM_MKW == 64
struct SegMk { long long t0, t1, t2, t3; bool on; };
__device__ __forceinline__ void segmk_done(const Args& A, SegMk& m, int s0, int s1) {
if (m.on) {
const long long t4 = gtime();
long long* mk = A.marks + (size_t)blockIdx.x * MKW;
mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;
}
}
#endif
#if VM_AD
template <int N>
__device__ __forceinline__ void m32e_finish(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
float* fb = A.fix + ((size_t)(pd.sslot * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + pd.sslot * 4 + q;
float d[N / 2];
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
if (wt == 0) *fc = 0;
epilogue<false, N>(A, T, pd.su >> 5, pd.su & 31, q, wp, lane, d, pd);
}
__device__ __noinline__ void m32e_resolve(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
int* fc = A.cnt + pd.sslot * 4 + q;
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, pd.scnt) + pd.scnt == 16);
bar_sync(1 + q, 128);
pd.son = 0;
if (!T.wlast[q]) return;
switch (pd.snt) {
case 1: m32e_finish<8>(A, T, pd, q, wp, lane); break;
case 2: m32e_finish<16>(A, T, pd, q, wp, lane); break;
case 3: m32e_finish<24>(A, T, pd, q, wp, lane); break;
case 4: m32e_finish<32>(A, T, pd, q, wp, lane); break;
case 5: m32e_finish<40>(A, T, pd, q, wp, lane); break;
case 6: m32e_finish<48>(A, T, pd, q, wp, lane); break;
case 7: m32e_finish<56>(A, T, pd, q, wp, lane); break;
default: m32e_finish<64>(A, T, pd, q, wp, lane); break;
}
}
#endif
template <bool PB, int N>
__device__ __forceinline__ void seg(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int u, int s0, int s1,
int q, int wp, int lane, Pend& pd) {
float d[N / 2];
SEGMK(SegMk smk{0, 0, 0, 0, A.marks != nullptr && q == 0 && wp == 0 && lane == 0}; if (smk.on) smk.t0 = gtime();)
bool sync_path = false;
#if VM_SYNC7
if constexpr (N <= 16) sync_path = true;
#else
if constexpr (N <= 16) sync_path = (A.flags & 128) != 0;
#endif
if (sync_path) {
float d2[2][N / 2];
#pragma unroll
for (int k = 0; k < N / 2; ++k) { d2[0][k] = 0.f; d2[1][k] = 0.f; }
stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
SEGMK(if (smk.on) smk.t1 = gtime();)
if constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);
SEGMK(if (smk.on) smk.t2 = smk.t3 = gtime();)
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = d2[0][k] + d2[1][k];
} else {
uint32_t hb[VcpCfg<N>::NB][8];
int prevx = -1;
stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);
SEGMK(if (smk.on) smk.t1 = gtime();)
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);
SEGMK(if (smk.on) smk.t2 = gtime();)
wg_wait<0>();
SEGMK(if (smk.on) smk.t3 = gtime();)
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
}
if constexpr (VM_DEFER2) pend_flush(A, pd, q, wp, lane);
constexpr int UBL = PB ? 4 : 5;
const int j = u >> UBL, ub = u & ((1 << UBL) - 1);
if (!(s0 == 0 && s1 == 16)) {
#if VM_BRED
if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); SEGMK(segmk_done(A, smk, s0, s1);) return; }
#endif
const int wt = (wp << 5) | lane;
#if VM_DP
const int s0u = (PB ? T.sB : T.sA) >> 4;
const int slot = (u >= s0u) ? (int)gridDim.x + (u - s0u) : cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTSD + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTSD + slot) * 4 + q;
#else
const int slot = cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
#endif
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
#if VM_AD
if constexpr (!PB) {
if (A.flags & 32768) {
if (pd.son) m32e_resolve(A, T, pd, q, wp, lane);
pd.son = 1; pd.su = u; pd.sslot = slot; pd.snt = N / 8; pd.scnt = s1 - s0; ++pd.stag;
SEGMK(segmk_done(A, smk, s0, s1);)
return;
}
}
#endif
SEGMK(long long t_at = smk.on ? gtime() : 0;)
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);
bar_sync(1 + q, 128);
SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 62] += gtime() - t_at;)
if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }
SEGMK(const long long t_rb = smk.on ? gtime() : 0;)
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
if (wt == 0) *fc = 0;
SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 55] += gtime() - t_rb;)
}
epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);
SEGMK(segmk_done(A, smk, s0, s1);)
}
template <bool PB>
#if VM_DP
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane, int& kq) {
#else
__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {
#endif
const int c = blockIdx.x;
const int g0 = PB ? T.bB[c] : T.bA[c], g1 = PB ? T.bB[c + 1] : T.bA[c + 1], sp = split_point(A, PB, g0, g1);
Pend pd{-1};
#if VM_DP
bool pool = false;
const int S0 = PB ? T.sB : T.sA;
#endif
#pragma unroll 1
for (int gs = sp, ge = g1; ;) {
#if VM_DP
int u, s0, s1;
if (!pool && gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; }
else {
pool = true;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 54 + PB] = gtime();
#endif
}
}
if (pool) {
if (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 5); break; }
}
}
__threadfence_block();
const int v = T.q[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
const int g2 = S0 + v * VM_DPI;
u = g2 >> 4; s0 = g2 & 15; s1 = s0 + VM_DPI;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 52 + PB] += 1;
#endif
} else {
u = gs >> 4; s0 = gs & 15; s1 = min(16, s0 + (ge - gs));
}
#else
if (gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
}
const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
#endif
const int j = PB ? (u >> 4) : (u >> 5);
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) {
long long* mk = A.marks + (size_t)c * MKW;
mk[8 + 16 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
mk[49 + PB] += 1;
if (!(s0 == 0 && s1 == 16)) mk[51] += 1;
}
#else
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;
#endif
#if VM_AD
const int tag0 = pd.stag;
#endif
switch ((T.jC[j] + 7) >> 3) {
case 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 2: seg<PB, 16>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 3: seg<PB, 24>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 4: seg<PB, 32>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 5: seg<PB, 40>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 6: seg<PB, 48>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#if VM_NTM > 8
case 8: seg<PB, 64>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 9
case 9: seg<PB, 72>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 10
case 10: seg<PB, 80>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 11
case 11: seg<PB, 88>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 12
case 12: seg<PB, 96>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 13
case 13: seg<PB, 104>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 14
case 14: seg<PB, 112>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
#if VM_NTM > 15
case 15: seg<PB, 120>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
#endif
default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;
}
#if VM_AD
if constexpr (!PB) { if (pd.son && pd.stag == tag0) m32e_resolve(A, T, pd, q, wp, lane); }
#endif
gs += s1 - s0;
}
#if VM_AD
if constexpr (!PB) { if (pd.son) m32e_resolve(A, T, pd, q, wp, lane); }
#endif
if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);   // before phase B: its x waits on these releases
}

__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {
extern __shared__ __align__(1024) uint8_t smem[];
Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
const uint32_t bars = sa(smem + BAR_OFF);
#if VM_WG5 >= 2
if (warp >= NCW) {   // producer WG: diverge before route / gather (no shared live range crosses the cap change)
asm volatile("setmaxnreg.dec.sync.aligned.u32 " VM_S(VM_PREG) ";\n" ::: "memory");
if (A.flags & 256) asm volatile("griddepcontrol.wait;\n" ::: "memory");
bar_sync(9, NTH);   // the consumers arrive after the mbarrier init + route()
const uint16_t* ent = reinterpret_cast<const uint16_t*>(smem + NS * US + (NX - 1) * XS + 4096);
if (warp > NCW + 1) {
if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1);
if (VM_W5B6) bar_arrive(6, VM_B6);
#if VM_RH
if (warp == NCW + 2 && lane == 0) rh_loop(A, T);   // R-vrh
#endif
return;
}
if (warp == NCW) {
if (A.flags & 4) bar_sync(7, 32 * NCW + 32);
if (lane == 0) prod_w(A, T, sa(smem), bars);
} else {
if (A.flags & 64) gather_w5(A, T, ent, lane, 0);
bar_sync(6, VM_B6);
if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
else if VM_F9(A) bperm_lanes(A, T, lane);
}
return;
}
#if VM_WG5 == 2
asm volatile("setmaxnreg.inc.sync.aligned.u32 " VM_S(VM_CREG) ";\n" ::: "memory");
#endif
#endif
if (tid == 0) {
for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }
#if VM_RH
for (int k = 0; k < 4; ++k) T.rhn[k] = 0;   // R-vrh: before bar 9 (the producer WG waits on it)
#endif
for (int i = 0; i < NX; ++i) { mbar_init(bars + 8 * (2 * NS + i), 1); mbar_init(bars + 8 * (2 * NS + NX + i), NCW); }
asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
if (A.marks) {   // v11: SM id + two L2 latency probes (4 KB apart: H200 partition class), then the start mark
long long* m = A.marks + (size_t)blockIdx.x * MKW;
unsigned smid;
asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
m[MKP] = smid;
m[MKP + 1] = lat_probe(A.ws13, A.flags);
m[MKP + 2] = lat_probe(A.ws13 + 4096, A.flags);
m[0] = gtime();
}
} else if (tid == 32 && (A.flags & 2)) {
const uint32_t tot = (uint32_t)A.M * (KDIM * 2), S = (mdiv(tot + gridDim.x - 1, A.magG) + 15u) & ~15u, off = blockIdx.x * S;
if (off < tot)
asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" :: "l"(reinterpret_cast<const uint8_t*>(A.x) + off),
"r"(min(S, tot - off)) : "memory");
}
if (A.flags & 256) asm volatile("griddepcontrol.wait;\n" ::: "memory");   // PDL: ids / x come from the previous kernel
#if VM_LD == 2
if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // VTR: vm_fin may launch now (after the wait: its pre-wait wts load is safe)
#endif
uint8_t* scr = smem + NS * US + (NX - 1) * XS;
int* hist = reinterpret_cast<int*>(scr);
uint16_t* ent = reinterpret_cast<uint16_t*>(scr + 4096);
route(A, T, hist, ent, reinterpret_cast<uint32_t*>(scr + 12288));
#if VM_WG5 >= 2
bar_arrive(9, NTH);
#endif
if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 7] = gtime();
#if VM_WG5 < 2
if (warp >= NCW) {
#if VM_WG5
asm volatile("setmaxnreg.dec.sync.aligned.u32 " VM_S(VM_PREG) ";\n" ::: "memory");
if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); return; }
#endif
if (warp == NCW) {
if (A.flags & 4) bar_sync(7, 32 * NCW + 32);
if (lane == 0) prod_w(A, T, sa(smem), bars);
} else {
#if VM_WG5
if (A.flags & 64) gather_w5(A, T, ent, lane, 0);
#else
if (A.flags & 64) gather_warp(A, T, ent, lane);
#endif
bar_sync(6, 32 * NCW + 32);
if (lane == 0) prod_x(A, T, sa(smem + NS * US), bars);
else if VM_F9(A) bperm_lanes(A, T, lane);
}
return;
}
#endif
if (A.flags & 64) self_fill(A, T, smem, bars, ent, nfill_of(A, T));
else gather(A, T, ent);
bar_arrive(6, VM_B6);
if (A.flags & 4) bar_arrive(7, 32 * NCW + 32);
#if VM_WG5 == 1 || VM_WG5 == 3
asm volatile("setmaxnreg.inc.sync.aligned.u32 " VM_S(VM_CREG) ";\n" ::: "memory");
#endif
if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 5] = gtime();
#if VM_MKW == 64
if (tid == 0 && A.marks && (T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) + (T.bB[blockIdx.x + 1] - T.bB[blockIdx.x]) > 0) {
mbar_wait(bars, 0, A.err);
A.marks[(size_t)blockIdx.x * MKW + 48] = gtime();
}
#endif
const int q = warp >> 2, wp = warp & 3;
int it = 0;
#if VM_DP
int kq = 0;
phase<false>(A, T, smem, bars, it, q, wp, lane, kq);
#else
phase<false>(A, T, smem, bars, it, q, wp, lane);
#endif
#if VM_RH
if (wp == 0 && lane == 0) rh_push(q, -2);   // R-vrh: this WG's phase A is done (its last flush is pushed)
#endif
if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();
#if VM_LD == 1
if (tid == 0) asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
#if VM_DP
phase<true>(A, T, smem, bars, it, q, wp, lane, kq);
#else
phase<true>(A, T, smem, bars, it, q, wp, lane);
#endif
if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();
}

__global__ void __launch_bounds__(256) vm_fin(const Args A) {
__shared__ int s_r[8], s_r8[8];
__shared__ float s_w[8], s_w8[8];
__shared__ int s_n;
__shared__ float s_ss[2][8][4];
const int t = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#if VM_FT
asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");   // VTR: the next kernel lands + preloads weights now (it still waits for our completion)
#endif
#if VM_FTR
const float wv = (tid < 8) ? __ldg(A.wts + t * 8 + tid) : 0.f;   // vm_main's input: ready before vm_main ran
#endif
float4 w4[4];
#pragma unroll
for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));
asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if VM_MKW == 64
if (tid == 0 && A.marks) atomicMin(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW), (unsigned long long)gtime());
#endif
if (t == 0) {
for (int i = tid; i < JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;
#if VM_DP
if (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }
#endif
}
#if VM_FTR
if (warp == 0) {   // VTR: ballot compaction, order kept (== the serial loop)
const int r = (lane < 8) ? __ldcg(A.rmap + t * 8 + lane) : -1;
const unsigned bm = __ballot_sync(0xffffffffu, r >= 0);
if (r >= 0) { const int pos = __popc(bm & ((1u << lane) - 1u)); s_r[pos] = r; s_w[pos] = wv; }
if (lane == 0) s_n = __popc(bm);
}
__syncthreads();
#else
if (tid < 8) { s_r8[tid] = __ldcg(A.rmap + t * 8 + tid); s_w8[tid] = __ldg(A.wts + t * 8 + tid); }
__syncthreads();
if (tid == 0) {
int n = 0;
#pragma unroll
for (int k = 0; k < 8; ++k) {
const int r = s_r8[k];
if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }
}
s_n = n;
}
__syncthreads();
#endif
const int nloc = s_n;
const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
float a[4][4];
#pragma unroll
for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
int par = 0;
for (int c0 = 0; c0 < nloc; c0 += 4, par ^= 1) {
float4 v[4][4];
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
#pragma unroll
for (int m = 0; m < 4; ++m)
v[jj][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid)) : z4;
}
#if VM_FTR
#pragma unroll
for (int jj = 0; jj < 4; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {   // VTR: products once
v[jj][m].x = bf16r(v[jj][m].x) * w4[m].x; v[jj][m].y = bf16r(v[jj][m].y) * w4[m].y;
v[jj][m].z = bf16r(v[jj][m].z) * w4[m].z; v[jj][m].w = bf16r(v[jj][m].w) * w4[m].w;
}
#endif
float ss[4];
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
#if VM_FTR
f = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;
f = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;
#else
f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;
f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;
#endif
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) {
#pragma unroll
for (int jj = 0; jj < 4; ++jj) s_ss[par][warp][jj] = ss[jj];
}
__syncthreads();
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
if (c0 + jj < nloc) {
float sm = 0.f;
for (int w = 0; w < 8; ++w) sm += s_ss[par][w][jj];
const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * s_w[c0 + jj];
#pragma unroll
for (int m = 0; m < 4; ++m) {
#if VM_FTR
a[m][0] += v[jj][m].x * fr; a[m][1] += v[jj][m].y * fr;
a[m][2] += v[jj][m].z * fr; a[m][3] += v[jj][m].w * fr;
#else
a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;
a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;
#endif
}
}
}
#if VM_BRED
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
if (r >= 0) {
#pragma unroll
for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid), z4);
}
}
#endif
}
#pragma unroll
for (int m = 0; m < 4; ++m) {
__nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
uint2 o;
o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * tid) = o;
}
#if VM_MKW == 64
__syncthreads();
if (tid == 0 && A.marks) atomicMax(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW + 1), (unsigned long long)gtime());
#endif
}
}

void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V) {
vm::Args a;
const int64_t M = V[0], G = V[2], mask = V[4];
a.x = reinterpret_cast<const __nv_bfloat16*>(T[0].data_ptr()); a.ids = T[1].data_ptr<int>(); a.wts = T[2].data_ptr<float>();
a.M = (int)M; a.ls = (int)V[1];
a.ws13 = reinterpret_cast<const uint8_t*>(T[3].data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(T[4].data_ptr());
a.gsc13 = T[5].data_ptr<float>(); a.gsc2 = T[6].data_ptr<float>(); a.nw = T[7].data_ptr<float>();
a.xA = reinterpret_cast<__nv_bfloat16*>(T[8].data_ptr()); a.xB = reinterpret_cast<__nv_bfloat16*>(T[9].data_ptr());
a.y = T[10].data_ptr<float>(); a.fix = T[11].data_ptr<float>(); a.cnt = T[12].data_ptr<int>();
a.xready = T[13].data_ptr<int>(); a.doneA = T[14].data_ptr<int>(); a.rmap = T[15].data_ptr<int>(); a.gbar = T[19].data_ptr<int>();
a.out = reinterpret_cast<__nv_bfloat16*>(T[16].data_ptr()); a.err = T[17].data_ptr<int>();
a.marks = T[18].numel() >= (G + (vm::MKW == 64 ? 1 : 0)) * vm::MKW ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;
#if VM_MKW == 64
a.G0 = (int)G;
#endif
TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, "vml_run: cost table too short");
#if VM_DP
TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, "vml_run: pool fractions missing");
a.poolA = (int)V[6 + 2 * VM_NTM]; a.poolB = (int)V[7 + 2 * VM_NTM];
#endif
#if VM_SC
TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, "vml_run: segment costs missing");
a.segA = (int)V[6 + 2 * VM_NTM]; a.segB = (int)V[7 + 2 * VM_NTM];
#endif
#if VM_PS
TORCH_CHECK((int64_t)V.size() >= 7 + 2 * VM_NTM, "vml_run: producer delay missing");
a.pdel = (int)V[6 + 2 * VM_NTM];
#endif
a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];
for (int k = 0; k < 2 * VM_NTM; ++k) { a.cost[k] = (uint32_t)V[6 + k]; a.mag[k] = ((1ull << 40) + V[6 + k] - 1) / V[6 + k]; }
auto stream = at::cuda::getCurrentCUDAStream();
if (mask & 1) {
cudaFuncSetAttribute(vm::vm_main, cudaFuncAttributeMaxDynamicSharedMemorySize, vm::SMEM);
cudaLaunchConfig_t cm = {};
cm.gridDim = dim3((unsigned)G); cm.blockDim = dim3(vm::NTH); cm.dynamicSmemBytes = vm::SMEM; cm.stream = stream;
cudaLaunchAttribute am[1];
am[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
am[0].val.programmaticStreamSerializationAllowed = 1;
cm.attrs = am; cm.numAttrs = (V[5] & 256) ? 1 : 0;
C10_CUDA_CHECK(cudaLaunchKernelEx(&cm, vm::vm_main, a));
}
if ((mask & 2) && M > 0) {
cudaLaunchConfig_t cfg = {};
cfg.gridDim = dim3((unsigned)M); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
cudaLaunchAttribute at[1];
at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
at[0].val.programmaticStreamSerializationAllowed = 1;
cfg.attrs = at; cfg.numAttrs = 1;
C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));
}
}
std::vector<int64_t> vml_cfg() { return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS}; }
"""
_vml_CUDA = (_vml_CUDA_T.replace("@WGSPECS@", "".join(_vml_wg_spec(n) + _vml_wg_spec64(n) for n in range(8, 129, 8)))
        .replace("@JMAX@", str(_vml_JMAX)).replace("@SLOTS@", str(_vml_SLOTS)).replace("@FIXW@", str(_vml_FIXW))
        )

_vml_CPP = r"""
#include <torch/extension.h>
#include <vector>
void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V);
std::vector<int64_t> vml_cfg();
"""
_vml_FLAGS = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]


def _vml_build(verbose=False, defines=()):
    from torch.utils.cpp_extension import load_inline
    fl = _vml_FLAGS + list(defines)
    tag = hashlib.sha1((_vml_CPP + _vml_CUDA + " ".join(fl)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vml_moe_{tag}", cpp_sources=_vml_CPP, cuda_sources=_vml_CUDA, functions=["vml_run", "vml_cfg"],
                      extra_cuda_cflags=fl, verbose=verbose)
    return mod, time.time() - t0


def _vml_alloc(dev, G):
    z = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=dev)
    xrows = _vml_RMAX + 8 * _vml_JMAX
    assert G <= _vml_SLOTS and G <= 136
    return {"xA": z((xrows * _vml_KDIM,), torch.bfloat16), "xB": z((xrows * _vml_KDIM,), torch.bfloat16),
            "y": z((_vml_RMAX * _vml_KDIM,), torch.float32), "fix": z((2 * (_vml_SLOTS + 384) * 4 * max(_vml_FIXW, 10 * 512),), torch.float32),
            "cnt": z((2 * (_vml_SLOTS + 384) * 4,)), "xready": z((_vml_JMAX,)), "doneA": z((_vml_JMAX * 16,)), "rmap": z((_vml_RMAX,)),
            "err": z((4,)), "gbar": z((2,)), "nomark": z((0,), torch.int64), "G": G, "magG": _vml_mag(G)}


def _vml_cost_ext(mod, cost):
    ntm = _vml_cfg(mod)[1]
    assert 8 <= ntm <= 10, "m32e scratch (fix regions of max(FIXW, 5120) floats) supports NTM 8..10 builds"
    c = list(cost)
    if len(c) == 2 * ntm:
        return c
    assert len(c) == 16 and ntm >= 8, (len(c), ntm)
    out = []
    for pb in (0, 1):
        h = c[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out += h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)]
    return out


def _vml_pool_ext(mod, pool):
    return [int(pool[0]), int(pool[1])] if _vml_cfg(mod)[6] else []


def _vml_seg_ext(mod, seg):
    return [int(seg[0]), int(seg[1])] if _vml_cfg(mod)[7] else []


def _vml_pdel_ext(mod, pdel):
    return [int(pdel)] if _vml_cfg(mod)[10] else []


def _vml_cfg(mod):
    c = getattr(mod, "_m32e_cfg", None)
    if c is None:
        c = tuple(int(v) for v in mod.vml_cfg())
        try:
            mod._m32e_cfg = c
        except Exception:
            pass
    return c


_vml_FLAGS_DEFAULT = 1


def _vml_call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None, pool=(0, 0), seg=(0, 0), pdel=0):
    m = x.shape[0]
    assert m <= _vml_MMAX
    flags = _vml_FLAGS_DEFAULT if flags is None else flags
    if out is None:
        out = torch.empty((m, _vml_KDIM), dtype=torch.bfloat16, device=x.device)
    mod.vml_run([x, ids, wts, k["ws13"], k["ws2"], k["gsc13"], k["gsc2"], k["nw"], sc["xA"], sc["xB"], sc["y"], sc["fix"],
                 sc["cnt"], sc["xready"], sc["doneA"], sc["rmap"], out, sc["err"], sc["nomark"] if marks is None else marks, sc["gbar"]],
                [m, ls, sc["G"], sc["magG"], mask, flags] + _vml_cost_ext(mod, _vml_COST + _vml_COST if cost is None else cost) + _vml_pool_ext(mod, pool) + _vml_seg_ext(mod, seg) + _vml_pdel_ext(mod, pdel))
    return out


# ================================================================ layer VMS (tools/mb_gen10.py: verbatim k3_b23_1.py VMS_CUDA_SRC, UNP L2 layout, built as the layer's _vms_build)
MB_LV_CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t unp_ok32(uint32_t c) {   // c, opaque to NVVM: the 3-way gather stays one IMAD
  uint32_t r;
  asm volatile("mov.b32 %0, %1;" : "=r"(r) : "r"(c));
  return r;
}

#ifndef VMS_NS
#define VMS_NS 3
#endif
#ifndef VMS_MK2
#define VMS_MK2 0
#endif
#ifndef VMS_RO
#define VMS_RO 0
#endif
#ifndef VMS_NRB
#define VMS_NRB 4
#endif
#ifndef VMS_FX
#define VMS_FX 0
#endif
#ifndef VMS_G
#define VMS_G 132
#endif

namespace vms {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = VMS_NRB, NS = VMS_NS, G = VMS_G;
constexpr int WAIT_MAX = 1 << 22;

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* tw;
  const unsigned char* wA; const unsigned char* wB;
  const float* gsA; const float* gsB; const float* nw;
  __nv_bfloat16* inter; float* ybuf; float* ss; int* ctr;
  __nv_bfloat16* out; long long* marks; int m; float eps; int lo;
};

struct __align__(16) Smem {
  uint64_t full[NS];
  uint64_t pfull[NRB];
  uint64_t pempty[NRB];
  int njobs, gen;
  int base[4][NE];
  int jE[MAXJ], jS[MAXJ], jC[MAXJ];
  int ent[MAXP], pos[MAXP];
  float4 red[NRB][NW][32];
};
constexpr int RING_OFF = (int)((sizeof(Smem) + 127) / 128 * 128);
constexpr int SMEM_BYTES = RING_OFF + NS * UB;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gt() { long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void mbar_init(uint64_t* b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(su32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(su32(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t par) {
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n W_%=:\n mbarrier.test_wait.parity.shared.b64 p, [%0], %1;\n @!p bra W_%=;\n}\n"
               :: "r"(su32(b)), "r"(par) : "memory");
#endif
}
__device__ __forceinline__ int ld_acq(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ int ld_rlx(const int* p) {
  int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void cta_sync_compute() { asm volatile("bar.sync 1, %0;\n" :: "n"(NT) : "memory"); }

#ifndef VMS_SH
#define VMS_SH 0   // R-vsh: > 0 = expert-strided shares when 2 <= njobs <= VMS_SH
#endif
#if VMS_SH && (VMS_FX & 4)
#error "VMS_SH needs VMS_FX without bit 4 (pre-route first copies use the contiguous plan)"
#endif
struct Plan { int a0, b0, nA, sh, ka, kaL, kb, kbL; };   // R-vsh: sh = strided shares
__device__ __forceinline__ int vA(const Plan& P, int q) { return P.sh ? (q / P.ka) * RBA + P.kaL + q % P.ka : P.a0 + q; }
__device__ __forceinline__ int vB(const Plan& P, int r) { return P.sh ? (r / P.kb) * RBB + P.kbL + r % P.kb : P.b0 + r; }
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, const Plan& P) {
  if (q < P.nA) { const int v = vA(P, q); return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = vB(P, q - P.nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  const Plan P{a0, b0, nA, 0, 1, 0, 1, 0};
  return unit_src(p, S, q, P);
}

__device__ __forceinline__ void issue(const unsigned char* src, unsigned char* dst, uint64_t* bar, int tid) {
#if __CUDA_ARCH__ >= 900
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(su32(bar)), "r"(UB) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(su32(dst)), "l"(src), "r"(UB), "r"(su32(bar)), "l"(pol) : "memory");
#else
  for (int i = tid; i < UB / 16; i += NT)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(su32(dst + i * 16)), "l"(src + i * 16) : "memory");
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(su32(bar)) : "memory");
#endif
}

__device__ __forceinline__ uint32_t scd(uint32_t w, uint32_t sel, uint32_t k16) {
  return __byte_perm(w, 0u, sel) * k16 + 0x5C005C00u;
}
__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const bool mk = p.marks != nullptr;
#if VMS_MK2
  long long* mrow = p.marks + (size_t)c * 32;
  long long cF = 0, cE = 0, cS = 0, cC = 0, cW = 0, cP = 0, cU = 0, nE = 0, ck;
  if (mk && tid == 0) { mrow[16] = clock64(); unsigned sm; asm volatile("mov.u32 %0, %%smid;" : "=r"(sm)); mrow[21] = sm; }
#else
  long long* mrow = p.marks + (size_t)c * 16;
#endif
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);
#endif
#if VMS_RO
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], NW); mbar_init(&S.pempty[i], 1); }
#else
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }
#endif
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
  }
#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900
  if (warp == 0) asm volatile("" :: "l"(p.ids), "l"(p.ctr), "l"(p.wA), "l"(p.wB), "r"(p.m), "r"(p.lo));
#endif
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
  if (warp == 0) {
    const int m8 = p.m * 8;
    int idv[4]; unsigned mm[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }
    if (tid == 0) S.gen = ld_rlx(p.ctr);
#pragma unroll
    for (int r = 0; r < 4; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      mm[r] = __match_any_sync(0xffffffffu, idv[r]);
      if (idv[r] >= 0 && lane == __ffs(mm[r]) - 1) S.base[r][idv[r]] = __popc(mm[r]);
    }
    __syncwarp();
    int cr[4], tot = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }
    int inc = tot, nj = (tot + 7) >> 3, jinc = nj;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);
      if (lane >= d) { inc += a; jinc += b; }
    }
    const int off = inc - tot, joff = jinc - nj;
    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + 8 * k; S.jC[joff + k] = min(8, tot - 8 * k); }
    int run = off;
#pragma unroll
    for (int r = 0; r < 4; ++r) { S.base[r][lane] = run; run += cr[r]; }
    if (lane == 31) S.njobs = jinc;
    __syncwarp();
#if (VMS_FX & 4) && __CUDA_ARCH__ >= 900
    {
      const int nj0 = __shfl_sync(0xffffffffu, jinc, 31);
      if (lane == 0) {
        const int NA0 = nj0 * RBA, NN0 = nj0 * (RBA + RBB);
        const int a00 = c * NA0 / G, nA0 = (c + 1) * NA0 / G - a00;
        const int t00 = c * NN0 / G, N0 = (c + 1) * NN0 / G - t00;
        for (int q = 0; q < NS && q < N0; ++q) issue(unit_src(p, S, q, nA0, a00, t00 - a00), ring + q * UB, &S.full[q], 0);
      }
    }
#endif
    const unsigned lt = (1u << lane) - 1u;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = r * 32 + lane;
      if (i < m8) {
        if (idv[r] >= 0) { const int pr = S.base[r][idv[r]] + __popc(mm[r] & lt); S.ent[pr] = i; S.pos[i] = pr; }
        else S.pos[i] = -1;
      }
    }
  }
  __syncthreads();
  const int njobs = S.njobs;
  const int NA = njobs * RBA, NN = njobs * (RBA + RBB);
  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;
  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;
  const int b0 = t0 - a0;
#if VMS_SH
  const int shp = (njobs >= 2 && njobs <= VMS_SH) ? 1 : 0;
  const int kaL = c * RBA / G, kbL = c * RBB / G;
  const Plan PL{a0, b0, shp ? njobs * ((c + 1) * RBA / G - kaL) : a1 - a0, shp, (c + 1) * RBA / G - kaL, kaL,
                (c + 1) * RBB / G - kbL, kbL};
  const int nA = PL.nA, N = shp ? njobs * (PL.ka + PL.kb) : t1 - t0;
#else
  const Plan PL{a0, b0, a1 - a0, 0, 1, 0, 1, 0};
  const int nA = a1 - a0, N = t1 - t0;
#endif
  if (mk && tid == 0) mrow[1] = gt();
  if (warp < NW) {
#if !((VMS_FX & 4) && __CUDA_ARCH__ >= 900)
    for (int q = 0; q < NS && q < N; ++q) {
#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, PL), ring + q * UB, &S.full[q], tid);
    }
#endif
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[16];
    int curj = -1;
#if VMS_SH
    int xrow = -3;   // R-vsh: token row this lane's xr holds (-1 zeros, -2 B h, -3 none)
#endif
    int sl = 0; uint32_t par = 0;
    for (int q = 0; q < N; ++q) {
      const bool PB = q >= nA;
      const int v = PB ? vB(PL, q - nA) : vA(PL, q);
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
        curj = j;
        const int jj = PB ? j - 4096 : j;
        const int cnt = S.jC[jj], st = S.jS[jj];
        if (PB) {
          if (mk && tid == 0 && mrow[3] == 0) mrow[3] = gt();
#if VMS_MK2
          ck = clock64();
#endif
#if VMS_FX & 1
          if (lane == 0) {
            int n = 0;
            while (ld_acq(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
#else
          if (lane == 0) {
            int n = 0;
            while (ld_rlx(p.ctr + 8 + jj) < RBA) {
              if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 1); break; }
              __nanosleep(64);
            }
          }
          __syncwarp();
          (void)ld_acq(p.ctr + 8 + jj);
#endif
          if (mk && tid == 0 && mrow[4] == 0) mrow[4] = gt();
#if VMS_MK2
          cW += clock64() - ck;
#endif
        }
#if VMS_SH
        const int nrow = PB ? -2 : (g < cnt ? (S.ent[st + g] >> 3) : -1);
        const bool xsame = !PB && nrow == xrow;   // R-vsh: same token row in this lane -> keep xr
        xrow = nrow;
        if (xsame) {
        } else
#endif
        if (g < cnt) {
          const uint4* src = PB ? reinterpret_cast<const uint4*>(p.inter + (size_t)(st + g) * ID)
                                : reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[st + g] >> 3) * HD);
          src += warp * 64 + 2 * t;
#pragma unroll
          for (int cc = 0; cc < 8; ++cc) {
            xr[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);
            xr[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);
          }
        } else {
#pragma unroll
          for (int cc = 0; cc < 16; ++cc) xr[cc] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
#if VMS_MK2
      ck = clock64();
#endif
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
#if VMS_MK2
      { const long long c2 = clock64(); if (q > 0) cF += c2 - ck; ck = c2; }
      if (mk && tid == 0 && q == nA - 1) mrow[15] = gt();
      if (mk && tid == 0 && q == N - 1) mrow[22] = gt();
#endif
      const unsigned char* slot = ring + sl * UB;
      const uint4 sc = *reinterpret_cast<const uint4*>(slot + UC + warp * 512 + lane * 16);
      float acc[4][4];
#pragma unroll
      for (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * 8 + cc) * 512 + lane * 16);
        const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
        const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
        const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
        const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
        const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                                xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t qw = ww[s];
          const uint32_t M = 0x81C081C0u;
          const uint32_t r0 = qw & M;
          const uint32_t r1 = __funnelshift_l(qw, qw, 10) & M;   // UNP L2 layout (tools/unp_patches.py)
          const uint32_t r2 = __funnelshift_l(qw, qw, 13) & M;
          const uint32_t r3 = ((qw & 0x001B001Bu) * unp_ok32(0x1084u)) & M;
          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
      float4 o;
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
#if VMS_RO
      const int bq = q & (NRB - 1);
#if VMS_MK2
      { const long long c2 = clock64(); cC += c2 - ck; ck = c2; }
#endif
      cta_sync_compute();
#if VMS_MK2
      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }
#endif
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, PL), ring + sl * UB, &S.full[sl], tid);
      }
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
#if VMS_MK2
      { const long long c2 = clock64(); cE += c2 - ck; if (c2 - ck > 200) ++nE; ck = c2; }
#endif
      S.red[bq][warp][lane] = o;
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pfull[bq]);
#else
      const int bq = q & (NRB - 1);
#if VMS_MK2
      { const long long c2 = clock64(); cC += c2 - ck; ck = c2; }
#endif
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
#if VMS_MK2
      { const long long c2 = clock64(); cE += c2 - ck; if (c2 - ck > 200) ++nE; ck = c2; }
#endif
      S.red[bq][warp][lane] = o;
      cta_sync_compute();
#if VMS_MK2
      { const long long c2 = clock64(); cS += c2 - ck; ck = c2; }
#endif
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, PL), ring + sl * UB, &S.full[sl], tid);
      }
      if (tid == 0) mbar_arrive(&S.pfull[bq]);
#endif
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    if (mk && tid == 0) mrow[5] = gt();
#if VMS_MK2
    if (mk && tid == 0) {
      mrow[10] = cF; mrow[11] = cE; mrow[12] = cS; mrow[19] = cC; mrow[18] = cW; mrow[20] = nE;
      mrow[13] = (long long)N | ((long long)nA << 16) | ((long long)njobs << 32);
    }
#endif
  } else {
    const int g = lane >> 2, t = lane & 3;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int q = 0; q < N; ++q) {
      const int bq = q & (NRB - 1);
#if VMS_MK2
      ck = clock64(); if (q > 0) cU += ck - cP;
#endif
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
#if VMS_MK2
      { const long long c2 = clock64(); cF += c2 - ck; cP = c2; }
#endif
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (q < nA) {
        const int v = vA(PL, q), j = v >> 9, rb = v & 511;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsA + S.jE[j]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          if (tk < cnt) {
            const float ga = (h ? sm.y : sm.x) * gs, up = (h ? sm.w : sm.z) * gs;
            const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
            p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
          }
        }
        if (PL.sh ? (q % PL.ka == PL.ka - 1) : (q == nA - 1 || ((v + 1) >> 9) != j)) {     // this CTA's last unit of job j: credit its units
#if VMS_FX & 2
          asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
#else
          __threadfence();
#endif
          __syncwarp();
          if (lane == 0) atomicAdd(p.ctr + 8 + j, PL.sh ? PL.ka : min(a1, (j + 1) * RBA) - max(a0, j * RBA));
          if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
        }
      } else {
        const int v = vB(PL, q - nA), j = v >> 8, rb = v & 255;
        const int cnt = S.jC[j], st = S.jS[j];
        const float gs = __ldg(p.gsB + S.jE[j]);
        const int n0 = rb * 16 + g;
        const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int tk = 2 * t + h;
          const float y0 = (h ? sm.y : sm.x) * w0, y1 = (h ? sm.w : sm.z) * w1;
          float sq = y0 * y0 + y1 * y1;
          if (tk < cnt) {
            p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
            p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
          }
          sq += __shfl_xor_sync(0xffffffffu, sq, 4);
          sq += __shfl_xor_sync(0xffffffffu, sq, 8);
          sq += __shfl_xor_sync(0xffffffffu, sq, 16);
          if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
        }
      }
    }
#if VMS_FX & 2
    asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
#else
    __threadfence();
#endif
    if (mk && lane == 0) mrow[7] = gt();
#if VMS_MK2
    if (mk && lane == 0) { mrow[14] = cF; mrow[23] = cU + (clock64() - cP); }
#endif
  }
  __syncthreads();
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
  const int gen = S.gen;
  if (tid == 0) {
#if VMS_FX & 2
    asm volatile("red.release.gpu.global.add.s32 [%0], 1;\n" :: "l"(p.ctr + 1) : "memory");
#else
    __threadfence();
    atomicAdd(p.ctr + 1, 1);
#endif
    const int target = (gen + 1) * G;
    int n = 0;
    while (ld_acq(p.ctr + 1) < target) {
      if (++n > WAIT_MAX) { atomicExch(p.ctr + 2, 2); break; }
      __nanosleep(32);
    }
    if (mk) mrow[8] = gt();
  }
  __syncthreads();
  if (c == 0) {   // reset for the next launch: every CTA has passed its doneA waits and read gen
    if (tid < MAXJ) p.ctr[8 + tid] = 0;
    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;
    if (tid == 0) atomicAdd(p.ctr, 1);
  }
  const float* ssc = p.ss + (gen & 1) * MAXP;
  for (int it = c; it < p.m * 4; it += G) {
    const int tk = it >> 2, col = (it & 3) * 1024 + (tid & 255) * 4;
    if (tid >= NT) break;
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int pr = S.pos[tk * 8 + s];
      if (pr >= 0) {
        const float cf = __ldg(p.tw + tk * 8 + s) * rsqrtf(__ldcg(ssc + pr) * (1.f / HD) + p.eps);
        const float4 y = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD + col));
        a.x += cf * y.x; a.y += cf * y.y; a.z += cf * y.z; a.w += cf * y.w;
      }
    }
    __nv_bfloat162 o0 = __floats2bfloat162_rn(a.x, a.y), o1 = __floats2bfloat162_rn(a.z, a.w);
    uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
    *reinterpret_cast<uint2*>(p.out + (size_t)tk * HD + col) = pk;
  }
  if (mk && tid == 0) mrow[9] = gt();
#if VMS_MK2
  if (mk && tid == 0) mrow[17] = clock64();
#endif
}
}  // namespace vms

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo) {
  TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vms::vms_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, vms::SMEM_BYTES));
    attr = true;
  }
  vms::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); p.ids = ids.data_ptr<int>(); p.tw = tw.data_ptr<float>();
  p.wA = wA.data_ptr<uint8_t>(); p.wB = wB.data_ptr<uint8_t>(); p.gsA = gsA.data_ptr<float>(); p.gsB = gsB.data_ptr<float>();
  p.nw = nw.data_ptr<float>(); p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); p.ybuf = ybuf.data_ptr<float>();
  p.ss = ss.data_ptr<float>(); p.ctr = ctr.data_ptr<int>(); p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.m = (int)m; p.eps = (float)eps; p.lo = (int)lo;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(vms::G); lc.blockDim = dim3(vms::NTL); lc.dynamicSmemBytes = vms::SMEM_BYTES;
  lc.stream = at::cuda::getCurrentCUDAStream(); lc.attrs = at; lc.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, vms::vms_kernel, p));
}

std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, vms::vms_kernel));
  return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)vms::SMEM_BYTES, vms::G, vms::NS};
}
'''
MB_LV_CPP = r'''
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo);
std::vector<int64_t> attrs();
'''
MB_LV_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((28, 29, 30, 5), (12, 13, 14, 21)),
           ((25, 26, 27, 2), (9, 10, 11, 18)), ((4, 0, 1, 3), (20, 16, 17, 19)))


def _mb_build_lv(G, extra=()):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1"] + list(extra) + ["-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((MB_LV_CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"mblv_{tag}", cpp_sources=MB_LV_CPP, cuda_sources=MB_LV_CUDA, functions=["run", "attrs"],
                       extra_cuda_cflags=flags)


def _mb_lv_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << MB_LV_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _mb_lv_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _mb_lv_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


# ================================================================ mb glue (tools/mb_gen.py)
MB_N = 14
MB_BUILDS = {
    's3': ('vms', 3, 16, 0),
    'xw6': ('vms', 3, 16, 1152, 0, 6),
    'bf2': ('vml', ('-DVM_TRICKS_DQ=0', '-DVM_BRED=1', '-DVM_FSW=1', '-DVM_SYNC7=1', '-DVM_NOB9=1', '-DVM_RT=1', '-DVM_FT=1', '-DVM_FTR=1', '-DVM_TA=1', '-DVM_DEFER=1', '-DVCP_DPI=0', '-DVCP_DPH=0', '-DVCP_PK=1', '-DSFX=1')),
    'w72': ('vml', ('-DVM_TRICKS_DQ=0', '-DVM_BRED=1', '-DVM_FSW=1', '-DVM_SYNC7=1', '-DVM_NOB9=1', '-DVM_WG5=2', '-DVM_PREG=32', '-DVM_CREG=112', '-DVM_GV=2', '-DVM_JN=72', '-DVM_XSN=72', '-DVM_RT=1', '-DVM_FTR=1', '-DVM_LD=2', '-DVM_TA=1', '-DVM_DEFER=1', '-DVCP_DPI=0', '-DVCP_DPH=0', '-DVCP_PK=1', '-DSFX=1')),
    'lv': ('lvms',),
    'lvf1': ('lvms', ('-DVMS_FX=1',)),
    'lvsh': ('lvms', ('-DVMS_SH=4',)),
    'lvshf1': ('lvms', ('-DVMS_SH=4', '-DVMS_FX=1')),
    'lvsh8': ('lvms', ('-DVMS_SH=8',)),
}
MB_ARMS = {
    1: ['lv', 'lvf1', 'lvsh', 'lvshf1', 'lvsh8'],
    8: ['lv', 'lvsh', 'lvsh8', 'lvf1'],
    16: ['lv', 'lvsh8', 's3'],
    32: ['bf2:193:L', 'w72:193:L'],
    128: ['w72:193:L', 'bf2:193:L'],
    256: ['w72:193:L', 'bf2:193:L'],
}
MB_LAYER_COST = {
    256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
    128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
    32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204],
}
MB_LAYER_PROFILES = {
    1: [[1, 1], [1, 1, 1]],
    8: [[3, 2, 2, 1, 1, 1, 1, 1], [2, 1, 1, 1, 1, 1, 1, 1, 1], [3, 2, 2, 2, 2, 2, 1, 1], [3, 2, 2, 2, 1, 1, 1, 1], [4, 2, 2, 2, 1], [4, 2, 2, 2, 1, 1, 1, 1]],
    16: [[14, 7, 5, 5, 3, 2, 1], [14, 12, 4, 2, 1, 1, 1, 1], [9, 4, 4, 4, 3, 1, 1, 1, 1, 1, 1], [14, 7, 5, 4, 2], [12, 10, 7, 3, 3, 1, 1], [11, 9, 5, 3, 3]],
    32: [[26, 11, 7, 6, 3, 3, 2, 1, 1, 1], [17, 17, 4, 2, 1, 1, 1, 1, 1], [22, 19, 11, 4, 3, 3, 3, 1, 1, 1], [23, 13, 11, 10], [24, 16, 10, 6, 4, 3, 3], [19, 15, 6, 5, 4]],
    256: [[122, 83, 45, 39, 30, 28, 15, 13, 9, 6, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1], [107, 91, 44, 36, 35, 10, 9, 6, 6, 5, 5, 5, 4, 3, 2, 2, 1, 1, 1], [81, 76, 65, 45, 43, 31, 27, 25, 24, 9, 9, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1], [72, 67, 50, 47, 41, 41, 28, 22, 20, 16, 6, 4, 2, 1, 1, 1, 1, 1], [107, 84, 38, 28, 26, 26, 17, 9, 8, 8, 7, 5, 5, 4, 4, 2, 2, 2, 1, 1], [117, 100, 48, 36, 25, 13, 8, 8, 8, 7, 5, 5, 3, 2, 2, 1, 1, 1, 1, 1]],
    128: [[49, 27, 27, 21, 13, 13, 8, 6, 4, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1], [50, 45, 15, 14, 12, 7, 4, 4, 3, 2, 2, 2, 1, 1, 1, 1], [47, 37, 32, 28, 20, 19, 17, 11, 8, 5, 3, 2, 2, 2, 1, 1, 1, 1, 1], [43, 26, 25, 25, 16, 12, 10, 10, 9, 8, 3, 2, 2, 2, 1, 1], [53, 41, 27, 24, 18, 12, 6, 6, 5, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1], [54, 49, 28, 21, 20, 7, 6, 5, 3, 3, 2, 2, 2, 2, 2, 1, 1]],
}
MB_BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, 320, 256))
MB_CASES = {1: ((1, 1, "normal"), (1, 2, "skewed"), (4, 3, "normal")),
            8: ((5, 4, "normal"), (8, 5, "normal"), (8, 6, "skewed"), (12, 7, "skewed")),
            16: ((13, 8, "normal"), (16, 9, "normal"), (16, 10, "skewed"), (16, 11, "hot"), (24, 12, "skewed")),
            32: ((25, 13, "normal"), (32, 14, "normal"), (32, 15, "skewed"), (32, 16, "hot"), (33, 17, "normal"),
                 (48, 18, "skewed")),
            128: ((49, 19, "normal"), (128, 20, "normal"), (128, 21, "skewed"), (192, 22, "skewed")),
            256: ((193, 23, "normal"), (256, 24, "normal"), (256, 25, "hot"), (257, 26, "normal"), (320, 27, "skewed"))}
MB_SEEDS = 6         # board-like timing routings per bucket (T <= 16)
MB_SEEDS_L = 16      # ... for T >= 32 (routing spread there is +-15 %)
MB_TUNE_N = 24       # routings per tuned-table objective (separate seeds from the timing set)
MB_REPS = 16         # cold-L2 graph replays per (arm, routing)
MB_FITS = 3          # routings per cost fit
MB_EPS = 1e-8
_MB = {"mods": {}, "berr": {}, "bsec": {}, "route": {}, "fb": {}, "costs": {}, "scr": {}, "t0": 0.0}


def _mb_bucket(m):
    for lo, hi, rep in MB_BUCKETS:
        if lo <= m <= hi:
            return rep
    return None


# ---------------------------------------------------------------- builds (threads: nvcc runs in parallel)
def _mb_build_vms(ns, maxt, fl, G, pad=0, wt=0, extra=()):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", f"-DVMS_NS={ns}", f"-DVMS_MAXT={maxt}", f"-DVMS_FL={fl}", f"-DVMS_PAD={pad}", f"-DVMS_WT={wt}"] + list(extra) + [
             "-Xptxas=-warn-spills",
             "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((MB_VMS_CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"mbs_{tag}", cpp_sources=MB_VMS_CPP, cuda_sources=MB_VMS_CUDA, functions=["run", "attrs"],
                       extra_cuda_cflags=flags)


def _mb_build_one(name, G):
    import threading  # noqa: F401
    t1 = time.time()
    try:
        b = MB_BUILDS[name]
        if b[0] == "lvms":
            _MB["mods"][name] = _mb_build_lv(G, b[1] if len(b) > 1 else ())
        elif b[0] == "vms":
            _MB["mods"][name] = _mb_build_vms(b[1], b[2], b[3], G, b[4] if len(b) > 4 else 0, b[5] if len(b) > 5 else 0,
                                              b[6] if len(b) > 6 else ())
        else:
            _MB["mods"][name], _ = _vml_build(defines=b[1])
    except Exception as ex:  # noqa: BLE001
        _MB["berr"][name] = repr(ex)[-300:]
    _MB["bsec"][name] = time.time() - t1


def _mb_start_builds(G):
    import threading
    ths = []
    for name in MB_BUILDS:
        th = threading.Thread(target=_mb_build_one, args=(name, G), name=f"mb-{name}", daemon=True)
        th.start()
        ths.append(th)
    return ths


# ---------------------------------------------------------------- weights
def _mb_vms_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << MB_VMS_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _mb_vms_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _mb_vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


def _mb_prepack(w, dev):
    """experts [0, 32): VMS units (scale bytes must sit in 0x08..0x7E for its PRMT decode) and VML / K2A v3.1 stages"""
    E = 32
    pk = {}
    b13, b2 = w["w13_sf"][:E].view(torch.uint8), w["w2_sf"][:E].view(torch.uint8)
    if bool(((b13 >= 8) & (b13 <= 0x7E)).all()) and bool(((b2 >= 8) & (b2 <= 0x7E)).all()):
        wA = torch.empty((E, 512, 36864), dtype=torch.uint8, device=dev)
        wB = torch.empty((E, 256, 36864), dtype=torch.uint8, device=dev)
        for e in range(E):
            wA[e] = _mb_vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            wB[e] = _mb_vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        two62 = float(2 ** 62)
        pk["vms"] = {"wA": wA, "wB": wB, "gsA": (w["w13_gs"][:E].float() * two62).contiguous(),
                     "gsB": (w["w2_gs"][:E].float() * two62).contiguous(), "nw": (1.0 + w["post_norm_w"].float()).contiguous()}
        lA = torch.empty((E, 512, 36864), dtype=torch.uint8, device=dev)   # mb_gen10: layer VMS layout (UNP L2)
        lB = torch.empty((E, 256, 36864), dtype=torch.uint8, device=dev)
        for e in range(E):
            lA[e] = _mb_lv_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            lB[e] = _mb_lv_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        pk["lvms"] = dict(pk["vms"], wA=lA, wB=lB)
    else:
        print("[mb] VMS off: block-scale bytes outside 0x08..0x7E", flush=True)
    ws13 = torch.empty((E, 64, 16, 18432), dtype=torch.uint8, device=dev)
    ws2 = torch.empty((E, 32, 16, 18432), dtype=torch.uint8, device=dev)
    for e in range(E):
        w13q, s13q = _k2a31_prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=K2A31_BITPOS3)
        w2q, s2q = _k2a31_prepack_expert(w["w2"][e], w["w2_sf"][e], gate_up=False, bitpos=K2A31_BITPOS3)
        ws13[e] = _k2a33_stage_repack(w13q, s13q)
        ws2[e] = _k2a33_stage_repack(w2q, s2q)
    pk["vml"] = {"ws13": ws13, "ws2": ws2, "gsc13": (w["w13_gs"][:E].float() * 256.0).contiguous(),
                 "gsc2": (w["w2_gs"][:E].float() * 256.0).contiguous(),
                 "nw": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()}
    torch.cuda.synchronize()
    return pk


def _mb_scratch(name, dev, G):
    b = MB_BUILDS[name]
    if b[0] in ("vms", "lvms"):
        mp = 8 * (b[2] if b[0] == "vms" else 16)
        return {"inter": torch.zeros((mp, 4096), dtype=torch.bfloat16, device=dev),
                "ybuf": torch.zeros((mp, 4096), dtype=torch.float32, device=dev),
                "ss": torch.zeros((2 * mp,), dtype=torch.float32, device=dev),
                "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
                "nomarks": torch.empty((0,), dtype=torch.int64, device=dev), "maxt": b[2] if b[0] == "vms" else 16}
    sc = _vml_alloc(dev, G)
    sc["maxt"] = _vml_MMAX
    return sc


def _mb_reset(name):
    sc = _MB["scr"][name]
    for k, v in sc.items():
        if isinstance(v, torch.Tensor) and k not in ("nomark", "nomarks", "xA", "xB", "y", "inter", "ybuf"):
            v.zero_()
    torch.cuda.synchronize()


# ---------------------------------------------------------------- calls
def _mb_arm(spec):
    p = spec.split(":")
    return {"spec": spec, "b": p[0], "kind": MB_BUILDS[p[0]][0], "flags": int(p[1]) if len(p) > 1 else 0,
            "ck": p[2] if len(p) > 2 else None}


def _mb_cost(arm, rep):
    if arm["kind"] != "vml":
        return None
    ck = arm["ck"]
    if ck in (None, "d"):
        return None
    if ck == "L":
        return MB_LAYER_COST.get(rep)
    if ck.startswith("Lx") and rep in MB_LAYER_COST:   # mb_gen11: L with every nt >= 2 entry scaled by pct / 100
        f, t = int(ck[2:]) / 100.0, MB_LAYER_COST[rep]
        return [t[0]] + [int(round(v * f)) for v in t[1:8]] + [t[8]] + [int(round(v * f)) for v in t[9:16]]
    return _MB["costs"].get((rep, arm["b"], arm["flags"], ck))


def _mb_call(arm, st, x, ids, tw, out, rep, marks=None, cost=None):
    sc = _MB["scr"][arm["b"]]
    mod = _MB["mods"][arm["b"]]
    m = x.shape[0]
    if arm["kind"] == "lvms":
        pk = st["pk"]["lvms"]
        mod.run(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"],
                out, sc["nomarks"] if marks is None else marks, m, MB_EPS, 0, 0)
        return out
    if arm["kind"] == "vms":
        pk = st["pk"]["vms"]
        mod.run(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"],
                out, sc["nomarks"] if marks is None else marks, m, MB_EPS, 0, 0, 0)
        return out
    return _vml_call(mod, st["pk"]["vml"], sc, x, ids, tw, 0, marks=marks, out=out, flags=arm["flags"],
                     cost=_mb_cost(arm, rep) if cost is None else cost)


def _mb_err(arm):
    sc = _MB["scr"][arm["b"]]
    return int(sc["ctr"][2]) if arm["kind"] in ("vms", "lvms") else int(sc["err"][0])


def _mb_torch(data, st):
    """FP32 torch fallback (correct, slow): only when no kernel arm passed for this shape"""
    w = st["w"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    lo = int(data["config"]["local_start"])
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < lo + 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + MB_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def kernel_fn(data, state):
    x = data["hidden_states"]
    m = x.shape[0]
    rep = _mb_bucket(m)
    arm = _MB["route"].get(rep)
    if arm is not None and int(data["config"]["local_start"]) == 0:
        if m > _MB["scr"][arm["b"]]["maxt"]:
            arm = _MB["fb"].get(rep)
        if arm is not None:
            x = x.contiguous()
            if x.data_ptr() % 16:
                x = x.clone()
            out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
            return _mb_call(arm, state, x, data["topk_ids"].to(torch.int32).contiguous(),
                            data["topk_weights"].to(torch.float32).contiguous(), out, rep)
    return _mb_torch(data, state)


# ---------------------------------------------------------------- inputs (synthetic; prepare only)
def _mb_board_input(bs, seed, dev, dist="normal"):
    """generate_input's routing distribution (normal: uniform + 0.03 x Student-t popularity; skewed: + rand^6 x 4;
    hot: 8 experts take every token), busiest GPU relabelled to experts 0-31, our own seeds"""
    gen = torch.Generator(device=dev).manual_seed(7919 * seed + 13 * bs + 1)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=dev)
        u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    elif dist == "skewed":
        scores = scores + torch.rand((256,), generator=gen, device=dev).pow(6) * 4
    else:
        scores[:, :8] += 10.0
    top = scores.topk(8, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "local_start": 0, "local_count": 32}}


def _mb_layer_input(bs, counts, seed, dev):
    """a routing with the given per-expert token counts (a real layer profile), experts placed at random local ids"""
    import random
    rnd = random.Random(1009 * seed + bs)
    loc = list(range(32))
    rnd.shuffle(loc)
    free = [8] * bs
    ids = [[-1] * 8 for _ in range(bs)]
    for k, c in enumerate(sorted(counts, reverse=True)):
        e = loc[k]
        order = sorted(range(bs), key=lambda t: (-free[t], rnd.random()))[:c]
        for t in order:
            ids[t][8 - free[t]] = e
            free[t] -= 1
    for t in range(bs):
        rem = rnd.sample(range(32, 256), 8)
        for s in range(8):
            if ids[t][s] < 0:
                ids[t][s] = rem[s]
    gen = torch.Generator(device=dev).manual_seed(4241 * seed + bs)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    tw = torch.rand((bs, 8), generator=gen, device=dev) + 0.05
    return {"hidden_states": x, "topk_ids": torch.tensor(ids, dtype=torch.int32, device=dev),
            "topk_weights": (tw / tw.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "local_start": 0, "local_count": 32}}


class _MbRef:
    """FP32 reference with dequantized expert weights cached for the self-tests"""
    def __init__(self, w):
        self.w, self.c = w, {}

    def deq(self, e):
        if e not in self.c:
            w = self.w
            self.c[e] = (_k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                         _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.c[e]

    def __call__(self, d):
        w = self.w
        x = d["hidden_states"].float()
        ids = d["topk_ids"].long()
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in torch.unique(ids).tolist():
            if not 0 <= e < 32:
                continue
            tok, slot = (ids == e).nonzero(as_tuple=True)
            a, b = self.deq(e)
            h = x[tok] @ a.T
            inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
            y = (inter @ b.T) * (1.0 + w["post_norm_w"].float())
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + MB_EPS)
            out.index_add_(0, tok, y * d["topk_weights"][tok, slot].float().unsqueeze(1))
        return out.to(torch.bfloat16)


# ---------------------------------------------------------------- VML cost fit from per-CTA marks (public t_fit2 idea)
def _mb_fit(st, arm, rep, ds, base, G, flush, sink, stream):
    marks = torch.zeros((G * 32,), dtype=torch.int64, device=st["dev"])
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    res = []
    for pb in (0, 1):
        X, Y = [], []
        for d in ds:
            marks.zero_()
            with torch.cuda.stream(stream):
                torch.sum(flush, dim=0, out=sink)
                _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:d["hidden_states"].shape[0]],
                         rep, marks=marks, cost=base)
            torch.cuda.synchronize()
            mv = marks.view(G, 32).cpu().double()
            t0 = mv[:, 0].min()
            aend = mv[:, 6] - t0
            end = mv[:, 1:5].max(1).values - t0
            X.append(torch.cat([torch.ones(G, 1, dtype=torch.float64), mv[:, 8 + 8 * pb:16 + 8 * pb]], 1))
            Y.append(aend if pb == 0 else end - aend)
        X, Y = torch.cat(X), torch.cat(Y) / 1e3
        seen = (X[:, 1:] > 0).any(0)
        ks = [k for k in range(8) if bool(seen[k])]
        coef = torch.linalg.lstsq(X[:, [0] + [1 + k for k in ks]], Y.unsqueeze(1)).solution.squeeze(1)
        c = {k: float(coef[i + 1]) for i, k in enumerate(ks)}
        ref = max(c, key=lambda k: float(X[:, 1 + k].sum())) if c else 0
        b0 = list(_vml_COST) if base is None else list(base[8 * pb:8 * pb + 8])
        tab = list(b0)
        if c and c[ref] > 0:
            for k, v in c.items():
                if v > 0:
                    tab[k] = int(min(255, max(16, round(v / c[ref] * b0[ref]))))
        res += tab
    return res


def _mb_eager_us(st, arm, rep, ds, cost, flush, sink, stream, reps=3):
    """mean over routings of the median of `reps` cold-L2 eager calls (the flush keeps the GPU busy while Python launches)"""
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    evs = []
    with torch.cuda.stream(stream):
        for _ in range(reps):
            for i, d in enumerate(ds):
                torch.sum(flush, dim=0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:d["hidden_states"].shape[0]],
                         rep, cost=cost)
                e1.record()
                evs.append((i, e0, e1))
    torch.cuda.synchronize()
    per = {}
    for i, e0, e1 in evs:
        per.setdefault(i, []).append(e0.elapsed_time(e1) * 1e3)
    return sum(sorted(v)[len(v) // 2] for v in per.values()) / len(per)


def _mb_ntiles(ds, flags):
    """cost-table entries (A: nt - 1, B: 8 + nt - 1) that the jobs of these routings use (VML job rule, bit 10 = chunks)"""
    ks = set()
    for d in ds:
        ids = d["topk_ids"].long()
        c = torch.bincount(ids[(ids >= 0) & (ids < 32)].flatten(), minlength=32).tolist()
        for tot in c:
            if tot == 0:
                continue
            nj = (tot + 63) // 64
            base = 64 if (flags & 1024) else -(-tot // nj)
            for k in range(nj):
                nt = (min(base, tot - k * base) + 7) // 8
                ks.update((nt - 1, 8 + nt - 1))
    return sorted(ks)


def _mb_tune(st, arm, rep, ds, start, seen, flush, sink, stream, budget_s=12.0):
    """coordinate descent on the stage-cost table entries of the n-tiles present (x0.92 / x1.08 steps, keep a step only if
    the mean cold-L2 time over `ds` drops by > 0.3 %), inside a time budget; returns (table, us start, us end, steps kept)"""
    t1 = time.time()
    cur = list(start)
    best = _mb_eager_us(st, arm, rep, ds, cur, flush, sink, stream, reps=1)
    b0, kept = best, 0
    for _ in range(2):
        for k in seen:
            for f in (0.92, 1.08):
                if time.time() - t1 > budget_s:
                    return cur, b0, best, kept
                cand = list(cur)
                cand[k] = int(min(255, max(16, round(cur[k] * f))))
                if cand[k] == cur[k]:
                    continue
                v = _mb_eager_us(st, arm, rep, ds, cand, flush, sink, stream, reps=1)
                if v < best * 0.997:
                    cur, best, kept = cand, v, kept + 1
                    break
    return cur, b0, best, kept


def _mb_marks(st, arm, rep, ds, G, flush, sink, stream):
    """per-CTA phase marks of one arm on cold-L2 eager calls -> one compact line (p50 / max over CTAs and routings, us after
    the earliest CTA entry). VML: route, gather (x ready), A end (WG 0), end (last WG), B doneA wait, A xready wait (both
    durations, x-producer warp). VMS: routed, first unit landed, first B wait start / end, compute end, helper A end / end,
    grid barrier passed, final pass done."""
    vms = arm["kind"] == "vms"
    W = 64 if vms else 32
    marks = torch.zeros((G * W,), dtype=torch.int64, device=st["dev"])
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    per = []   # one dict per routing: mark name -> per-CTA values (us after that routing's earliest CTA entry)
    for d in ds:
        m = d["hidden_states"].shape[0]
        if m > _MB["scr"][arm["b"]]["maxt"]:
            continue
        marks.zero_()
        with torch.cuda.stream(stream):
            torch.sum(flush, dim=0, out=sink)
            _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:m], rep, marks=marks)
        torch.cuda.synchronize()
        mv = marks.view(G, W).cpu().double()
        t0 = float(mv[:, 0].min())
        acc = {}
        if vms:
            cols = {"route": 1, "first": 2, "bw0": 3, "bw1": 4, "comp": 5, "hA": 6, "hend": 7, "bar": 8, "fin": 9}
            for k, c in cols.items():
                acc[k] = [(float(x) - t0) / 1e3 for x in mv[:, c] if x > 0]
        else:
            for k, c in (("route", 7), ("gather", 5), ("Aend", 6)):
                acc[k] = [(float(x) - t0) / 1e3 for x in mv[:, c] if x > 0]
            acc["end"] = [(float(x) - t0) / 1e3 for x in mv[:, 1:5].max(1).values]
            acc["Bwait"] = [float(x) / 1e3 for x in mv[:, 24]]
            acc["Xwait"] = [float(x) / 1e3 for x in mv[:, 26]]
        per.append(acc)
    _mb_reset(arm["b"])

    def q(k):
        p50, mx = [], []
        for acc in per:
            v = sorted(acc.get(k, []))
            if v:
                p50.append(v[len(v) // 2])
                mx.append(v[-1])
        return f"{sum(p50) / len(p50):.1f}/{sum(mx) / len(mx):.1f}" if p50 else "-"
    return " ".join(f"{k} {q(k)}" for k in (per[0] if per else {}))


# ---------------------------------------------------------------- prepare
def _mb_rel(o, ref):
    tot, row = _k2a_rel_err(o, ref)
    return tot, row


def prepare(weights):
    _MB["t0"] = t0 = time.time()
    dev = weights["w13"].device
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    ths = _mb_start_builds(G)
    w = {k: weights[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_norm_w")}
    st = {"dev": dev, "G": G, "w": w, "pk": {}}
    try:
        st["pk"] = _mb_prepack(w, dev)
    except Exception as ex:  # noqa: BLE001
        print(f"[mb] prepack raised {ex!r}"[:300], flush=True)
    tp = time.time() - t0
    for th in ths:
        th.join()
    bl = " ".join(f"{k}={v:.0f}s" for k, v in sorted(_MB["bsec"].items()))
    print(f"[mb] v{MB_N}: G={G} prepack {tp:.0f}s builds {bl} errors {_MB['berr'] or 'none'}", flush=True)
    for name, mod in _MB["mods"].items():
        if MB_BUILDS[name][0] in ("vms", "lvms"):
            try:
                print(f"[mb] build {name} {MB_BUILDS[name]}: regs/local/smem/G/NS {list(mod.attrs())}", flush=True)
            except Exception:  # noqa: BLE001
                pass
    for name in list(_MB["mods"]):
        need = MB_BUILDS[name][0] if MB_BUILDS[name][0] in ("vms", "lvms") else "vml"
        if need not in st["pk"]:
            del _MB["mods"][name]
            continue
        _MB["scr"][name] = _mb_scratch(name, dev, G)
    try:
        _mb_sweep(st, G)
    except Exception as ex:  # noqa: BLE001
        print(f"[mb] sweep raised {ex!r}"[:400], flush=True)
    print(f"[mb] routes { {r: a['spec'] for r, a in _MB['route'].items()} } fallbacks "
          f"{ {r: a['spec'] for r, a in _MB['fb'].items()} } prepare {time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    return st


def _mb_sweep(st, G):
    dev = st["dev"]
    ref = _MbRef(st["w"])
    flush = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream()
    for rep, specs in MB_ARMS.items():
        arms = [_mb_arm(s) for s in specs if s.split(":")[0] in _MB["mods"]]
        if not arms:
            continue
        board = [_mb_board_input(rep, 100 + i, dev) for i in range(MB_SEEDS if rep <= 16 else MB_SEEDS_L)]
        layer = [_mb_layer_input(rep, c, 200 + i, dev) for i, c in enumerate(MB_LAYER_PROFILES.get(rep, []))]
        # 1. cost fits (VML arms with ck f / f2 / g) and tuned tables (t: coordinate descent from f on separate routings)
        tune = [_mb_board_input(rep, 300 + i, dev) for i in range(MB_TUNE_N)]
        for a in arms:
            if a["kind"] != "vml" or a["ck"] not in ("f", "f2", "g", "t"):
                continue
            key = (rep, a["b"], a["flags"])
            try:
                if a["ck"] in ("f", "f2") or (a["ck"] == "t" and rep not in MB_LAYER_COST):
                  if key + ("f",) not in _MB["costs"]:
                    _MB["costs"][key + ("f",)] = _mb_fit(st, a, rep, board[:MB_FITS], None, G, flush, sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "f2" and key + ("f2",) not in _MB["costs"]:
                    _MB["costs"][key + ("f2",)] = _mb_fit(st, a, rep, board[:MB_FITS], _MB["costs"][key + ("f",)], G, flush,
                                                          sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "g" and key + ("g",) not in _MB["costs"] and layer:
                    _MB["costs"][key + ("g",)] = _mb_fit(st, a, rep, layer[:MB_FITS], None, G, flush, sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "t" and key + ("t",) not in _MB["costs"]:
                    t_start = list(MB_LAYER_COST[rep]) if rep in MB_LAYER_COST else _MB["costs"][key + ("f",)]
                    tab, u0, u1, kept = _mb_tune(st, a, rep, tune, t_start, _mb_ntiles(tune, a["flags"]), flush, sink, stream)
                    _MB["costs"][key + ("t",)] = tab
                    print(f"[mb] b{rep} tune {a['b']}:{a['flags']}: {u0:.1f} -> {u1:.1f} us (kept {kept})", flush=True)
                    _mb_reset(a["b"])
            except Exception as ex:  # noqa: BLE001
                print(f"[mb] b{rep} fit {a['spec']} raised {ex!r}"[:300], flush=True)
                _mb_reset(a["b"])
        for k, v in _MB["costs"].items():
            if k[0] == rep:
                print(f"[mb] b{rep} cost {k[1]}:{k[2]}:{k[3]} A {v[:8]} B {v[8:]}", flush=True)
        # 2. self-tests
        good = {a["spec"]: True for a in arms}
        for bs, seed, dist in MB_CASES[rep]:
            d = _mb_board_input(bs, seed, dev, dist)
            r = ref(d)
            out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
            for a in arms:
                if not good[a["spec"]] or bs > _MB["scr"][a["b"]]["maxt"]:
                    continue
                if a["kind"] == "vml" and a["ck"] not in (None, "d", "L") and not str(a["ck"]).startswith("Lx") and _mb_cost(a, rep) is None:
                    good[a["spec"]] = False
                    continue
                try:
                    for k in range(2):
                        o = _mb_call(a, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out, rep)
                        torch.cuda.synchronize()
                        tot, row = _mb_rel(o, r)
                        e = _mb_err(a)
                        ok = tot < 1e-2 and row < 5e-2 and e == 0 and bool(torch.isfinite(o.float()).all())
                        if not ok:
                            print(f"[mb] b{rep} FAIL {a['spec']} bs={bs} {dist} call {k}: rel {tot:.2e} row {row:.2e} err {e}",
                                  flush=True)
                            good[a["spec"]] = False
                            _mb_reset(a["b"])
                            break
                except Exception as ex:  # noqa: BLE001
                    print(f"[mb] b{rep} FAIL {a['spec']} bs={bs} raised {ex!r}"[:300], flush=True)
                    good[a["spec"]] = False
                    _mb_reset(a["b"])
        ok_arms = [a for a in arms if good[a["spec"]]]
        if not ok_arms:
            print(f"[mb] b{rep}: no arm passed", flush=True)
            continue
        # 3. timing: cold-L2 CUDA graph replays, rotated order
        sets = [("brd", board), ("lay", layer)]
        graphs = []
        for a in ok_arms:
            for sname, ds in sets:
                for i, d in enumerate(ds):
                    m = d["hidden_states"].shape[0]
                    if m > _MB["scr"][a["b"]]["maxt"]:
                        continue
                    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
                    fn = (lambda a=a, d=d, out=out: _mb_call(a, st, d["hidden_states"], d["topk_ids"], d["topk_weights"],
                                                             out, rep))
                    with torch.cuda.stream(stream):
                        fn()
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=stream):
                            fn()
                    torch.cuda.synchronize()
                    graphs.append((a["spec"], sname, i, g))
        ts = {(s, n, i): [] for s, n, i, _ in graphs}
        for rd in range(MB_REPS):
            k0 = (rd * 7) % len(graphs)
            evs = []
            with torch.cuda.stream(stream):
                for s, n, i, g in graphs[k0:] + graphs[:k0]:
                    torch.sum(flush, dim=0, out=sink)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    evs.append(((s, n, i), e0, e1))
            torch.cuda.synchronize()
            for key, e0, e1 in evs:
                ts[key].append(e0.elapsed_time(e1) * 1e3)
        del graphs
        med = {k: sorted(v)[len(v) // 2] for k, v in ts.items()}
        res = {}
        for a in ok_arms:
            for sname, _ in sets:
                v = [x for (s, n, i), x in med.items() if s == a["spec"] and n == sname]
                res[(a["spec"], sname)] = (sum(v) / len(v), min(v), max(v), len(v)) if v else None
        for a in ok_arms:
            # error flags after the timing replays (a bounded wait that expired leaves a flag, never a hang)
            if _mb_err(a) != 0:
                print(f"[mb] b{rep} {a['spec']}: error flag {_mb_err(a)} after timing -> excluded", flush=True)
                good[a["spec"]] = False
                _mb_reset(a["b"])
        ok_arms = [a for a in ok_arms if good[a["spec"]]]
        best = min((a for a in ok_arms if res[(a["spec"], "brd")] is not None), key=lambda a: res[(a["spec"], "brd")][0],
                   default=None)
        vml_ok = [a for a in ok_arms if a["kind"] == "vml" and res[(a["spec"], "brd")] is not None]
        fb = min(vml_ok, key=lambda a: res[(a["spec"], "brd")][0]) if vml_ok else None
        ref_arm = ok_arms[0]

        def paired(a, sname):
            """mean over routings of (this arm / reference arm) per-routing medians, and its standard error"""
            rs = [x / med[(ref_arm["spec"], n, i)] for (s, n, i), x in med.items()
                  if s == a["spec"] and n == sname and (ref_arm["spec"], n, i) in med]
            if not rs:
                return "-"
            mu = sum(rs) / len(rs)
            se = (sum((r - mu) ** 2 for r in rs) / max(1, len(rs) - 1) / len(rs)) ** 0.5
            return f"r{mu:.4f}+-{se:.4f}"
        print(f"[mb] b{rep} ref {ref_arm['spec']}; r = arm / ref per routing (lower is faster), mean +- se", flush=True)
        for a in sorted(ok_arms, key=lambda a: res[(a["spec"], "brd")][0] if res[(a["spec"], "brd")] else 1e9):
            rb, rl = res[(a["spec"], "brd")], res[(a["spec"], "lay")]
            print(f"[mb] b{rep} {a['spec']:<10} brd {rb[0]:7.1f} {paired(a, 'brd')} n{rb[3]}"
                  + (f" | lay {rl[0]:7.1f} {paired(a, 'lay')} n{rl[3]}" if rl else " | lay -"), flush=True)
        bad = [s for s, v in good.items() if not v]
        if bad:
            print(f"[mb] b{rep} failed: {bad}", flush=True)
        if best is not None:
            _MB["route"][rep] = best
        if fb is not None:
            _MB["fb"][rep] = fb
        diag = [a for a in (best, fb) if a is not None]
        vms_ok = [a for a in ok_arms if a["kind"] == "vms" and res[(a["spec"], "brd")] is not None]
        if vms_ok:
            diag += vms_ok if rep == 16 else [min(vms_ok, key=lambda a: res[(a["spec"], "brd")][0])]
        seen_d = set()
        for a in diag:
            if a["spec"] in seen_d:
                continue
            seen_d.add(a["spec"])
            try:
                print(f"[mb] b{rep} marks brd {a['spec']}: {_mb_marks(st, a, rep, board[:3], G, flush, sink, stream)}",
                      flush=True)
                if layer and (a["kind"] == "vml" or rep == 16):
                    print(f"[mb] b{rep} marks lay {a['spec']}: {_mb_marks(st, a, rep, layer[:3], G, flush, sink, stream)}",
                          flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[mb] b{rep} marks {a['spec']} raised {ex!r}"[:200], flush=True)
                _mb_reset(a["b"])
        torch.cuda.synchronize()
    del flush
    ref.c.clear()

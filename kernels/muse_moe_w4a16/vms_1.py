# okc: problem=muse_moe_w4a16
# vms-board-1: our layer MoE (K2A, board copy of k3_v35_pt1's) + VMS one-launch MoE at bs <= 16 (notes/study/vm_small.md); check only, never scored.
"""study-a sa_k35moe: our layer MoE (K2A v3.3 / v3.6 / v3.7 routes + v4.x levers + KW1, as in k3_v35_pt1.py) on the muse_moe_w4a16 board."""
import hashlib
import time
import torch

R1_LOCAL_ID_TABLE = True

N_EXPERTS = 256

K2A_E_LOC = 32

K2A_D_HIDDEN = 4096

K2A_D_EXPERT = 4096

K2A_TOP_K = 8

K2A_POST_NORM_EPS = 1e-8

K2A_M_MAX = 320

K2A_META_HDR = 128

K2A_BITPOS = ((6, 7, 8, 15), (22, 23, 24, 31), (9, 10, 11, 18), (25, 26, 27, 2),
          (12, 13, 14, 21), (28, 29, 30, 5), (0, 1, 3, 4), (16, 17, 19, 20))

K2A_W_SHIFT = 256.0

K2A_BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, K2A_M_MAX, 256))

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

def _k2a_bucket_of(m: int):
    for lo, hi, rep in K2A_BUCKETS:
        if lo <= m <= hi:
            return rep
    return None

_K2A_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

def _k2a_dequant(packed, sf, gs):
    grid = torch.tensor(_K2A_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)

def _k2a_ref_fp32(data, w):
    cfg = data["config"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    lo, hi = cfg["local_start"], cfg["local_start"] + cfg["local_count"]
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < hi:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :K2A_D_EXPERT]) * h[:, K2A_D_EXPERT:]
        y = (inter @ _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + K2A_POST_NORM_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)

def _k2a_rel_err(a, b):
    a, b = a.float(), b.float()
    tot = float((a - b).norm() / b.norm().clamp(min=1e-12))
    rn = b.norm(dim=-1)
    row = float(((a - b).norm(dim=-1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-12)).max())
    return tot, row

def _k2a_synth_inputs(bs, seed, device, dist="normal"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, K2A_D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=device)
        u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=gen, device=device).pow(6) * 4
    top = scores.topk(K2A_TOP_K, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": K2A_D_HIDDEN, "d_expert": K2A_D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": K2A_E_LOC, "top_k": K2A_TOP_K, "sharding": "ep8"},
    }

def _k2a_self_test_inputs(bs, seed, device, dist="normal"):
    """k2a_v22.py's synthetic inputs (this GPU's experts are 0..31, lo = 0), with remote ids mapped to -1 when
    R1_LOCAL_ID_TABLE is on: the id convention the layer actually hands K2A (r1's device-side local-id table)."""
    d = _k2a_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d

K2A31_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
           (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))

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

K2A33_HOT_T = 16

K2A33_HOT_W4 = 8

K2A33_M3 = 320

K2A33_SW3 = 18432

K2A33_VARIANTS = [(1, 3, 2, 0), (2, 3, 2, 0), (2, 2, 2, 1), (1, 3, 2, 2, 4), (1, 3, 2, 3, 1), (1, 3, 2, 3, 4)]

K2A33_PLAN = {1: 0, 8: 0, 16: 0, 32: 1, 128: 2, 256: 2}

K2A33_HOT_T_BUCKET = {128: 16, 256: 16}

def _k2a33_stage_repack(wq: torch.Tensor, sq: torch.Tensor) -> torch.Tensor:
    """v2.2 (wq int32 [rt, 64, 32, 4], sq uint8 [rt, 32, 32, 4]) -> uint8 [rt/8, 16, 18432]: stage (rb, s) =
    [tile w 8][kb i 4][lane][16 B] = wq[8rb+w][4s+i] then [w 8][kp 2][lane][4 B] = sq[8rb+w][2s+kp]. Lossless."""
    rt = wq.shape[0]
    nrb = rt // 8
    wb = wq.contiguous().view(torch.uint8).view(nrb, 8, 16, 4, 512).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 16384)
    sb = sq.contiguous().view(nrb, 8, 16, 2, 128).permute(0, 2, 1, 3, 4).reshape(nrb, 16, 2048)
    return torch.cat((wb, sb), dim=-1).contiguous()

K2A33_KERNEL_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace k3 {

constexpr int KDIM = 4096;
constexpr int M3 = 320;
constexpr int P3 = 8 * M3;
constexpr int SW = 18432;
constexpr int XROW = 528;            // bytes per staged X row (512 + 16 pad: conflict-free LDS.128)
constexpr int XST = XROW / 2;        // bf16 elements per staged X row
constexpr int MH = 128;
constexpr unsigned FULL = 0xffffffffu;
#ifndef K3_NW_DISPATCH
#define K3_NW_DISPATCH 1
#endif
#ifndef V3_EARLY
#define V3_EARLY 0
#endif
#ifndef K36_HALF_FENCE
#define K36_HALF_FENCE 0   // v3.6 hc_item: compiler barrier between the two kb halves (no effect on ptxas; kept off)
#endif
#ifndef K36_KB_ROLL
#define K36_KB_ROLL 0      // v3.6 hc_item: 1 = kb loop not unrolled (register pressure fallback)
#endif
#ifndef K36_LAUNDER
#define K36_LAUNDER 1      // v3.6 flush_item: opaque zero offset on the output bases (register pressure knob)
#endif
constexpr long long WAIT_BUDGET = 1ll << 31;   // clock64 cycles (~1 s) per mbarrier wait -> error flag
constexpr int HO_BUDGET = 10;   // v3.6: whole K1 hot items handed to the epilogue warp while npart < 10 (ids <= 15)

struct Params {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* meta; int* cnt; int* err;
  __nv_bfloat16* out; unsigned* sink;
  float* ss; int* cnt2;   // v3.1: per-route sum of squares of bf16(y)*(1+w) (K2), W2 split-item counters
  uint8_t* xw; uint8_t* xw2;   // v3.2: hot passes' W13 / W2 B operands in wgmma's core-matrix layout
  int hot_t;                   // v3.3: expert is warm (wgmma, 64-token passes) if cnt > hot_t (> 0); 0 = off
  int hot_w4;                  // v3.3: warm-work scale in the CTA role split's byte model, x8; v3.6 role split: per-chunk
                               // cost of a hot pass in sixteenths of a stage
  int hot_mode;                // v3.6 (HK2): bit 0 dual range, bit 1 second-wave CTAs cold part first; 0 = role split
  int M, lo, probe;
};

struct Route {
  int cnt[32], off[32], np[32], hp[32];
  int sbh[33], sbc[33];   // v3.3: stage prefixes over warm experts only / cold experts only
  int sb[33];
  int wc[8][32];
  int R;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 layout
  o[0] = w & 0x81C081C0u;
  o[1] = __funnelshift_l(w, w, 6) & 0x81C081C0u;
  o[2] = __funnelshift_l(w, w, 12) & 0x81C081C0u;
  o[3] = ((w >> 6) & 0x01000100u) | ((w << 2) & 0x80C080C0u);
}
__device__ __forceinline__ uint32_t bmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
  return d;
}
__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }

// ---- mbarrier / bulk copy / PDL (PTX ISA 8.x, sm_90) ----
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
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return; }
  }
}
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
__device__ __forceinline__ void cp_async16_ef(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "l"(pol) : "memory");
}
__device__ __forceinline__ void cp_async_mbar_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }

// ---- routing: every CTA, deterministic (v2.2's algorithm; warps 0-7) ----
// TPH: tokens per hot pass (v3.3 wgmma warm passes 64); HGE (v3.6): expert is hot iff cnt > hot_t (hot_t >= 0).
// LIST (v3.7): false skips the scatter pass (s_list / meta rows); K2 never reads s_list (its X rows are route positions).
template <int NC, int TPH = 64, bool HGE = false, bool LIST = true>
__device__ void route(const Params& P, Route& S, int8_t* s_ide, int16_t* s_list, int nrb, bool write_meta) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int NP = 8 * P.M;
  for (int i = tid; i < 8 * 32; i += blockDim.x) (&S.wc[0][0])[i] = 0;
  __syncthreads();
  const int L = ((NP + 255) / 256) * 32;
  const int p0 = warp * L;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) {
        const int v = P.ids[p] - P.lo;
        e = (v >= 0 && v < 32) ? v : -1;
      }
      if (e >= 0) atomicAdd(&S.wc[warp][e], 1);
    }
  }
  __syncthreads();
  if (warp == 0) {
    int c = 0;
    for (int w = 0; w < 8; ++w) { const int v = S.wc[w][lane]; S.wc[w][lane] = c; c += v; }
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    // v3.3: warm expert -> 64-token wgmma passes; v3.6 (HGE): hot expert -> passes of TPH = 8 * NCH tokens
    const bool hot = HGE ? (c > P.hot_t) : (P.hot_t > 0 && c > P.hot_t);
    const int np = hot ? (c + TPH - 1) / TPH : (c + 8 * NC - 1) / (8 * NC);
    int hv = hot ? np : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, hv, d); if (lane >= d) hv += nb; }
    S.hp[lane] = hv - (hot ? np : 0);
    {
      int cv = hot ? 0 : np;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, cv, d); if (lane >= d) cv += nb; }
      S.sbh[lane] = 16 * nrb * (hv - (hot ? np : 0));
      S.sbc[lane] = 16 * nrb * (cv - (hot ? 0 : np));
      if (lane == 31) { S.sbh[32] = 16 * nrb * hv; S.sbc[32] = 16 * nrb * cv; }
    }
    int sv = np;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, sv, d); if (lane >= d) sv += nb; }
    S.cnt[lane] = c; S.off[lane] = off; S.np[lane] = np;
    S.sb[lane] = 16 * nrb * (sv - np);
    if (lane == 31) { S.sb[32] = 16 * nrb * sv; S.R = v; }
    for (int w = 0; w < 8; ++w) S.wc[w][lane] += off;
  }
  __syncthreads();
  if (!LIST) return;
  if (warp < 8) {
    for (int base = 0; base < L; base += 32) {
      const int p = p0 + base + lane;
      int e = -1;
      if (p < NP) { const int v = __ldg(P.ids + p) - P.lo; e = (v >= 0 && v < 32) ? v : -1; }
      const unsigned m = __match_any_sync(FULL, e);
      const int rank = __popc(m & ((1u << lane) - 1u));
      const int b = (e >= 0) ? S.wc[warp][e] : 0;
      __syncwarp();
      if (e >= 0) {
        const int c = b + rank;
        s_list[c] = (int16_t)p;
        if (write_meta) P.meta[MH + p] = c;
        if (rank == 0) S.wc[warp][e] = b + __popc(m);
      } else if (write_meta && p < NP) {
        P.meta[MH + p] = -1;
      }
      __syncwarp();
    }
  }
  __syncthreads();
}

// Stream-K range of this CTA over the stage sequence [0, S).
__device__ __forceinline__ void my_range2(long long S, int cta, int ncta, int& beg, int& end) {
  beg = (int)(S * (long long)cta / ncta);
  end = (int)(S * (long long)(cta + 1) / ncta);
}
__device__ __forceinline__ void my_range(long long S, int& beg, int& end) {
  const long long G = gridDim.x;
  beg = (int)(S * (long long)blockIdx.x / G);
  end = (int)(S * (long long)(blockIdx.x + 1) / G);
}

// Start of the segment processed first: the item holding the range's last stage (shared with the next CTA) goes
// first, so both contributions to every split item land early and its finalize overlaps the stream, not the tail.
__device__ __forceinline__ int tail_first(const Route& S, int beg, int end) {
  if (end <= beg) return beg;
  int e = 0;
  while (S.sb[e + 1] <= end - 1) ++e;
  const int m = S.sb[e] + ((end - 1 - S.sb[e]) & ~15);
  return max(beg, m);
}

// One kb (16 B of codes) of this warp's tile, B from the staged X slice.
template <int NC>
__device__ __forceinline__ void consume_kb(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                           int kloc, int nch, int t, float (*acc)[4]) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      const uint4* xp = reinterpret_cast<const uint4*>(xrow + q * 8 * XST + kloc + 16 * t);
      const uint4 x0 = xp[0], x1 = xp[1];
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

// Lab probes: B operand from registers (no X smem reads) / decode only.
template <int NC>
__device__ __forceinline__ void consume_kb_breg(const uint4 w, const uint32_t sw2, int par, uint4 x0, uint4 x1,
                                                int nch, float (*acc)[4], uint32_t& sink, bool mma) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  if (!mma) {
#pragma unroll
    for (int j = 0; j < 8; ++j) sink ^= ag[j] ^ a8[j];
    return;
  }
#pragma unroll
  for (int q = 0; q < NC; ++q) {
    if (q < nch) {
      mma16816(acc[q], ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
      mma16816(acc[q], ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
      mma16816(acc[q], ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
      mma16816(acc[q], ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
    }
  }
}

// MODE 1: W13 (x -> inter / h), MODE 2: W2 (interS -> y).
// v3.2 hot passes: bf16 element index of (pass pid, token n of the pass, k = col) in the wgmma B layout
// (v2.3's no-swizzle K-major core matrices; per stage of 256 k: [n8][kc 32][row 8][16 B], u32 q of the 32 B piece
// k = 64 i + 16 t + 2q is core matrix kc = 8 i + q, u32 t of row n & 7).
__device__ __forceinline__ size_t xw_u32(int pid, int NW, int s2, int n, int i, int t, int q) {
  return (size_t)(pid * 16 + s2) * (NW * 128) + (n >> 3) * 1024 + 256 * i + 32 * q + 4 * (n & 7) + t;
}
__device__ __forceinline__ size_t xw_elem(int pid, int NW, int n, int col) {
  const int kk = col & 255;
  return 2 * xw_u32(pid, NW, col >> 8, n, kk >> 6, (kk >> 4) & 3, (kk >> 1) & 7) + (kk & 1);
}
__device__ __forceinline__ uint64_t wg_desc_sa(uint32_t a, uint32_t sbo) {
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(128u >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int NPEND>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(NPEND) : "memory"); }
#define K3_WG_HEAD "{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned."
template <int N> __device__ __forceinline__ void wgmma_rs(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                          uint64_t db);
template <> __device__ __forceinline__ void wgmma_rs<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                         uint64_t db) {
  asm volatile(K3_WG_HEAD "m64n64k16.f32.bf16.bf16 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
               "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 0;\n}\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
                 "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
                 "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
                 "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
@WGSPECS@
#undef K3_WG_HEAD
// v4.0 LEAN: wgmma m64nNk16 RS with the descriptor formed in the asm from two 32-bit halves
template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);
template <> __device__ __forceinline__ void wgmma_rs2<16>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%12, %13};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<32>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%20, %21};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<48>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%28, %29};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, {%24,%25,%26,%27}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
template <> __device__ __forceinline__ void wgmma_rs2<64>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 dd;\nmov.b64 dd, {%36, %37};\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, dd, p, 1, 1, 0;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(lo), "r"(hi));
}
__device__ __forceinline__ void wgmma_nw(int nw, float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
  if (nw <= 16) wgmma_rs<16>(d, a0, a1, a2, a3, db);
  else if (nw <= 32) wgmma_rs<32>(d, a0, a1, a2, a3, db);
  else if (nw <= 48) wgmma_rs<48>(d, a0, a1, a2, a3, db);
  else wgmma_rs<64>(d, a0, a1, a2, a3, db);
}

// v4.2 WSK: Stream-K cut at equal byte-model weight (pick_role's model: a stage of a pass with n token rows weighs 36 + n,
// hot n rounded up to 8) instead of equal stage counts; HOTCAP calls only (P.hot_t > 0). Only the cut points move.
template <bool HOT, int NC>
__device__ __forceinline__ long long wsk_wpass(int c, int jj) {
  constexpr int TP = HOT ? 64 : 8 * NC;
  const int nt = min(TP, c - jj * TP);
  return HOT ? 72 + ((nt + 7) & ~7) : 36 + nt;   // v4.3: hot rows weigh half as much (lab sweep)
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ int wsk_pos(const Route& S, long long T) {
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    const int np = S.np[e], c = S.cnt[e];
    long long wrb = 0;
    for (int jj = 0; jj < np; ++jj) wrb += 16 * wsk_wpass<HOT, NC>(c, jj);
    if (T >= wrb * NRB) { T -= wrb * NRB; continue; }
    const int rb = (int)(T / wrb);
    T -= rb * wrb;
    for (int jj = 0; jj < np; ++jj) {
      const long long w = wsk_wpass<HOT, NC>(c, jj);
      if (T >= 16 * w) { T -= 16 * w; continue; }
      return S.sb[e] + (rb * np + jj) * 16 + (int)(T / w);
    }
    return S.sb[e] + (rb + 1) * np * 16;
  }
  return S.sb[32];
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ void wsk_range(const Route& S, int cta, int ncta, int& beg, int& end) {
  long long W = 0;
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    for (int jj = 0; jj < S.np[e]; ++jj) W += 16LL * NRB * wsk_wpass<HOT, NC>(S.cnt[e], jj);
  }
  beg = wsk_pos<HOT, NC, NRB>(S, W * cta / ncta);
  end = (cta + 1 == ncta) ? S.sb[32] : wsk_pos<HOT, NC, NRB>(S, W * (cta + 1) / ncta);
}

// v3.3 warm-expert item (all of its stages s0..s1): wgmma m64nNk16, A = this warp's dequantized tile (RS form, the
// mma.m16n8k16 wiring), B = the stage's X slice in core-matrix layout. Half-kb groups (2 wgmmas) alternate between
// two 8-register A buffers (wait_group 1 before reuse); a stage's slot is released after the next stage's second
// wait, the last one after wait_group 0. Columns >= the pass's tokens read stale rows and are discarded.
template <int N, int NST, int STG, int MODE>
__device__ __forceinline__ void hot_item(const Params& P, const Route& S, const uint8_t* ring, uint32_t full_a,
                                      uint32_t empty_a, int& it_ref, int s0, int s1, int e, int j, int rb, int c0,
                                      int ntok, int npart) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float hacc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) hacc[i] = 0.f;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
    const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);   // v4.0 LEAN: the low word's add never carries
    // per kb: decode into one of two 16-register A buffers, 4 wgmmas, commit; wait_group 1 before a buffer is
    // rewritten, 0 before the slot is released (this form is the one ptxas does not serialize)
    uint32_t hb[2][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if (i >= 2) wg_wait<1>();
      uint32_t* h = hb[i & 1];
      deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
      wg_fence();
      wgmma_rs2<N>(hacc, h[0], h[8], h[1], h[9], dlo + 64 * i, dhi);
      wgmma_rs2<N>(hacc, h[2], h[10], h[3], h[11], dlo + 64 * i + 16, dhi);
      wgmma_rs2<N>(hacc, h[4], h[12], h[5], h[13], dlo + 64 * i + 32, dhi);
      wgmma_rs2<N>(hacc, h[6], h[14], h[7], h[15], dlo + 64 * i + 48, dhi);
      wg_commit();
    }
    wg_wait<0>();
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  // ---- flush (v4.0 FLUSH2: one base pointer per lane, compile-time offsets per (q, hh); same values, stores, atomics) ----
  const int nch = (ntok + 7) >> 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  const bool whole = (s0 == 0 && s1 == 16);
  if constexpr (MODE == 1) {
    const int col = 8 * tile + g;
    if (whole) {   // xw_elem(pid, 64, n, col) = xw_elem(pid, 64, 2t, col) + 2048 (n >> 3) + 8 hh for n = 8q + 2t + hh
      __nv_bfloat16* xb = reinterpret_cast<__nv_bfloat16*>(P.xw2) + xw_elem(S.hp[e] + j, 64, 2 * t, col);
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok)
            xb[2048 * q + 8 * hh] = __float2bfloat16_rn(swiglu(hacc[4 * q + hh] * gs, hacc[4 * q + 2 + hh] * gs));
    } else {
      float* hb = P.h + (size_t)(c0 + 2 * t) * (2 * KDIM) + col;
#pragma unroll
      for (int q = 0; q < N / 8; ++q)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh)
          if (q * 8 + 2 * t + hh < ntok) {
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM), hacc[4 * q + hh] * gs);
            atomicAdd(hb + (8 * q + hh) * (2 * KDIM) + KDIM, hacc[4 * q + 2 + hh] * gs);
          }
    }
  } else {
    float* yb = P.y + (size_t)(c0 + 2 * t) * KDIM + 16 * tile + g;
#pragma unroll
    for (int q = 0; q < N / 8; ++q)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
        if (q * 8 + 2 * t + hh < ntok) {
          const float v0 = hacc[4 * q + hh] * gs, v1 = hacc[4 * q + 2 + hh] * gs;
          float* yp = yb + (8 * q + hh) * KDIM;
          if (whole) { yp[0] = v0; yp[8] = v1; }
          else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
        }
  }
  if (MODE == 2 && whole) {
    const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
    for (int q = 0; q < N / 8; ++q) {
      if (q < nch) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float z0 = bf16r(hacc[4 * q + hh] * gs) * w0, z1 = bf16r(hacc[4 * q + 2 + hh] * gs) * w1;
          float pz = z0 * z0 + z1 * z1;
          pz += __shfl_xor_sync(FULL, pz, 4);
          pz += __shfl_xor_sync(FULL, pz, 8);
          pz += __shfl_xor_sync(FULL, pz, 16);
          const int n = q * 8 + 2 * t + hh;
          if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
        }
      }
    }
  }
  if (!whole) asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
}

// HM 0: cold items (mma.sync), HM 1 (v3.3): warm-expert items (64-token wgmma passes). S.sb holds the role's stage
// prefix; this CTA is number cta of ncta in its role.
template <int NC, int NST, int MODE, int HM, int STG>
__device__ void gemm(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,
                     int cta, int ncta) {
  constexpr bool HOTCAP = (HM == 1);
  constexpr int QMAX = HOTCAP ? 8 : NC;   // accumulator chunks (8 tokens each)
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const long long Stot = S.sb[32];
  int beg, end;
  if constexpr (NC == 2) {   // v4.2 WSK: NC 2 builds only (HOTCAP roles; vid 0's NC 1 code stays byte-identical)
    if (P.hot_t > 0) wsk_range<HOTCAP, NC, NRB>(S, cta, ncta, beg, end);
    else my_range2(Stot, cta, ncta, beg, end);
  } else {
    my_range2(Stot, cta, ncta, beg, end);
  }
  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // ---------------- producer ----------------
    // X copies that depend on the PDL predecessor (K2: interS / xw2 from K1; K1 hot items: xw from P0) are deferred
    // until griddepcontrol.wait, which is taken lazily: only when a deferred slot would otherwise be reused, or at the
    // end. Weight copies (and K1's cold X copies from the input) never wait.
    int it = 0, e = 0;
    bool waited = !pdl;
    int nd = 0;
    const int pmode = P.probe & 15;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int mid = tail_first(S, beg, end);
    for (int seg = 0; seg < 2; ++seg) {
    const int xb = seg ? beg : mid, xe = seg ? mid : end;
    e = 0;
    for (int x = xb; x < xe;) {
      while (S.sb[e + 1] <= x) ++e;
      const int np = S.np[e];
      constexpr bool hot = HOTCAP;
      const int TP = hot ? 64 : 8 * NC;
      const int l = x - S.sb[e];
      const int item = l >> 4;
      const int s0 = l & 15;
      const int s1 = min(16, s0 + (xe - x));
      const int rb = item / np, j = item - rb * np;
      const int ntok = min(TP, S.cnt[e] - j * TP);
      const int c0 = S.off[e] + j * TP;
      const uint8_t* wb = wsrc + ((size_t)(e * NRB + rb) * 16) * SW;
      const int ntx = (pmode == 5) ? 0 : ntok;
      int tok = 0;
      if (MODE == 1 && NC <= 2 && !hot && lane >= 1 && lane <= ntx) tok = ((int)s_list[c0 + lane - 1]) >> 3;
      const uint32_t xbytes = hot ? ((ntok + 7) >> 3) * 4096 : (MODE == 1) ? ntx * 512 : ntok * XROW;
      for (int s = s0; s < s1; ++s, ++it) {
        const int slot = it % NST;
        const uint32_t par = ((it / NST) & 1) ^ 1;
        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now
          pdl_wait();
          asm volatile("fence.proxy.async;\n" ::: "memory");
          waited = true;
          if (lane == 0) {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
          }
          nd = 0;
        }
        mbar_wait(empty_a + 8 * slot, par, P.err);
        const uint32_t dst = ring_a + slot * STG;
        const uint32_t fb = full_a + 8 * slot;
        if (lane == 0) {
          mbar_expect_tx(fb, SW + xbytes);
          const uint8_t* src = wb + (size_t)s * SW;
          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
          else bulk_g2s(dst, src, SW, fb);
        }
        mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
        __syncwarp();
        if (MODE == 1 && !hot) {
          if constexpr (NC <= 2) {
            if (lane >= 1 && lane <= ntx)
              bulk_g2s(dst + SW + (lane - 1) * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
          } else {   // up to 8 NC rows: lane i copies rows i, i + 32
            for (int i = lane; i < ntx; i += 32)
              bulk_g2s(dst + SW + i * XROW, P.x + (size_t)(((int)s_list[c0 + i]) >> 3) * KDIM + s * 256, 512, fb);
          }
        } else {
          const int pid = S.hp[e] + j;
          const __nv_bfloat16* src = hot
              ? reinterpret_cast<const __nv_bfloat16*>(((MODE == 1) ? P.xw : P.xw2) + (size_t)(pid * 16 + s) * (64 * 512))
              : P.inter + ((size_t)s * (8 * M3) + c0) * XST;
          if (waited) {
            if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
          } else {
#pragma unroll
            for (int d = 0; d < NST; ++d)
              if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
            ++nd;
          }
        }
      }
      x += s1 - s0;
    }
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
    }
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item ----------------
    // K1: SwiGLU(h) -> interS, h -> 0. K2: per-route sum of squares of z = bf16(y)*(1+w) over the item's 128 columns.
    {
      if ((P.probe & 15) != 0) return;   // probes: consumers never hand over
      int npart = 0, e = 0;
      const int mid = tail_first(S, beg, end);
      for (int seg = 0; seg < 2; ++seg) {
        const int xb = seg ? beg : mid, xe = seg ? mid : end;
        e = 0;
        for (int x = xb; x < xe;) {
          while (S.sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - S.sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          x += s1 - s0;
          if (s0 == 0 && s1 == 16) continue;
          constexpr bool hot = HOTCAP;
          const int TP = hot ? 64 : 8 * NC;
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          asm volatile("bar.sync %0, 288;\n" :: "r"(2 + npart) : "memory");
          ++npart;
          __threadfence();   // cumulative over the consumers' h atomics (ordered before us by the barrier)
          int last = 0;
          if (lane == 0) {
            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
            const int old = atomicAdd(cp, s1 - s0);
            last = (old + (s1 - s0) == 16);
            if (last) atomicExch(cp, 0);
          }
          last = __shfl_sync(FULL, last, 0);
          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            // ntok * 64 columns; lane handles column pair (2*(lane&31)) of 64 for tokens n = 0..ntok-1, batched
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  if (hot) {
                    *reinterpret_cast<uint32_t*>(P.xw2 + 2 * xw_elem(S.hp[e] + j, 64, n0 + u, col)) = pk;
                  } else {
                    *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  }
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------
  if (MODE == 1 && ((P.probe & 15) == 0 || (P.probe & 15) >= 6)) {
    // y rows [0, R) and ss[0, R) -> 0 while the first stages are in flight (K2 accumulates into both after its wait)
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = 0, e = 0;
  uint32_t sink = 0u;
  int npart = 0;
  const int mid = tail_first(S, beg, end);
  for (int seg = 0; seg < 2; ++seg) {
  const int xb = seg ? beg : mid, xe = seg ? mid : end;
  e = 0;
  for (int x = xb; x < xe;) {
    while (S.sb[e + 1] <= x) ++e;
    const int np = S.np[e];
    const int l = x - S.sb[e];
    const int item = l >> 4;
    const int s0 = l & 15;
    const int s1 = min(16, s0 + (xe - x));
    constexpr bool hot = HOTCAP;
    const int TP = hot ? 64 : 8 * NC;
    const int rb = item / np, j = item - rb * np;
    const int ntok = min(TP, S.cnt[e] - j * TP);
    const int nch = (ntok + 7) >> 3;
    const int c0 = S.off[e] + j * TP;
    const int nw = ((ntok + 15) >> 4) << 4;   // hot: wgmma N in {16, 32, 48, 64}
    float acc[QMAX][4], acc2[NC][4];   // acc2: second mma chain (NC == 1 only)
    if constexpr (HOTCAP) {
      // warm-expert item on wgmma (this CTA's role has only warm items)
      if ((P.probe & 15) == 0) {
        if (nw <= 16) hot_item<16, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 32) hot_item<32, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else if (nw <= 48) hot_item<48, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
        else hot_item<64, NST, STG, MODE>(P, S, ring, full_a, empty_a, it, s0, s1, e, j, rb, c0, ntok, npart);
      }
      if (!(s0 == 0 && s1 == 16)) ++npart;
      x += s1 - s0;
      continue;
    } else {
#pragma unroll
    for (int q = 0; q < QMAX; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc2[q][i] = 0.f;
    for (int s = s0; s < s1; ++s, ++it) {
      const int slot = it % NST;
      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
      const uint8_t* st = ring + slot * STG;
      const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
      const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
      if ((P.probe & 15) == 1 || (P.probe & 15) == 5) {
        const uint4 a = wp[0], b = wp[96];
        sink ^= a.x ^ b.w ^ sp[0];
      } else if ((P.probe & 15) == 9 || (P.probe & 15) == 10) {
        uint4 wv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);   // operands in registers: release the slot first
        const uint4 xb0 = make_uint4(s0v, s1v, s0v ^ 1u, s1v ^ 1u), xb1 = make_uint4(s1v, s0v, s1v ^ 2u, s0v ^ 2u);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, xb0, xb1, nch, (i < 2 || NC > 1) ? acc : acc2, sink,
                              (P.probe & 15) == 9);
        continue;
      } else if (NC == 1 && V3_EARLY) {
        // v3.1: all of this warp's operands of the stage (codes, scales, X fragments) into registers, release the
        // slot, then decode + mma, so the ring cycle no longer includes the consumer's math.
        uint4 wv[4], x0[4], x1[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          wv[i] = wp[i * 32];
          const uint4* xp = reinterpret_cast<const uint4*>(xrow + i * 64 + 16 * t);
          x0[i] = xp[0]; x1[i] = xp[1];
        }
        const uint32_t s0v = sp[0], s1v = sp[32];
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);
#pragma unroll
        for (int i = 0; i < 4; ++i)
          consume_kb_breg<NC>(wv[i], (i < 2) ? s0v : s1v, i & 1, x0[i], x1[i], nch, (i < 2 || NC > 1) ? acc : acc2, sink, true);
        continue;
      } else {
        uint4 wv[4];
        uint32_t sv[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
        sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
        for (int i = 0; i < 4; ++i) consume_kb<NC>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, nch, t, (i < 2 || NC > 1) ? acc : acc2);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    }
    // ---- flush ----
#pragma unroll
    for (int q = 0; q < NC; ++q)
#pragma unroll
      for (int i = 0; i < 4; ++i) if (NC == 1) acc[q][i] += acc2[q][i];
    const int fmode = P.probe & 15;   // 6: partial items stored like whole ones (timing only); 7: no finalize
    if (fmode != 0 && fmode < 6) {
#pragma unroll
      for (int q = 0; q < QMAX; ++q) sink ^= __float_as_uint(acc[q][0] + acc[q][1] + acc[q][2] + acc[q][3]);
    } else {
      const int tile = rb * 8 + warp;
      const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
      const bool whole = (s0 == 0 && s1 == 16) || fmode == 6;
      if constexpr (MODE == 1) {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                const int c = c0 + n;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                const int col = 8 * tile + g;
                if (whole && hot) {
                  reinterpret_cast<__nv_bfloat16*>(P.xw2)[xw_elem(S.hp[e] + j, 64, n, col)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else if (whole) {
                  P.inter[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
                } else {
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + col, v0);
                  atomicAdd(P.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
                }
              }
            }
          }
        }
        if (!whole && fmode != 7) {
          // hand the split item to the epilogue warp (warp 9): non-blocking for the consumers
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
        }
      } else {
#pragma unroll
        for (int q = 0; q < QMAX; ++q) {
          if (q < nch) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
              const int n = q * 8 + 2 * t + hh;
              if (n < ntok) {
                float* yp = P.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
                const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
                if (whole) { yp[0] = v0; yp[8] = v1; }
                else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
              }
            }
          }
        }
        if (whole) {
          // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)
          const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
          for (int q = 0; q < QMAX; ++q) {
            if (q < nch) {
#pragma unroll
              for (int hh = 0; hh < 2; ++hh) {
                const float z0 = bf16r(acc[q][hh] * gs) * w0, z1 = bf16r(acc[q][2 + hh] * gs) * w1;
                float pz = z0 * z0 + z1 * z1;
                pz += __shfl_xor_sync(FULL, pz, 4);
                pz += __shfl_xor_sync(FULL, pz, 8);
                pz += __shfl_xor_sync(FULL, pz, 16);
                const int n = q * 8 + 2 * t + hh;
                if (g == 0 && n < ntok) atomicAdd(P.ss + c0 + n, pz);
              }
            }
          }
        } else if (fmode != 7) {
          asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
        }
      }
    }
    if (!(s0 == 0 && s1 == 16)) ++npart;
    x += s1 - s0;
  }
  }
  if (sink == 0x9E3779B9u) atomicAdd(P.sink, 1u);   // keeps probe work alive; never true in practice
}

template <int NC, int NST>
__device__ __forceinline__ void setup_bars(uint64_t* bars) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + NST + i), 8); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
}

// v3.3 role split (HOTCAP): the first Gh CTAs take the warm experts' stages (wgmma), the rest the cold stages; Gh is
// proportional to the work (a warm stage weighs ~2 cold ones). Each CTA then runs one code path only (an item-level
// branch between the two made ptxas serialize every wgmma: C7520 / C7513).
template <int NC>
__device__ __forceinline__ bool pick_role(Route& S, int P_hot_w4, int hot_t, int& cta, int& ncta) {
  // work model: a stage costs its bulk bytes, 36 half-KB of codes + scales plus 1 half-KB per staged token row
  // (warm rows rounded to 8), warm work scaled by hot_w4 / 8
  const long long Sh = S.sbh[32], Sc = S.sbc[32];
  const int G = (int)gridDim.x;
  int Gh = 0;
  if (Sh > 0) {
    long long Wh = 0, Wc = 0;
    for (int e = 0; e < 32; ++e) {
      const int c = S.cnt[e];
      if (c == 0) continue;
      if (c > hot_t) {
        const int full = c >> 6, rem = c & 63;
        Wh += (long long)full * (36 + 64) + (rem ? 36 + ((rem + 7) & ~7) : 0);
      } else {
        const int tp = 8 * NC, full = c / tp, rem = c - full * tp;
        Wc += (long long)full * (36 + tp) + (rem ? 36 + rem : 0);
      }
    }
    if (P_hot_w4 >= 50) { Wh *= P_hot_w4 - 50; Wc *= 32; } else { Wh *= P_hot_w4; Wc *= 8; }   // v4.0 fine weight
    Gh = (int)((Wh * G + (Wh + Wc) / 2) / (Wh + Wc));
    Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
  }
  const bool hr = (int)blockIdx.x < Gh;
  cta = hr ? (int)blockIdx.x : (int)blockIdx.x - Gh;
  ncta = hr ? Gh : G - Gh;
  __syncthreads();
  if (threadIdx.x < 33) S.sb[threadIdx.x] = hr ? S.sbh[threadIdx.x] : S.sbc[threadIdx.x];
  __syncthreads();
  return hr;
}

// ======================================================================================================================
// v3.6 (HK2 variants): per-expert pass width. Cold experts (<= hot_t tokens) = v3.3's NC1 consumer + flush verbatim;
// hot experts (> hot_t tokens) = one pass of up to 8*NCH tokens per row block (their weights stream from HBM once),
// consumed by hc_item<NQ> with a compile-time chunk count NQ = ceil(ntok / 8) in [2, NCH] (a hot pass with <= 8 tokens
// takes the cold code). One ring of NST stages of SW + 8*NCH*XROW bytes for both kinds. The stage sequences of the two
// kinds are S.sbh (hot, TPH = 8*NCH tokens per pass) and S.sbc (cold, 8 per pass); every CTA processes up to two
// "parts" (a stream-K range of one kind each), see ranges36.
// ======================================================================================================================
__device__ __forceinline__ int tail_first2(const int* sb, int beg, int end) {
  if (end <= beg) return beg;
  int e = 0;
  while (sb[e + 1] <= end - 1) ++e;
  const int m = sb[e] + ((end - 1 - sb[e]) & ~15);
  return max(beg, m);
}

// Flush of one item's accumulators (NQ 8-token chunks): v3.3's flush code with QMAX = NQ (identical formulas).
// NOSWI (K1 hot items with >= 3 chunks): even a whole item goes to h by FP32 reds (onto zero: exact) and the epilogue
// warp applies the same SwiGLU -> bf16 -> interS, so no IEEE-division slow-path call sits in the 16-accumulator flush
// (that call kept K1 at NCH 4 above 96 registers). arrive: hand the item to the epilogue warp (barrier 2 + npart).
template <int NQ, int MODE, bool NOSWI = false>
__device__ __forceinline__ void flush_item(const Params& P, float (*acc)[4], bool whole, int e, int rb, int c0, int ntok,
                                           int npart, bool arrive) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tile = rb * 8 + warp;
  const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
  // output bases offset by an opaque zero (K36_LAUNDER): the per-chunk 64-bit address bases are then formed inside
  // the flush instead of being hoisted into the kernel prologue (~6 registers across every item, spilled at NCH 4).
  // An integer, not the pointer, goes through the asm, so the bases stay derived from the kernel's global pointers
  // (global atomics / stores; laundering the pointer itself made them generic).
  int z = 0;
#if K36_LAUNDER
  asm volatile("" : "+r"(z));
#endif
  float* hb = P.h + z;
  __nv_bfloat16* ib = P.inter + z;
  float* yb = P.y + z;
  float* ssb = P.ss + z;
  if constexpr (MODE == 1) {
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          const int c = c0 + n;
          const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
          const int col = 8 * tile + g;
          if (whole && !NOSWI) {
            ib[((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
          } else {
            atomicAdd(hb + (size_t)c * (2 * KDIM) + col, v0);
            atomicAdd(hb + (size_t)c * (2 * KDIM) + KDIM + col, v1);
          }
        }
      }
    }
    if (arrive) asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");   // split item -> epilogue warp
  } else {
#pragma unroll
    for (int q = 0; q < NQ; ++q) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int n = q * 8 + 2 * t + hh;
        if (n < ntok) {
          float* yp = yb + (size_t)(c0 + n) * KDIM + 16 * tile + g;
          const float v0 = acc[q][hh] * gs, v1 = acc[q][2 + hh] * gs;
          if (whole) { yp[0] = v0; yp[8] = v1; }
          else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
        }
      }
    }
    if (whole) {
      // sum of squares of z = bf16(y) * (1 + w) over this item's 128 columns, per route (one red per warp)
      const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
      for (int q = 0; q < NQ; ++q) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const float z0 = bf16r(acc[q][hh] * gs) * w0, z1 = bf16r(acc[q][2 + hh] * gs) * w1;
          float pz = z0 * z0 + z1 * z1;
          pz += __shfl_xor_sync(FULL, pz, 4);
          pz += __shfl_xor_sync(FULL, pz, 8);
          pz += __shfl_xor_sync(FULL, pz, 16);
          const int n = q * 8 + 2 * t + hh;
          if (g == 0 && n < ntok) atomicAdd(ssb + c0 + n, pz);
        }
      }
    } else if (arrive) {
      asm volatile("bar.arrive %0, 288;\n" :: "r"(2 + npart) : "memory");
    }
  }
}

// One cold item (<= 8 tokens): v3.3's NC1 consumer (two independent mma chains: kb 0-1 and kb 2-3), then its flush.
template <int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void cold_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                          int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float acc[1][4], acc2[1][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) { acc[0][i] = 0.f; acc2[0][i] = 0.f; }
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
    uint4 wv[4];
    uint32_t sv[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
    sv[0] = sp[0]; sv[1] = sp[32];
#pragma unroll
    for (int i = 0; i < 4; ++i) consume_kb<1>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, 1, t, (i < 2) ? acc : acc2);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[0][i] += acc2[0][i];
  const bool whole = s0 == 0 && s1 == 16;
  flush_item<1, MODE>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : !whole);
}

// One hot item with exactly NQ (2..4) 8-token chunks: the warp's 16 x 64 weight tile of a kb is decoded once, in two
// halves (k16 steps 0-1: words x / z, then steps 2-3: words y / w; 8 A registers live), and each half feeds NQ chunks of
// 1 LDS.128 (X) + 2 mma.m16n8k16. Per chunk the mma order is consume_kb's (k16 steps 0, 1, 2, 3), so the result is
// bit-identical to consume_kb<NQ>; only the interleaving across (independent) chunks differs. NQ is a compile-time
// constant (v3.3's NC2/NC4 builds bounded the chunk loop at run time).
template <int NQ, int NST, int STG, int MODE, bool MB = false>
__device__ __forceinline__ void hc_item(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                        int& it_ref, int s0, int s1, int e, int rb, int c0, int ntok, int npart, bool ho) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  float acc[NQ][4];
#pragma unroll
  for (int q = 0; q < NQ; ++q)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[q][i] = 0.f;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;
#if K36_KB_ROLL
#pragma unroll 1
#else
#pragma unroll
#endif
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
#if K36_HALF_FENCE
        if (hf) asm volatile("" ::: "memory");   // keep the second half's X loads after the first half's mma
#endif
        uint32_t a[4], b[4];
        deq4(hf ? w.y : w.x, a);   // gate rows g: consume_kb's ag[4 hf .. 4 hf + 3]
        deq4(hf ? w.w : w.z, b);   // rows g + 8:  consume_kb's a8[4 hf .. 4 hf + 3]
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);   // x0 / x1
          mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);
          mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  constexpr bool NOSWI = (MODE == 1 && NQ >= 3);
  const bool whole = s0 == 0 && s1 == 16;
  if (!MB && NOSWI && whole && !ho) { atomicExch(P.err, 1); atomicExch(P.err + 3, 1); }   // barrier-id budget exhausted
  flush_item<NQ, MODE, NOSWI>(P, acc, whole, e, rb, c0, ntok, npart, MB ? false : (NOSWI ? (ho || !whole) : !whole));
}

// v3.6 gemm over four segments pr[3 sg .. 3 sg + 2] = (begin, end, kind) in shared memory (segs36: per part, its
// tail-first item, then the rest; kind 1 = hot). Within a part:
// v3.3's tail-first order (the item holding the range's last stage first). Producer, consumers and epilogue walk the
// identical (part, segment, item, stage) sequence, so ring parity and split-item barrier ids stay in step.
template <int NCH, int NST, int MODE>
__device__ void gemm36(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl,
                       const int* pr) {
  constexpr int STG = SW + 8 * NCH * XROW;
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // ---------------- producer ----------------
    // K2's X copies (interS from K1) are deferred until griddepcontrol.wait, taken lazily as in v3.3; weight copies
    // and K1's X copies never wait. K1: lane (r + 1) & 31 copies X row r of the pass (ntok <= 32).
    int it = 0;
    bool waited = !pdl;
    int nd = 0;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int r = (lane + 31) & 31;
    for (int sg = 0; sg < 4; ++sg) {
      const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
      if (xe <= xb) continue;
      const bool hk = pr[3 * sg + 2] != 0;
      const int* sb = hk ? S.sbh : S.sbc;
      const int TP = hk ? 8 * NCH : 8;
      {
        int e = 0;
        for (int x = xb; x < xe;) {
          while (sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          const uint8_t* wb = wsrc + ((size_t)(e * NRB + rb) * 16) * SW;
          int tok = 0;
          if (MODE == 1 && r < ntok) tok = ((int)s_list[c0 + r]) >> 3;
          const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;
          for (int s = s0; s < s1; ++s, ++it) {
            const int slot = it % NST;
            const uint32_t par = ((it / NST) & 1) ^ 1;
            if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now
              pdl_wait();
              asm volatile("fence.proxy.async;\n" ::: "memory");
              waited = true;
              if (lane == 0) {
#pragma unroll
                for (int d = 0; d < NST; ++d)
                  if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
              }
              nd = 0;
            }
            mbar_wait(empty_a + 8 * slot, par, P.err);
            const uint32_t dst = ring_a + slot * STG;
            const uint32_t fb = full_a + 8 * slot;
            if (lane == 0) {
              mbar_expect_tx(fb, SW + xbytes);
              const uint8_t* src = wb + (size_t)s * SW;
              if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
              else bulk_g2s(dst, src, SW, fb);
            }
            mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
            __syncwarp();
            if (MODE == 1) {
              if (r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
            } else {
              const __nv_bfloat16* src = P.inter + ((size_t)s * (8 * M3) + c0) * XST;
              if (waited) {
                if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
              } else {
#pragma unroll
                for (int d = 0; d < NST; ++d)
                  if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
                ++nd;
              }
            }
          }
          x += s1 - s0;
        }
      }
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
    }
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp: split-item counters; the last contributor finishes the item (v3.3) ----------------
    int npart = 0;
    for (int sg = 0; sg < 4; ++sg) {
      const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
      if (xe <= xb) continue;
      const bool hk = pr[3 * sg + 2] != 0;
      const int* sb = hk ? S.sbh : S.sbc;
      const int TP = hk ? 8 * NCH : 8;
      {
        int e = 0;
        for (int x = xb; x < xe;) {
          while (sb[e + 1] <= x) ++e;
          const int np = S.np[e];
          const int l = x - sb[e];
          const int item = l >> 4;
          const int s0 = l & 15;
          const int s1 = min(16, s0 + (xe - x));
          x += s1 - s0;
          const int rb = item / np, j = item - rb * np;
          const int ntok = min(TP, S.cnt[e] - j * TP);
          const int c0 = S.off[e] + j * TP;
          const bool ho = (MODE == 1) && hk && ((ntok + 7) >> 3) >= 3 && npart < HO_BUDGET;   // = the consumers' ho
          if (s0 == 0 && s1 == 16 && !ho) continue;
          asm volatile("bar.sync %0, 288;\n" :: "r"(2 + npart) : "memory");
          ++npart;
          __threadfence();   // cumulative over the consumers' h / y atomics (ordered before us by the barrier)
          int last = 0;
          if (lane == 0) {
            int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
            const int old = atomicAdd(cp, s1 - s0);
            last = (old + (s1 - s0) == 16);
            if (last) atomicExch(cp, 0);
          }
          last = __shfl_sync(FULL, last, 0);
          if (MODE == 2 && last) {
            __threadfence();
            const int col = rb * 128 + 4 * lane;
            const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float4 yv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u)
                if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                float pz = 0.f;
                if (n0 + u < ntok) {
                  float z;
                  z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
                  z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
                }
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
                if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
              }
            }
          }
          if (MODE == 1 && last) {
            __threadfence();
            // ntok * 64 columns; lane handles column pair (2*(lane&31)) of 64 for tokens n = 0..ntok-1, batched
            const int cl = 2 * lane;
            const int col = rb * 64 + cl;
            for (int n0 = 0; n0 < ntok; n0 += 4) {
              float2 gv[4], uv[4];
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
                  gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
                  uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
                }
              }
#pragma unroll
              for (int u = 0; u < 4; ++u) {
                if (n0 + u < ntok) {
                  const int c = c0 + n0 + u;
                  float* hp = P.h + (size_t)c * (2 * KDIM) + col;
                  const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
                  *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
                  *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
                }
              }
            }
          }
        }
      }
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------
  if (MODE == 1) {
    // y rows [0, R) and ss[0, R) -> 0 while the first stages are in flight (K2 accumulates into both after its wait)
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  int it = 0, npart = 0;
  for (int sg = 0; sg < 4; ++sg) {
    const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
    if (xe <= xb) continue;
    const bool hk = pr[3 * sg + 2] != 0;
    const int* sb = hk ? S.sbh : S.sbc;
    const int TP = hk ? 8 * NCH : 8;
    {
      int e = 0;
      for (int x = xb; x < xe;) {
        while (sb[e + 1] <= x) ++e;
        const int np = S.np[e];
        const int l = x - sb[e];
        const int item = l >> 4;
        const int s0 = l & 15;
        const int s1 = min(16, s0 + (xe - x));
        const int rb = item / np, j = item - rb * np;
        const int ntok = min(TP, S.cnt[e] - j * TP);
        const int nch = (ntok + 7) >> 3;
        const int c0 = S.off[e] + j * TP;
        // K1 hot items with >= 3 chunks always go through the epilogue warp (NOSWI); a whole one takes a barrier id
        // only while npart < HO_BUDGET, which leaves ids for the <= 4 split items (max id 2 + 9 + 4 = 15)
        const bool ho = (MODE == 1) && hk && nch >= 3 && npart < HO_BUDGET;
        if (nch <= 1) {
          cold_item<NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart);
        } else if constexpr (NCH >= 4) {
          if (nch >= 4) hc_item<4, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);
          else if (nch == 3) hc_item<3, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, ho);
          else hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);
        } else {
          hc_item<2, NST, STG, MODE>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, npart, false);
        }
        if (!(s0 == 0 && s1 == 16) || ho) ++npart;
        x += s1 - s0;
      }
    }
  }
}

// Ranges of this CTA (all threads compute the same from S). hot_mode bit 0 = 1: dual range, CTA c takes
// [c Sh / G, (c+1) Sh / G) of the hot and [c Sc / G, (c+1) Sc / G) of the cold sequence (bit 1: CTAs c >= G/2, the
// second wave, run the cold part first). bit 0 = 0: role split, CTAs c < Gh take the hot sequence, the rest the cold
// one; Gh from the cost model (cold pass 16, hot pass 16 + hot_w4 * (nch - 1), sixteenths of a stage).
template <int NCH>
__device__ __forceinline__ void ranges36(const Params& P, const Route& S, int& b0, int& e0, int& k0, int& b1, int& e1,
                                         int& k1) {
  const int G = (int)gridDim.x, c = (int)blockIdx.x;
  const long long Sh = S.sbh[32], Sc = S.sbc[32];
  if (P.hot_mode & 1) {
    int hb, he, cb, ce;
    my_range2(Sh, c, G, hb, he);
    my_range2(Sc, c, G, cb, ce);
    const bool cold_first = (P.hot_mode & 2) && (2 * c >= G);
    k0 = cold_first ? 0 : 1; b0 = cold_first ? cb : hb; e0 = cold_first ? ce : he;
    k1 = 1 - k0;             b1 = cold_first ? hb : cb; e1 = cold_first ? he : ce;
    return;
  }
  int Gh = 0;
  if (Sh > 0) {
    long long Wh = 0, Wc = 0;
    for (int x = 0; x < 32; ++x) {
      const int n = S.cnt[x];
      if (n == 0) continue;
      if (n > P.hot_t) {
        const int tp = 8 * NCH, full = n / tp, rem = n - full * tp;
        Wh += (long long)full * (16 + P.hot_w4 * (NCH - 1)) + (rem ? 16 + P.hot_w4 * (((rem + 7) >> 3) - 1) : 0);
      } else {
        Wc += 16ll * ((n + 7) >> 3);
      }
    }
    Gh = (Sc > 0) ? (int)((Wh * G + (Wh + Wc) / 2) / (Wh + Wc)) : G;
    Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
  }
  const bool hr = c < Gh;
  k0 = hr ? 1 : 0;
  if (hr) my_range2(Sh, c, Gh, b0, e0);
  else my_range2(Sc, c - Gh, G - Gh, b0, e0);
  k1 = 0; b1 = 0; e1 = 0;
}

// The CTA's four segments (thread 0 -> shared): per part (ranges36), v3.3's tail-first order: [mid, end) holds the
// item with the range's last stage (shared with the next CTA, so both contributions land early), then [begin, mid).
template <int NCH>
__device__ __forceinline__ void segs36(const Params& P, const Route& S, int* sg) {
  int b[2], e[2], k[2];
  ranges36<NCH>(P, S, b[0], e[0], k[0], b[1], e[1], k[1]);
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int mid = tail_first2(k[part] ? S.sbh : S.sbc, b[part], e[part]);
    sg[6 * part + 0] = mid;     sg[6 * part + 1] = e[part]; sg[6 * part + 2] = k[part];
    sg[6 * part + 3] = b[part]; sg[6 * part + 4] = mid;     sg[6 * part + 5] = k[part];
  }
}

// K1 / K2 of the HK2 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.
template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k1_kernel36(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<1, NST>(bars);
  pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 3)
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  __shared__ int s_pr[12];
  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);
  __syncthreads();
  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);
}

template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k2_kernel36(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<1, NST>(bars);
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 32, false);
  __shared__ int s_pr[12];
  if (threadIdx.x == 0) segs36<NCH>(P, S, s_pr);
  __syncthreads();
  gemm36<NCH, NST, 2>(P, S, s_list, smem, bars, true, s_pr);
}

// ======================================================================================================================
// v3.7 (HK3 variants): v3.6's HK2 consumer / flush code with (a) a dynamic tail queue: per kind sequence (hot sbh,
// cold sbc) the last nd = D * floor(f * S / D) stages are D-stage chunks (D | 16, pool start D-aligned: a chunk never
// straddles an item) taken by ticket after the CTA's static part (v3.6's ranges over [0, S - nd)); hot chunks first,
// any CTA takes any kind (probe bit 23: cold chunks only); (b) every consumer -> epilogue hand-off through an
// NE37-entry smem mailbox (mbarriers: efull 8 = one arrival per consumer warp after __syncwarp, eempty 1), so there is
// no named-barrier id budget; (c) warp-parallel expert search (find_e) and setup.
// Ticket counters meta[DQ37 + 0..3] = K1 ticket, K1 done, K2 ticket, K2 done; the producer that takes the grid's last
// END resets its kernel's pair (every other CTA already holds its END ticket), so each launch starts from zeros.
// ======================================================================================================================
constexpr int NE37 = 8;
constexpr int DQ37 = 120;

__device__ __forceinline__ bool mbar_wait_ok(uint32_t b, uint32_t par, int* err) {
  if (mbar_try(b, par)) return true;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 2, 1); return false; }
  }
  return true;
}

// Full warp: number of experts e with sb[e + 1] <= x, i.e. the expert holding stage x (x < sb[32]); sb non-decreasing.
__device__ __forceinline__ int find_e(const int* sb, int x) {
  return __popc(__ballot_sync(FULL, sb[(threadIdx.x & 31) + 1] <= x));
}
__device__ __forceinline__ int tail_first37(const int* sb, int beg, int end) {   // = tail_first2, full warp
  if (end <= beg) return beg;
  const int e = find_e(sb, end - 1);
  const int m = sb[e] + ((end - 1 - sb[e]) & ~15);
  return max(beg, m);
}

// Ticket -> chunk [x0, x0 + D) of kind hk. sd: [0] hot pool stages, [1] cold pool stages, [2] chunks, [3] log2 D,
// [4] Sh, [5] Sc.
__device__ __forceinline__ void chunk37(const int* sd, int tk, int& hk, int& x0, int& x1) {
  const int dsh = sd[3], nhc = sd[0] >> dsh;
  hk = tk < nhc;
  x0 = hk ? sd[4] - sd[0] + (tk << dsh) : sd[5] - sd[1] + ((tk - nhc) << dsh);
  x1 = x0 + (1 << dsh);
}

// Warp 0: the CTA's static segments (v3.6's ranges36 / segs36 over [0, S - nd) per kind; role-split cost model by
// warp reduction, lane = expert) and the pool geometry. f = pool fraction in 64ths, D = 1 << (probe >> 20 & 7).
template <int NCH>
__device__ __forceinline__ void segs37(const Params& P, const Route& S, int f, int* sg, int* sd, int w4 = -1) {
  const bool kw1 = w4 > 0;            // KW1: K2-only hot weight, hot role >= half the grid
  if (w4 <= 0) w4 = P.hot_w4;
  const int lane = threadIdx.x & 31;
  const int G = (int)gridDim.x, c = (int)blockIdx.x;
  const int dsh = min((P.probe >> 20) & 7, 4);
  const int Sh = S.sbh[32], Sc = S.sbc[32];
  const int ndh = ((P.probe >> 23) & 1) ? 0 : (int)(((long long)Sh * f / 64) >> dsh) << dsh;
  const int ndc = (int)(((long long)Sc * f / 64) >> dsh) << dsh;
  const int Shs = Sh - ndh, Scs = Sc - ndc;
  int b[2], en[2], k[2];
  if (P.hot_mode & 1) {
    int hb, he, cb, ce;
    my_range2(Shs, c, G, hb, he);
    my_range2(Scs, c, G, cb, ce);
    const bool cold_first = (P.hot_mode & 2) && (2 * c >= G);
    k[0] = cold_first ? 0 : 1; b[0] = cold_first ? cb : hb; en[0] = cold_first ? ce : he;
    k[1] = 1 - k[0];           b[1] = cold_first ? hb : cb; en[1] = cold_first ? he : ce;
  } else {
    int Gh = 0;
    if (Sh > 0) {
      long long wh = 0, wc = 0;
      const int n = S.cnt[lane];
      if (n > 0) {
        if (n > P.hot_t) {
          const int tp = 8 * NCH, full = n / tp, rem = n - full * tp;
          wh = (long long)full * (16 + w4 * (NCH - 1)) + (rem ? 16 + w4 * (((rem + 7) >> 3) - 1) : 0);
        } else {
          wc = 16ll * ((n + 7) >> 3);
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) { wh += __shfl_xor_sync(FULL, wh, o); wc += __shfl_xor_sync(FULL, wc, o); }
      Gh = (Sc > 0) ? (int)((wh * G + (wh + wc) / 2) / (wh + wc)) : G;
      Gh = max(1, min(Gh, Sc > 0 ? G - 1 : G));
      if (kw1) Gh = min(max(Gh, G / 2), Sc > 0 ? G - 1 : G);   // KW1: keep a hot CTA on every SM
    }
    const bool hr = c < Gh;
    k[0] = hr ? 1 : 0;
    if (hr) my_range2(Shs, c, Gh, b[0], en[0]);
    else my_range2(Scs, c - Gh, G - Gh, b[0], en[0]);
    k[1] = 0; b[1] = 0; en[1] = 0;
  }
#pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int mid = tail_first37(k[part] ? S.sbh : S.sbc, b[part], en[part]);
    if (lane == 0) {
      sg[6 * part + 0] = mid;     sg[6 * part + 1] = en[part]; sg[6 * part + 2] = k[part];
      sg[6 * part + 3] = b[part]; sg[6 * part + 4] = mid;      sg[6 * part + 5] = k[part];
    }
  }
  if (lane == 0) { sd[0] = ndh; sd[1] = ndc; sd[2] = (ndh + ndc) >> dsh; sd[3] = dsh; sd[4] = Sh; sd[5] = Sc; }
}

template <int NST>
__device__ __forceinline__ void setup_bars37(uint64_t* bars) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NST; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + NST + i), 8); }
    for (int i = 0; i < NE37; ++i) { mbar_init(sa(bars + 2 * NST + i), 8); mbar_init(sa(bars + 2 * NST + NE37 + i), 1); }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
}

// v3.7 gemm: static segments pr[3 sg .. 3 sg + 2] (segs37), then dynamic chunks. Producer, consumers and epilogue walk
// the same unit sequence: static from pr, dynamic from the ring tags (consumers) and the mailbox (epilogue).
template <int NCH, int NST, int MODE>
__device__ void gemm37(const Params& P, Route& S, const int16_t* s_list, uint8_t* ring, uint64_t* bars, bool pdl) {
  constexpr int STG = SW + 8 * NCH * XROW;
  constexpr int NRB = (MODE == 1) ? 64 : 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  __shared__ int pr[12], sd[8], s_tag[NST];
  __shared__ int s_mb[NE37][8];
  if (warp == 0) segs37<NCH>(P, S, (MODE == 1) ? ((P.probe >> 8) & 63) : ((P.probe >> 14) & 63), pr, sd,
                             (MODE == 2) ? ((P.probe >> 26) & 31) : 0);   // KW1: K2 hot weight (0 = hot_w4)
  __syncthreads();
  const uint32_t ring_a = sa(ring);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);
  const uint32_t ef_a = sa(bars + 2 * NST), ee_a = sa(bars + 2 * NST + NE37);

  if (warp == 8) {
    // ---------------- producer ----------------
    // K2's X copies (interS from K1) are deferred until griddepcontrol.wait, taken lazily as in v3.3 / v3.6.
    const uint8_t* wsrc = (MODE == 1) ? P.ws13 : P.ws2;
    int* dq = P.meta + DQ37 + ((MODE == 1) ? 0 : 2);
    const int ndyn = sd[2];
    int it = 0;
    bool waited = !pdl;
    int nd = 0;
    const bool hint = (P.probe & 16) != 0;
    const uint64_t pol = policy_evict_first();
    uint32_t dDst[NST], dBar[NST];
    const __nv_bfloat16* dSrc[NST];
    uint32_t dB[NST];
    const int r = (lane + 31) & 31;
    // order: static segments 0-3, then the pool (a "pool after the tail-first item" order was measured +10..24 us)
    int fseg = -1;   // the last non-empty static segment: its last stage requests the first ticket
#pragma unroll
    for (int q = 0; q < 4; ++q) if (pr[3 * q + 1] > pr[3 * q]) fseg = q;
    int sg = 0, tnext = 0;
    bool fetched = false;
    auto take_wait = [&]() {
      pdl_wait();
      asm volatile("fence.proxy.async;\n" ::: "memory");
      waited = true;
      if (lane == 0) {
#pragma unroll
        for (int d = 0; d < NST; ++d)
          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
      }
      nd = 0;
    };
    while (true) {
      int hk, x0, x1, tag = 0;
      int fpos = -1;   // stage position at which the next ticket is requested (-1: none)
      if (sg < 4) {
        x0 = pr[3 * sg]; x1 = pr[3 * sg + 1]; hk = pr[3 * sg + 2];
        if (sg == fseg) fpos = x1 - 1;
        ++sg;
        if (x1 <= x0) continue;
      } else {
        if (!fetched && lane == 0 && ndyn > 0) tnext = atomicAdd(dq, 1);
        fetched = false;
        const int tk = __shfl_sync(FULL, tnext, 0);
        if (tk >= ndyn) {
          // END: the next slot carries tag -1 and completes without tx (1 + 32 arrivals)
          const int slot = it % NST;
          if (!waited && nd > 0 && it >= NST) take_wait();
          mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
          if (lane == 0) { s_tag[slot] = -1; mbar_arrive(full_a + 8 * slot); }
          mbar_arrive(full_a + 8 * slot);
          ++it;
          if (lane == 0 && ndyn > 0 && atomicAdd(dq + 1, 1) == (int)gridDim.x - 1) { atomicExch(dq, 0); atomicExch(dq + 1, 0); }
          break;
        }
        tag = tk;
        chunk37(sd, tk, hk, x0, x1);
        fpos = x0;
      }
      const int* sb = hk ? S.sbh : S.sbc;
      const int TP = hk ? 8 * NCH : 8;
      int e = find_e(sb, x0);
      for (int x = x0; x < x1;) {
        while (sb[e + 1] <= x) ++e;
        const int np = S.np[e];
        const int l = x - sb[e];
        const int item = l >> 4;
        const int s0 = l & 15;
        const int s1 = min(16, s0 + (x1 - x));
        const int rb = item / np, j = item - rb * np;
        const int ntok = min(TP, S.cnt[e] - j * TP);
        const int c0 = S.off[e] + j * TP;
        const uint8_t* wb = wsrc + ((size_t)(e * NRB + rb) * 16) * SW;
        int tok = 0;
        if (MODE == 1 && r < ntok) tok = ((int)s_list[c0 + r]) >> 3;
        const uint32_t xbytes = (MODE == 1) ? ntok * 512 : ntok * XROW;
        for (int s = s0; s < s1; ++s, ++it) {
          const int slot = it % NST;
          const uint32_t par = ((it / NST) & 1) ^ 1;
          if (!waited && nd > 0 && it >= NST) take_wait();   // a deferred slot is about to be reused
          if (x + (s - s0) == fpos && lane == 0 && ndyn > 0) { tnext = atomicAdd(dq, 1); fetched = true; }
          mbar_wait(empty_a + 8 * slot, par, P.err);
          const uint32_t dst = ring_a + slot * STG;
          const uint32_t fb = full_a + 8 * slot;
          if (lane == 0) {
            s_tag[slot] = tag;
            mbar_expect_tx(fb, SW + xbytes);
            const uint8_t* src = wb + (size_t)s * SW;
            if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
            else bulk_g2s(dst, src, SW, fb);
          }
          mbar_arrive(fb);   // the full barrier counts 1 expect_tx + 32 lane arrivals
          __syncwarp();
          if (MODE == 1) {
            if (r < ntok) bulk_g2s(dst + SW + r * XROW, P.x + (size_t)tok * KDIM + s * 256, 512, fb);
          } else {
            const __nv_bfloat16* src = P.inter + ((size_t)s * (8 * M3) + c0) * XST;
            if (waited) {
              if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);
            } else {
#pragma unroll
              for (int d = 0; d < NST; ++d)
                if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }
              ++nd;
            }
          }
        }
        x += s1 - s0;
      }
      fetched = __shfl_sync(FULL, (int)fetched, 0) != 0;
    }
    if (waited && !pdl) pdl_wait();   // no-op without a PDL predecessor; orders completion in a PDL chain
    if (!waited) take_wait();
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp: mailbox entries in order; split-item counters, the last contributor finishes ------
    // K1: SwiGLU(h) -> interS, h -> 0. K2: per-route sum of squares of z = bf16(y)*(1+w) over the item's 128 columns.
    for (int k = 0;; ++k) {
      const int i = k % NE37;
      if (!mbar_wait_ok(ef_a + 8 * i, (k / NE37) & 1, P.err)) break;
      const int e = s_mb[i][0], rb = s_mb[i][1], j = s_mb[i][2], ns = s_mb[i][3], ntok = s_mb[i][4], c0 = s_mb[i][5];
      __syncwarp();
      if (lane == 0) mbar_arrive(ee_a + 8 * i);
      if (e < 0) break;
      __threadfence();   // cumulative over the consumers' h / y reds (ordered before us by the mailbox barrier)
      int last = 0;
      if (lane == 0) {
        int* cp = ((MODE == 1) ? P.cnt : P.cnt2) + (e * NRB + rb) * 64 + j;
        const int old = atomicAdd(cp, ns);
        last = (old + ns == 16);
        if (last) atomicExch(cp, 0);
      }
      last = __shfl_sync(FULL, last, 0);
      if (MODE == 2 && last) {
        __threadfence();
        const int col = rb * 128 + 4 * lane;
        const float4 wv = __ldg(reinterpret_cast<const float4*>(P.nw + col));
        for (int n0 = 0; n0 < ntok; n0 += 4) {
          float4 yv[4];
#pragma unroll
          for (int u = 0; u < 4; ++u)
            if (n0 + u < ntok) yv[u] = __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)(c0 + n0 + u) * KDIM + col));
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            float pz = 0.f;
            if (n0 + u < ntok) {
              float z;
              z = bf16r(yv[u].x) * wv.x; pz += z * z;  z = bf16r(yv[u].y) * wv.y; pz += z * z;
              z = bf16r(yv[u].z) * wv.z; pz += z * z;  z = bf16r(yv[u].w) * wv.w; pz += z * z;
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) pz += __shfl_xor_sync(FULL, pz, o);
            if (lane == 0 && n0 + u < ntok) atomicAdd(P.ss + c0 + n0 + u, pz);
          }
        }
      }
      if (MODE == 1 && last) {
        __threadfence();
        const int cl = 2 * lane;
        const int col = rb * 64 + cl;
        for (int n0 = 0; n0 < ntok; n0 += 4) {
          float2 gv[4], uv[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            if (n0 + u < ntok) {
              const float* hp = P.h + (size_t)(c0 + n0 + u) * (2 * KDIM) + col;
              gv[u] = __ldcg(reinterpret_cast<const float2*>(hp));
              uv[u] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
            }
          }
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            if (n0 + u < ntok) {
              const int c = c0 + n0 + u;
              float* hp = P.h + (size_t)c * (2 * KDIM) + col;
              const uint32_t pk = pack_bf2(swiglu(gv[u].x, uv[u].x), swiglu(gv[u].y, uv[u].y));
              *reinterpret_cast<uint32_t*>(P.inter + ((size_t)(col >> 8) * (8 * M3) + c) * XST + (col & 255)) = pk;
              *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
              *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
            }
          }
        }
      }
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the row block) ----------------
  if (MODE == 1) {
    // y rows [0, R) and ss[0, R) -> 0 while the first stages are in flight (K2 accumulates into both after its wait)
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(P.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)blockIdx.x * 256 + tid; i < n4; i += (size_t)gridDim.x * 256) yp[i] = z;
    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;
  }
  int it = 0, k = 0, sg = 0;
  // hand an item to the epilogue warp: every consumer thread after its own reds (release at CTA scope)
  // (warp-level: __syncwarp orders the lanes' reds before lane 0's release-arrive; one arrival per warp, count 8)
  auto post = [&](int e, int rb, int j, int ns, int ntok, int c0) {
    const int i = k % NE37;
    mbar_wait(ee_a + 8 * i, ((k / NE37) & 1) ^ 1, P.err);
    if (tid == 0) { s_mb[i][0] = e; s_mb[i][1] = rb; s_mb[i][2] = j; s_mb[i][3] = ns; s_mb[i][4] = ntok; s_mb[i][5] = c0; }
    __syncwarp();
    if (lane == 0) mbar_arrive(ef_a + 8 * i);
    ++k;
  };
  while (true) {
    int hk, x0, x1;
    if (sg < 4) {
      x0 = pr[3 * sg]; x1 = pr[3 * sg + 1]; hk = pr[3 * sg + 2];
      ++sg;
      if (x1 <= x0) continue;
    } else {
      const int slot = it % NST;
      if (!mbar_wait_ok(full_a + 8 * slot, (it / NST) & 1, P.err)) break;
      const int tk = s_tag[slot];
      if (tk < 0) break;   // END (its slot is never released: the producer is done)
      chunk37(sd, tk, hk, x0, x1);
    }
    const int* sb = hk ? S.sbh : S.sbc;
    const int TP = hk ? 8 * NCH : 8;
    int e = find_e(sb, x0);
    for (int x = x0; x < x1;) {
      while (sb[e + 1] <= x) ++e;
      const int np = S.np[e];
      const int l = x - sb[e];
      const int item = l >> 4;
      const int s0 = l & 15;
      const int s1 = min(16, s0 + (x1 - x));
      const int rb = item / np, j = item - rb * np;
      const int ntok = min(TP, S.cnt[e] - j * TP);
      const int nch = (ntok + 7) >> 3;
      const int c0 = S.off[e] + j * TP;
      // K1 hot items with >= 3 chunks always go through the epilogue warp (NOSWI), whole or split
      const bool ho = (MODE == 1) && hk && nch >= 3;
      if (nch <= 1) {
        cold_item<NST, STG, MODE, true>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, 0);
      } else if constexpr (NCH >= 4) {
        if (nch >= 4) hc_item<4, NST, STG, MODE, true>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, 0, ho);
        else if (nch == 3) hc_item<3, NST, STG, MODE, true>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, 0, ho);
        else hc_item<2, NST, STG, MODE, true>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, 0, false);
      } else if constexpr (NCH >= 2) {
        hc_item<2, NST, STG, MODE, true>(P, ring, full_a, empty_a, it, s0, s1, e, rb, c0, ntok, 0, false);
      }
      // K1: the mailbox arguments are recomputed from (x, e) after the item (fresh smem loads after the item's mbarrier
      // asm), so they are not live across the 16-accumulator hot item (that kept K1 NCH 4 at 96 registers + 8 B spill)
      if constexpr (MODE == 2) {
        if (!(s0 == 0 && s1 == 16) || ho) post(e, rb, j, s1 - s0, ntok, c0);
        x += s1 - s0;
      } else {
        const int* sb2 = hk ? S.sbh : S.sbc;
        const int l2 = x - sb2[e];
        const int t0 = l2 & 15, t1 = min(16, t0 + (x1 - x));
        const int np2 = S.np[e], it2 = l2 >> 4;
        const int rb2 = it2 / np2, j2 = it2 - rb2 * np2;
        const int tp2 = hk ? 8 * NCH : 8;
        const int nt2 = min(tp2, S.cnt[e] - j2 * tp2);
        const bool ho2 = (MODE == 1) && hk && ((nt2 + 7) >> 3) >= 3;
        if (!(t0 == 0 && t1 == 16) || ho2) post(e, rb2, j2, t1 - t0, nt2, S.off[e] + j2 * tp2);
        x += t1 - t0;
      }
    }
  }
  post(-1, 0, 0, 0, 0, 0);
}

// K1 / K2 of the HK3 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.
template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k1_kernel37(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars37<NST>(bars);
  pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 5)
  route<1, 8 * NCH, true>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);
}

template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k2_kernel37(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars37<NST>(bars);
  route<1, 8 * NCH, true, false>(P, S, nullptr, nullptr, 32, false);
  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);
}

// K1: route + W13 (+ SwiGLU). HOTCAP (v3.3): warm experts on wgmma CTAs, their X from P0 (PDL chain, lazy wait).
template <int NC, int NST, int MINB, bool HOTCAP>
__global__ void __launch_bounds__(320, MINB) k1_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  if constexpr (!HOTCAP) pdl_wait();   // v4.7: K1 is a PDL secondary of the routing tail (vid 0 / 1)
  route<NC>(P, S, nullptr, s_list, 64, blockIdx.x == 0 && (P.probe & 15) == 0);
  if constexpr (HOTCAP) {
    // cold-role CTAs re-cut the same ring into more, smaller stages (no 32 KB warm X slot)
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 1, 0, STG>(P, S, s_list, smem, bars, false, blockIdx.x, gridDim.x);
  }
}

// K2: route + W2. PDL: weights of the first NST stages are requested before griddepcontrol.wait.
template <int NC, int NST, int MINB, bool HOTCAP>
__global__ void __launch_bounds__(320, MINB) k2_kernel(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = HOTCAP ? SW + 32768 : SW + 8 * NC * XROW;
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  if constexpr (!HOTCAP) setup_bars<NC, NST>(bars);
  route<NC>(P, S, nullptr, s_list, 32, false);
  if constexpr (HOTCAP) {
    constexpr int STGC = SW + 8 * NC * XROW, NSTC = (NST * STG) / STGC;
    int cta, ncta;
    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);
    if (threadIdx.x == 0) {
      const int n = hr ? NST : NSTC;
      for (int i = 0; i < n; ++i) { mbar_init(sa(bars + i), 33); mbar_init(sa(bars + n + i), 8); }
      asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);
    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);
  } else {
    gemm<NC, NST, 2, 0, STG>(P, S, s_list, smem, bars, true, blockIdx.x, gridDim.x);
  }
}

// P0 (v3.3): warm experts' x rows packed into wgmma's core-matrix layout (64-token passes), one 32 B piece per thread.
template <int NC>
__global__ void __launch_bounds__(256) p0_kernel(const Params P) {
  __shared__ int16_t s_list[P3];
  __shared__ Route S;
  pdl_launch();
  route<NC>(P, S, nullptr, s_list, 64, false);
  int tot = 0;
  for (int e = 0; e < 32; ++e) if (S.cnt[e] > P.hot_t) tot += S.cnt[e];
  uint32_t* xw = reinterpret_cast<uint32_t*>(P.xw);
  for (int idx = blockIdx.x * 256 + threadIdx.x; idx < tot * 256; idx += gridDim.x * 256) {
    int r = idx >> 8;
    const int piece = idx & 255, s2 = piece >> 4, i = (piece >> 2) & 3, t = piece & 3;
    int e = 0;
    for (; e < 32; ++e) {
      if (S.cnt[e] <= P.hot_t) continue;
      if (r < S.cnt[e]) break;
      r -= S.cnt[e];
    }
    const int j = r >> 6, n = r & 63;
    const int tok = ((int)s_list[S.off[e] + r]) >> 3;
    const uint4* src = reinterpret_cast<const uint4*>(P.x + (size_t)tok * KDIM + s2 * 256 + 64 * i + 16 * t);
    const uint4 a = __ldg(src), b = __ldg(src + 1);
    const uint32_t v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    const size_t u0 = xw_u32(S.hp[e] + j, 64, s2, n, i, t, 0);
#pragma unroll
    for (int q = 0; q < 8; ++q) xw[u0 + 32 * q] = v[q];
  }
}

// K3: post-expert norm + weighted sum, one pass (v3.1): the per-route sums of squares come from K2 (ss), so an item
// (token, column quarter) needs one batch of meta / weights / ss loads and one batch of y loads. PDL wait first.
__global__ void __launch_bounds__(256) k3_kernel(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fr[8];
  if ((P.probe >> 25) & 1) pdl_launch();   // SU1: the staged unpooler may launch now (its weights do not depend on K3)
  pdl_wait();
  const int tid = threadIdx.x;
  const int items = 4 * P.M;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int it = blockIdx.x; it < items; it += gridDim.x) {
    const int tok = it >> 2;
    if (tid < 8) {
      const int c = __ldcg(P.meta + MH + tok * 8 + tid);
      const float wt = __ldg(P.wts + tok * 8 + tid);
      s_fc[tid] = c;
      s_fr[tid] = (c >= 0) ? rsqrtf(__ldcg(P.ss + c) * (1.f / 4096.f) + 1e-8f) * wt : 0.f;
    }
    __syncthreads();
    const int col = (it & 3) * 1024 + 4 * tid;
    float4 v[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const int c = s_fc[s];
      v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
    }
    const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      const float r = s_fr[s];
      a0 += bf16r(v[s].x) * w.x * r; a1 += bf16r(v[s].y) * w.y * r;
      a2 += bf16r(v[s].z) * w.z * r; a3 += bf16r(v[s].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(P.out + (size_t)tok * KDIM + col) = o;
    __syncthreads();
  }
}

}  // namespace k3

template <int NC, int NST, bool HOT = false>
static constexpr int smem_bytes() { return NST * (k3::SW + 8 * NC * (HOT ? 512 : k3::XROW)) + 2 * NST * 8; }

// mask: 1 K1, 2 K2, 4 K3 (7 = full call). HOTCAP variants (v3.3) launch P0 first and K1 as a PDL secondary.
template <int NC, int NST, int MINB, bool HOTCAP>
static constexpr int smem_bytes3() {
  return HOTCAP ? NST * (k3::SW + 32768) + 2 * 8 * (NST * (k3::SW + 32768) / (k3::SW + 8 * NC * k3::XROW))
                : NST * (k3::SW + 8 * NC * k3::XROW) + 2 * NST * 8;
}

template <int NC, int NST, int MINB, int HOTCAP>
static void launch3(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes3<NC, NST, MINB, (bool)HOTCAP>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 1) {
    if (HOTCAP) {
      k3::p0_kernel<NC><<<G3, 256, 0, st>>>(p);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>, p));
    } else {
      cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
      at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail
      C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>, p));
      at[0].val.programmaticStreamSerializationAllowed = 1;
    }
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NC, int NST, int MINB, int HOTCAP>
static std::vector<int64_t> attrs3() {
  constexpr int SM = smem_bytes3<NC, NST, MINB, (bool)HOTCAP>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);
  one((const void*)k3::k2_kernel<NC, NST, MINB, (bool)HOTCAP>, 320);
  return r;
}

// v3.6 HK2 variants: K1 normal launch, K2 / K3 PDL (as the cold v3.3 variants); one NST-stage ring of hot-size slots.
template <int NCH, int NST>
static constexpr int smem_bytes36() { return NST * (k3::SW + 8 * NCH * k3::XROW) + 2 * NST * 8; }

template <int NCH, int NST>
static void launch36(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes36<NCH, NST>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 1) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel36<NCH, NST>, p));
    at[0].val.programmaticStreamSerializationAllowed = 1;
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel36<NCH, NST>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NCH, int NST>
static std::vector<int64_t> attrs36() {
  constexpr int SM = smem_bytes36<NCH, NST>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel36<NCH, NST>, 320);
  one((const void*)k3::k2_kernel36<NCH, NST>, 320);
  return r;
}

// v3.7 HK3 variants: same launch chain as HK2 (K1 normal launch, K2 / K3 PDL); ring + 2*NE37 mailbox barriers.
template <int NCH, int NST>
static constexpr int smem_bytes37() { return NST * (k3::SW + 8 * NCH * k3::XROW) + (2 * NST + 2 * k3::NE37) * 8; }

template <int NCH, int NST>
static void launch37(const k3::Params& p, int G, int G3, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes37<NCH, NST>();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 1) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    at[0].val.programmaticStreamSerializationAllowed = (int)((p.probe >> 24) & 1);   // v4.7: K1 after the tail
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k1_kernel37<NCH, NST>, p));
    at[0].val.programmaticStreamSerializationAllowed = 1;
  }
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_kernel37<NCH, NST>, p));
  }
  if (mask & 4) {
    cfg.gridDim = dim3(G3); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k3_kernel, p));
  }
}

template <int NCH, int NST>
static std::vector<int64_t> attrs37() {
  constexpr int SM = smem_bytes37<NCH, NST>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, SM);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, SM) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + SM), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel37<NCH, NST>, 320);
  one((const void*)k3::k2_kernel37<NCH, NST>, 320);
  return r;
}

void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe) {
  k3::Params p;
  p.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ids = ids.data_ptr<int>();
  p.wts = wts.data_ptr<float>();
  p.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr());
  p.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  p.gsc13 = gsc13.data_ptr<float>();
  p.gsc2 = gsc2.data_ptr<float>();
  p.nw = nw.data_ptr<float>();
  p.h = h.data_ptr<float>();
  p.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr());
  p.y = y.data_ptr<float>();
  p.meta = meta.data_ptr<int>();
  p.cnt = cnt.data_ptr<int>();
  p.err = err.data_ptr<int>();
  p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  p.sink = reinterpret_cast<unsigned*>(sink.data_ptr());
  p.ss = ss.data_ptr<float>();
  p.cnt2 = cnt2.data_ptr<int>();
  p.xw = reinterpret_cast<uint8_t*>(xw.data_ptr());
  p.xw2 = reinterpret_cast<uint8_t*>(xw2.data_ptr());
  p.hot_t = (int)(hot_t % 1000);
  p.hot_w4 = (hot_t >= 1000) ? (int)((hot_t / 1000) % 100) : 8;   // x8 (v3.3) / sixteenths (v3.6 role split)
  p.hot_mode = (int)(hot_t / 100000);                             // v3.6 balance mode (HK2 only)
  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;
  switch (vid) {
@CASES3@
    default: TORCH_CHECK(false, "k3: unknown variant");
  }
}

std::vector<int64_t> attrs3v(int64_t vid) {
  switch (vid) {
@ATTRS3@
    default: return std::vector<int64_t>{-1};
  }
}
"""

K2A33_CPP_SRC = r"""
#include <torch/extension.h>
#include <vector>
void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe);
std::vector<int64_t> attrs3v(int64_t vid);
"""

def _k2a33_wg_specs():
    """wgmma.mma_async m64nNk16 (RS form) specializations for N = 16, 32, 48 (64 is hand-written above)."""
    out = []
    for n in (16, 32, 48):
        k = n // 2
        regs = ",".join("%%%d" % i for i in range(k))
        dl = ", ".join('"+f"(d[%d])' % i for i in range(k))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint64_t db) {\n  asm volatile(K3_WG_HEAD \"m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, "
            "%%%d, p, 1, 1, 0;\\n}\\n\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(db));\n}"
            % (n, n, regs, k, k + 1, k + 2, k + 3, k + 4, dl))
    return "\n".join(out)

def _k2a33_build_ext(name, variants, verbose=False):
    from torch.utils.cpp_extension import load_inline
    cases, attrs = _k2a36_variant_cases(variants)
    if list(variants) == list(K2A33_VARIANTS):
        assert (cases, attrs) == (_k2a36_gate_cases(), _k2a36_gate_attrs()), "gate cases out of date"
    src = K2A33_KERNEL_SRC.replace("@CASES3@", cases).replace("@ATTRS3@", attrs).replace("@WGSPECS@", _k2a33_wg_specs())
    tag = hashlib.sha1((K2A33_CPP_SRC + src).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=K2A33_CPP_SRC, cuda_sources=src, functions=["run3", "attrs3v"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)   # wgmma: sm_90a
    return mod, time.time() - t0

_K2A33_OWN_BUILD = _k2a33_build_ext

def _k2a33_prepare_st(weights, st2=None, variants=K2A33_VARIANTS, name="k2a_v37", log=print, share=None):
    """Stage repack (from v2.2's repacked tensors when given, else repacked here) + build + occupancy."""
    dev = weights["w13"].device
    t0 = time.time()
    if share is not None:   # another v3 state's repacked stages (same layout; the bench shares k2a_v33's)
        ws13, ws2 = share["ws13"], share["ws2"]
    else:
        ws13 = torch.empty((K2A_E_LOC, 64, 16, K2A33_SW3), dtype=torch.uint8, device=dev)
        ws2 = torch.empty((K2A_E_LOC, 32, 16, K2A33_SW3), dtype=torch.uint8, device=dev)
        for e in range(K2A_E_LOC):
            w13q, s13q = _k2a31_prepack_expert(weights["w13"][e], weights["w13_sf"][e], gate_up=True, bitpos=K2A31_BITPOS3)
            w2q, s2q = _k2a31_prepack_expert(weights["w2"][e], weights["w2_sf"][e], gate_up=False, bitpos=K2A31_BITPOS3)
            ws13[e] = _k2a33_stage_repack(w13q, s13q)
            ws2[e] = _k2a33_stage_repack(w2q, s2q)
    st = dict(
        ws13=ws13, ws2=ws2,
        gsc13=(weights["w13_gs"][:K2A_E_LOC].float() * K2A_W_SHIFT).contiguous(),
        gsc2=(weights["w2_gs"][:K2A_E_LOC].float() * K2A_W_SHIFT).contiguous(),
        nw=(weights["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous(),
        h=torch.zeros((8 * K2A33_M3, 2 * K2A_D_EXPERT), dtype=torch.float32, device=dev),
        inter=torch.zeros((16 * 8 * K2A33_M3 * 264,), dtype=torch.bfloat16, device=dev),
        y=torch.zeros((8 * K2A33_M3, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
        meta=torch.zeros((K2A_META_HDR + 8 * K2A33_M3,), dtype=torch.int32, device=dev),
        cnt=torch.zeros((K2A_E_LOC * 64 * 64,), dtype=torch.int32, device=dev),
        err=torch.zeros((4,), dtype=torch.int32, device=dev),
        sink=torch.zeros((4,), dtype=torch.int32, device=dev),
        ss=torch.zeros((8 * K2A33_M3,), dtype=torch.float32, device=dev),
        cnt2=torch.zeros((K2A_E_LOC * 32 * 64,), dtype=torch.int32, device=dev),
        # warm passes (> K2A33_HOT_T tokens, 64 per pass): <= 8 * K2A33_M3 / (K2A33_HOT_T + 1) + 32 passes x 16 stages x 64 x 512 B
        xw=torch.zeros(((8 * K2A33_M3 // (K2A33_HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
        xw2=torch.zeros(((8 * K2A33_M3 // (K2A33_HOT_T + 1) + 32) * 16 * 64 * 512,), dtype=torch.uint8, device=dev),
    )
    torch.cuda.synchronize()
    log(f"[k2a3] stage repack in {time.time() - t0:.1f}s")
    mod, dt = _k2a33_build_ext(name, variants)
    log(f"[k2a3] compiled {len(variants)} variants in {dt:.1f}s")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    grid = []
    for i, v in enumerate(variants):
        a = mod.attrs3v(i)
        occ = max(1, min(a[3], a[7], v[2]))
        grid.append(sms * occ)
        log(f"[k2a3] v{i} {tuple(v)}: K1 regs {a[0]} spill {a[1]} smem {a[2]} occ {a[3]} | K2 regs {a[4]} "
            f"spill {a[5]} smem {a[6]} occ {a[7]} -> grid {grid[-1]}"
            + ("" if min(a[3], a[7]) >= v[2] else f"  <-- WARNING: occupancy below {v[2]} CTAs/SM"))
    st.update(mod=mod, grid=grid, sms=sms, variants=variants)
    return st

K2A47_K1_PDL = True

K2A47_K1_PDL_T128 = False

K2A33_HINT = 16

def _k2a33_call(st, data, vid, probe=K2A33_HINT, mask=7, hot_t=None, hot_w4=None):
    v = st["variants"][vid]
    hot_t = (K2A33_HOT_T if hot_t is None else hot_t) if (len(v) >= 4 and v[3]) else 0
    hot_t += 1000 * (K2A33_HOT_W4 if hot_w4 is None else hot_w4)
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], hot_t, m,
                   data["config"]["local_start"], st["grid"][vid], _su1_g3(st, m), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))
    return out

def _k2a33_reset(st):
    st["h"].zero_()
    st["cnt"].zero_()
    st["cnt2"].zero_()
    torch.cuda.synchronize()

K2A33_SELF_TEST_CASES = ((1, 11, "normal"), (5, 12, "normal"), (16, 17, "normal"), (33, 13, "normal"),
                      (100, 14, "normal"), (128, 15, "skewed"), (257, 16, "normal"), (320, 18, "normal"))

def _v11_k2a33_call(x, local_ids, topk_weights, st):
    """K2A v3.3 on the layer: x [T, 4096] BF16 (pre-expert-normed), local ids (0..31, remote -1; lo = 0). v14: K2A v3.6's
    role-split variant where K2A36_PLAN routes the bucket, st["r36"] holds it (its prepare self-test passed) and K2A36_ENABLE /
    K2A36_BUCKETS allow it (read at call time); a host exception from that call drops the v3.6 routes for the run (once,
    with a line; the call is made on v3.3's plan, nothing was launched). v17: K2A v3.7's vid 5 first where K2A37_PLAN routes the bucket,
    st["r37"] holds it (its prepare self-test passed) and K2A37_ENABLE allows it (read at call time); a host exception from that call drops
    the v3.7 routes for the run (once, with a line; the call then goes on with the v3.6 / v3.3 routes below, nothing was launched).
    Otherwise exactly k3_v16's call."""
    x = x.contiguous()
    if x.data_ptr() % 16:   # bulk copies need 16 B-aligned rows (a pointer property, not a value)
        x = x.clone()
    data = {"hidden_states": x, "topk_ids": local_ids, "topk_weights": topk_weights, "config": {"local_start": 0}}
    rep = _k2a_bucket_of(x.shape[0])
    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
    if r37 is not None:
        try:
            return _k2a37_call(st, data, r37[0], r37[1], r37[2])
        except Exception as ex:  # noqa: BLE001
            print(f"[k3] K2A v3.7 route raised ({ex!r}): v3.6 / v3.3 routes from here on", flush=True)
            st["r37"].clear()
    r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
    if r36 is not None:
        try:
            return _k2a36_call(st, data, r36[0], r36[1])
        except Exception as ex:  # noqa: BLE001
            print(f"[k3] K2A v3.6 route raised ({ex!r}): v3.3's plan from here on", flush=True)
            st["r36"].clear()
    return _k2a33_call(st, data, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep),
                       hot_w4=(76 if rep == 256 else None))

K2A36_ENABLE = True

K2A36_BUCKETS = {16: True, 32: True}

K2A36_PLAN = {16: (3, (8, 0, 8)), 32: (3, (8, 0, 12))}

K2A36_HOT_DEFAULT = (8, 3, 8)

K2A36_SELF_TEST_CASES = ((1, 11, "normal"), (5, 12, "normal"), (12, 23, "skewed"), (16, 17, "normal"), (16, 19, "skewed"),
                      (20, 25, "hot8"), (24, 22, "skewed"), (32, 21, "skewed"), (32, 26, "hot8"), (33, 13, "normal"),
                      (100, 14, "normal"), (128, 15, "skewed"), (257, 16, "normal"), (320, 18, "normal"))

def _k2a36_synth_inputs(bs, seed, device, dist="normal"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, K2A_D_HIDDEN), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=device)
        u = torch.rand((256,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    elif dist == "hot8":   # v3.6 self-test: every token picks experts 0-7 (8 experts x bs tokens: nch = ceil(bs / 8))
        scores[:, :8] += 10.0
    else:
        scores = scores + torch.rand((256,), generator=gen, device=device).pow(6) * 4
    top = scores.topk(K2A_TOP_K, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {
        "hidden_states": x,
        "topk_ids": ids.to(torch.int32),
        "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float(),
        "config": {"bs": bs, "d_hidden": K2A_D_HIDDEN, "d_expert": K2A_D_EXPERT, "n_experts": 256,
                   "local_start": 0, "local_count": K2A_E_LOC, "top_k": K2A_TOP_K, "sharding": "ep8"},
    }

def _k2a36_variant_cases(variants):
    """(run3 switch cases, attrs3v switch cases) for a variant list: HK 0/1 -> launch3 (v3.3), HK 2 -> launch36."""
    cases, attrs = [], []
    for i, v in enumerate(variants):
        nc, nst, minb, hk, nch = (tuple(v) + (0,) * (5 - len(v)))[:5]
        if hk == 2:
            assert nc == 1 and minb == 2 and nch in (2, 4), v
            cases.append("    case %d: launch36<%d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nch, nst))
            attrs.append("    case %d: return attrs36<%d, %d>();" % (i, nch, nst))
        elif hk == 3:
            assert nc == 1 and minb == 2 and nch in (1, 2, 4), v
            cases.append("    case %d: launch37<%d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nch, nst))
            attrs.append("    case %d: return attrs37<%d, %d>();" % (i, nch, nst))
        else:
            cases.append("    case %d: launch3<%d, %d, %d, %d>(p, (int)G, (int)G3, (int)mask); break;" % (i, nc, nst, minb, hk))
            attrs.append("    case %d: return attrs3<%d, %d, %d, %d>();" % (i, nc, nst, minb, hk))
    return "\n".join(cases), "\n".join(attrs)

def _k2a36_gate_cases():
    """sass_gate --subst '@CASES3@=call:_k2a36_gate_cases': run3 cases of the default K2A33_VARIANTS (literal copy; the module's
    own build asserts it matches _k2a36_variant_cases(K2A33_VARIANTS))."""
    return ("    case 0: launch3<1, 3, 2, 0>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 1: launch3<2, 3, 2, 0>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 2: launch3<2, 2, 2, 1>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 3: launch36<4, 3>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 4: launch37<1, 3>(p, (int)G, (int)G3, (int)mask); break;\n"
            "    case 5: launch37<4, 3>(p, (int)G, (int)G3, (int)mask); break;")

def _k2a36_gate_attrs():
    return ("    case 0: return attrs3<1, 3, 2, 0>();\n"
            "    case 1: return attrs3<2, 3, 2, 0>();\n"
            "    case 2: return attrs3<2, 2, 2, 1>();\n"
            "    case 3: return attrs36<4, 3>();\n"
            "    case 4: return attrs37<1, 3>();\n"
            "    case 5: return attrs37<4, 3>();")

def _k2a36_hot_code(variant, hot=None):
    """run3's hot_t argument: hot_t + 1000 * hot_w4 + 100000 * hot_mode (v3.6 HK2), v3.3's hot_t + 1000 * K2A33_HOT_W4."""
    hk = (tuple(variant) + (0, 0))[3]
    if hk in (2, 3):
        ht, mode, w4 = K2A36_HOT_DEFAULT if hot is None else hot
        assert 0 <= ht < 1000 and 1 <= w4 < 100 and 0 <= mode < 4, hot
        return ht + 1000 * w4 + 100000 * mode
    if hk == 1:
        return (K2A33_HOT_T if hot is None else hot[0]) + 1000 * K2A33_HOT_W4
    return 1000 * K2A33_HOT_W4

def _k2a36_call(st, data, vid, hot=None, probe=K2A33_HINT, mask=7):
    """One call of variant vid; hot = (hot_t, hot_mode, hot_w4) for HK2 variants (default K2A36_HOT_DEFAULT). HK2 serves
    M <= 48 only (the split-item barrier-id budget, k2a_v36_emulate.py)."""
    x = data["hidden_states"].contiguous()
    if (tuple(st["variants"][vid]) + (0, 0))[3] == 2 and x.shape[0] > 48:
        raise ValueError("k2a_v36: HK2 variants serve M <= 48")
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], _k2a36_hot_code(st["variants"][vid], hot), m,
                   data["config"]["local_start"], st["grid"][vid], _su1_g3(st, m), probe | ((1 << 24) if K2A47_K1_PDL else 0) | ((1 << 25) if _SU1_FLAG["k3trig"] else 0))
    return out

def _v14_self_test_inputs(bs, seed, device, dist="normal"):
    """_k2a_self_test_inputs on k2a_v36's synthetic inputs (which add the "hot8" distribution): this GPU's experts are
    0..31, remote ids mapped to -1 when R1_LOCAL_ID_TABLE is on (the id convention the layer hands K2A)."""
    d = _k2a36_synth_inputs(bs, seed, device, dist)
    if R1_LOCAL_ID_TABLE:
        ids = d["topk_ids"]
        d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
    return d

def _v14_k2a36_prepare(st, w, dev):
    """K2A v3.6's routes (K2A36_PLAN) per bucket: k2a_v36's self-test cases of the bucket (normal / skewed / hot8 = every
    token on the same 8 experts, i.e. 4-chunk hot passes) through _k2a36_call on the layer's id convention, two calls per
    case (the second checks that h / the counters were restored), rel-L2 < 1e-2, worst row < 5e-2, no timeout flag. Sets
    st["r36"] = {bucket rep: (vid, hot)} for the buckets that pass; any other bucket (a failure, a variant not built or
    below its CTAs/SM target, K2A36_ENABLE off at prepare) keeps v3.3's plan. Prepare-time only."""
    st["r36"] = {}
    print(f"[v14] levers: K2A36_ENABLE={K2A36_ENABLE} K2A36_BUCKETS={K2A36_BUCKETS} K2A36_PLAN={K2A36_PLAN} "
          f"K2A33_PLAN={K2A33_PLAN} K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A36_ENABLE:
        return
    t2 = time.time()
    for rep, (vid, hot) in sorted(K2A36_PLAN.items()):
        if vid >= len(st["variants"]) or st["grid"][vid] < st["sms"] * st["variants"][vid][2]:
            print(f"[k2a36] bucket {rep}: v{vid} not built or below its CTAs/SM target -> v3.3's plan", flush=True)
            continue
        good_all = True
        for bs, seed, dist in K2A36_SELF_TEST_CASES:
            if _k2a_bucket_of(bs) != rep:
                continue
            try:
                d = _v14_self_test_inputs(bs, seed, dev, dist)
                ref = _k2a_ref_fp32(d, w)
                data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                        "config": {"local_start": 0}}
                for k in range(2):
                    st["err"].zero_()
                    got = _k2a36_call(st, data, vid, hot)
                    torch.cuda.synchronize()
                    tot, row = _k2a_rel_err(got, ref)
                    errf = int(st["err"][0])
                    good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                    print(f"[k2a36] layer self-test bs={bs} {dist} v{vid} hot={hot} call {k}: rel {tot:.3e} worst row "
                          f"{row:.3e} err={errf} -> {'ok' if good else 'FAIL'}", flush=True)
                    good_all &= good
            except Exception as ex:  # noqa: BLE001
                print(f"[k2a36] self-test bs={bs} v{vid} raised {ex!r}", flush=True)
                good_all = False
            _k2a33_reset(st)
            st["err"].zero_()
            if not good_all:
                break
        if good_all:
            st["r36"][rep] = (vid, tuple(hot))
        print(f"[k2a36] bucket {rep}: v{vid} hot {hot} -> {'route live' if good_all else 'FAILED: v3.3 plan'}", flush=True)
    print(f"[v14] K2A v3.6 routes live {st['r36']} (self-tests {time.time() - t2:.1f}s)", flush=True)

K2A37_ENABLE = True

K2A37_PLAN = {128: (5, (8, 0, 12), (8, 0, 3))}

K2A37_DQ = 120

def _k2a37_dyn_code(dyn):
    """probe bits of the v3.7 tail queue: f1 << 8 | f2 << 14 | log2(D) << 20 | cold_only << 23 (f in 64ths, D = chunk
    stages; cold_only: the pool holds cold chunks only, hot stages stay static)."""
    if dyn is None:
        return 0
    f1, f2, dsh, cold_only = (tuple(dyn) + (0,))[:4]
    assert 0 <= f1 < 64 and 0 <= f2 < 64 and 0 <= dsh <= 4, dyn
    return (f1 << 8) | (f2 << 14) | (dsh << 20) | ((1 << 23) if cold_only else 0)

KW1_K2_W4 = 5

def _k2a37_call(st, data, vid, hot=None, dyn=None, probe=K2A33_HINT, mask=7):
    """One call of variant vid: hot = (hot_t, hot_mode, hot_w4) for HK2 / HK3 variants, dyn = (f1, f2, log2 D) for HK3
    (None = static plan). HK2 serves M <= 48 (named-barrier ids); HK3 (mailbox epilogue) every M <= 320."""
    x = data["hidden_states"].contiguous()
    v = tuple(st["variants"][vid]) + (0, 0)
    if v[3] == 2 and x.shape[0] > 48:
        raise ValueError("k2a_v37: HK2 variants serve M <= 48")
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    pb = probe | (_k2a37_dyn_code(dyn) if v[3] == 3 else 0)
    if v[3] == 3 and 0 < KW1_K2_W4 < 32:   # KW1: K2-only hot weight (probe bits 26-30), read at call time
        pb |= KW1_K2_W4 << 26
    st["mod"].run3(vid, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], st["h"],
                   st["inter"], st["y"], st["meta"], st["cnt"], st["err"], out, st["sink"], st["ss"], st["cnt2"],
                   st["xw"], st["xw2"], _k2a36_hot_code(st["variants"][vid], hot), m,
                   data["config"]["local_start"], st["grid"][vid], st["sms"] * 2, pb | ((1 << 24) if K2A47_K1_PDL_T128 else 0))
    return out

def _v17_k2a37_prepare(st, w, dev):
    """K2A v3.7's routes (K2A37_PLAN) per bucket: k2a_v37's self-test cases of the bucket (bucket 128: M = 100 normal, M = 128 skewed) through
    _k2a37_call on the layer's id convention, two calls per case (the second checks that h / the counters were restored), rel-L2 < 1e-2,
    worst row < 5e-2, no timeout flag, finite, and the ticket counters meta[K2A37_DQ .. +3] back at zero after every call. Sets
    st["r37"] = {bucket rep: (vid, hot, dyn)} for the buckets that pass; any other bucket (a failure, a variant not built or below its CTAs/SM
    target, K2A37_ENABLE off at prepare) keeps k3_v16's route. Prepare-time only."""
    st["r37"] = {}
    print(f"[v17] levers: K2A37_ENABLE={K2A37_ENABLE} K2A37_PLAN={K2A37_PLAN} K2A36_PLAN={K2A36_PLAN} K2A33_PLAN={K2A33_PLAN} "
          f"K2A33 variants={len(K2A33_VARIANTS)}", flush=True)
    if not K2A37_ENABLE:
        return
    t2 = time.time()
    for rep, (vid, hot, dyn) in sorted(K2A37_PLAN.items()):
        if vid >= len(st["variants"]) or st["grid"][vid] < st["sms"] * st["variants"][vid][2]:
            print(f"[k2a37] bucket {rep}: v{vid} not built or below its CTAs/SM target -> k3_v16's route", flush=True)
            continue
        good_all = True
        for bs, seed, dist in K2A36_SELF_TEST_CASES:
            if _k2a_bucket_of(bs) != rep:
                continue
            try:
                d = _v14_self_test_inputs(bs, seed, dev, dist)
                ref = _k2a_ref_fp32(d, w)
                data = {"hidden_states": d["hidden_states"], "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
                        "config": {"local_start": 0}}
                for k in range(2):
                    st["err"].zero_()
                    got = _k2a37_call(st, data, vid, hot, dyn)
                    torch.cuda.synchronize()
                    tot, row = _k2a_rel_err(got, ref)
                    errf = int(st["err"][0])
                    dq = st["meta"][K2A37_DQ:K2A37_DQ + 4].tolist()
                    good = tot < 1.0e-2 and row < 5.0e-2 and not errf and not any(dq) and bool(torch.isfinite(got.float()).all())
                    print(f"[k2a37] layer self-test bs={bs} {dist} v{vid} hot={hot} dyn={dyn} call {k}: rel {tot:.3e} worst row "
                          f"{row:.3e} err={errf} counters={dq} -> {'ok' if good else 'FAIL'}", flush=True)
                    good_all &= good
            except Exception as ex:  # noqa: BLE001
                print(f"[k2a37] self-test bs={bs} v{vid} raised {ex!r}", flush=True)
                good_all = False
            _k2a33_reset(st)
            st["err"].zero_()
            if not good_all:
                break
        if good_all:
            st["r37"][rep] = (vid, tuple(hot), tuple(dyn))
        print(f"[k2a37] bucket {rep}: v{vid} hot {hot} dyn {dyn} -> {'route live' if good_all else 'FAILED: k3_v16 route'}", flush=True)
    print(f"[v17] K2A v3.7 routes live {st['r37']} (self-tests {time.time() - t2:.1f}s)", flush=True)

_SU1_FLAG = {"k3trig": False}

SU1_K3_GRID = True

def _su1_g3(st, m):
    """K3's grid: 4 m (one item per CTA) when this call's K3 triggers early, else K2A's 2 x SMs."""
    return max(4 * m, 1) if (SU1_K3_GRID and _SU1_FLAG["k3trig"]) else st["sms"] * 2


# ---------------------------------------------------------------------------------------------------------------
# study-a board wrapper (tools/sa_moe_extract.py): the layer's K2A MoE on the muse_moe_w4a16 contract
def _sa_prepare(weights):
    t0 = time.time()
    dev = weights["w13"].device
    w = {"w13": weights["w13"], "w13_sf": weights["w13_sf"], "w13_gs": weights["w13_gs"],
         "w2": weights["w2"], "w2_sf": weights["w2_sf"], "w2_gs": weights["w2_gs"],
         "post_norm_w": weights["post_norm_w"]}
    state = {"w": w, "st": None, "ok": {}}
    try:
        st = _k2a33_prepare_st(w, log=lambda s: print(s, flush=True))
    except Exception as ex:  # noqa: BLE001
        print("[sa_moe] K2A repack/build FAILED, torch fallback everywhere. Tail:\n" + str(ex)[-8000:], flush=True)
        return state
    ok = {rep: True for _, _, rep in K2A_BUCKETS}
    for bs, seed, dist in K2A33_SELF_TEST_CASES:
        rep = _k2a_bucket_of(bs)
        try:
            d = _k2a_self_test_inputs(bs, seed, dev, dist)
            ref = _k2a_ref_fp32(d, w)
            for k in range(2):
                st["err"].zero_()
                got = _v11_k2a33_call(d["hidden_states"], d["topk_ids"], d["topk_weights"], st)
                torch.cuda.synchronize()
                tot, row = _k2a_rel_err(got, ref)
                errf = int(st["err"][0])
                good = tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(got.float()).all())
                print(f"[sa_moe] self-test bs={bs} {dist} call {k}: rel {tot:.3e} worst row {row:.3e} err={errf} -> "
                      f"{'ok' if good else 'FAIL'}", flush=True)
                if not good:
                    ok[rep] = False
        except Exception as ex:  # noqa: BLE001
            print(f"[sa_moe] self-test bs={bs} raised {ex!r}", flush=True)
            ok[rep] = False
        _k2a33_reset(st)
    _v14_k2a36_prepare(st, w, dev)
    _v17_k2a37_prepare(st, w, dev)
    torch.cuda.synchronize()
    state["st"], state["ok"] = st, ok
    print(f"[sa_moe] K2A buckets {ok}; r36 {st.get('r36')}; r37 {st.get('r37')}; prepare {time.time() - t0:.1f}s",
          flush=True)
    return state


def _sa_torch_moe(data, w):
    x, ids, tw = data["hidden_states"], data["topk_ids"].long(), data["topk_weights"].float()
    lo = int(data["config"]["local_start"])
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
    loc = ids - lo
    for e in range(K2A_E_LOC):
        tok, slot = (loc == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        d = {"hidden_states": x[tok], "topk_ids": torch.zeros((tok.numel(), 1), dtype=torch.int32, device=x.device),
             "topk_weights": torch.ones((tok.numel(), 1), dtype=torch.float32, device=x.device)}
        y = _k2a_ref_fp32_one(d["hidden_states"], w, e)
        out.index_add_(0, tok, y * tw[tok, slot].unsqueeze(1))
    return out.to(torch.bfloat16)


def _k2a_ref_fp32_one(x, w, e):
    d = {"hidden_states": x, "topk_ids": torch.full((x.shape[0], 1), e, dtype=torch.int32, device=x.device),
         "topk_weights": torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)}
    return _k2a_ref_fp32(d, w).float()


def _sa_kernel_fn(data, state):
    st = state["st"]
    x = data["hidden_states"]
    m = x.shape[0]
    rep = _k2a_bucket_of(m)
    cfg = data["config"]
    if st is not None and rep is not None and state["ok"].get(rep) and m <= K2A33_M3:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
        # _v11_k2a33_call's dispatch, with the board's global ids and local_start (K2A's route maps ids to local
        # experts modulo 256 against local_start; the layer passes local ids with local_start 0)
        d2 = {"hidden_states": x, "topk_ids": data["topk_ids"], "topk_weights": data["topk_weights"],
              "config": {"local_start": int(cfg["local_start"])}}
        r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
        if r37 is not None:
            return _k2a37_call(st, d2, r37[0], r37[1], r37[2])
        r36 = st["r36"].get(rep) if (K2A36_ENABLE and K2A36_BUCKETS.get(rep) and st.get("r36")) else None
        if r36 is not None:
            return _k2a36_call(st, d2, r36[0], r36[1])
        return _k2a33_call(st, d2, K2A33_PLAN[rep], hot_t=K2A33_HOT_T_BUCKET.get(rep), hot_w4=(76 if rep == 256 else None))
    return _sa_torch_moe(data, state["w"])

VMS_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef VMS_NS
#define VMS_NS 3
#endif
#ifndef VMS_G
#define VMS_G 132
#endif

namespace vms {
constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;
constexpr int UB = 36864, UC = 32768, RBA = 512, RBB = 256;
constexpr int NW = 8, NT = NW * 32, NTL = NT + 32, NRB = 4, NS = VMS_NS, G = VMS_G;
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

// unit q of this CTA's sequence (phase-A units a0.., then phase-B units b0..) -> global source
__device__ __forceinline__ const unsigned char* unit_src(const Params& p, const Smem& S, int q, int nA, int a0, int b0) {
  if (q < nA) { const int v = a0 + q; return p.wA + ((size_t)S.jE[v >> 9] * RBA + (v & 511)) * UB; }
  const int v = b0 + (q - nA); return p.wB + ((size_t)S.jE[v >> 8] * RBB + (v & 255)) * UB;
}

// sm_90: thread 0 only (bulk copy + expect_tx); sm_86: every compute thread (cp.async + arrive.noinc, count NT)
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

__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {
  extern __shared__ __align__(128) unsigned char smem[];
  Smem& S = *reinterpret_cast<Smem*>(smem);
  unsigned char* ring = smem + RING_OFF;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  const bool mk = p.marks != nullptr;
  long long* mrow = p.marks + (size_t)c * 16;
  if (tid == 0) {
    if (mk) mrow[0] = gt();
#if __CUDA_ARCH__ >= 900
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], 1);
#else
    for (int i = 0; i < NS; ++i) mbar_init(&S.full[i], NT);
#endif
    for (int i = 0; i < NRB; ++i) { mbar_init(&S.pfull[i], 1); mbar_init(&S.pempty[i], 1); }
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
  const int nA = a1 - a0, N = t1 - t0;
  if (mk && tid == 0) mrow[1] = gt();
  if (warp < NW) {
    // ================= compute warps
    for (int q = 0; q < NS && q < N; ++q) {
#if __CUDA_ARCH__ >= 900
      if (tid == 0)
#endif
      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);
    }
    const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);   // opaque 16: the scale build's multiply stays an IMAD
    const int g = lane >> 2, t = lane & 3;
    uint4 xr[16];
    int curj = -1;
    int sl = 0; uint32_t par = 0;
    for (int q = 0; q < N; ++q) {
      const bool PB = q >= nA;
      const int v = PB ? b0 + (q - nA) : a0 + q;
      const int j = PB ? (v >> 8) + 4096 : (v >> 9);
      if (j != curj) {
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
      mbar_wait(&S.full[sl], par);
      if (mk && tid == 0 && q == 0) mrow[2] = gt();
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
          const uint32_t r1 = (qw * 16u) & M;
          const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
          const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);
        }
      }
      float4 o;
      o.x = acc[0][0] + acc[1][0] + acc[2][0] + acc[3][0];
      o.y = acc[0][1] + acc[1][1] + acc[2][1] + acc[3][1];
      o.z = acc[0][2] + acc[1][2] + acc[2][2] + acc[3][2];
      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));
      S.red[bq][warp][lane] = o;
      cta_sync_compute();
      if (q + NS < N) {
#if __CUDA_ARCH__ >= 900
        if (tid == 0)
#endif
        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);
      }
      if (tid == 0) mbar_arrive(&S.pfull[bq]);
      if (++sl == NS) { sl = 0; par ^= 1u; }
    }
    if (mk && tid == 0) mrow[5] = gt();
  } else {
    // ================= helper warp: per-unit K reduction + epilogue
    const int g = lane >> 2, t = lane & 3;
    float* ssc = p.ss + (S.gen & 1) * MAXP;
    for (int q = 0; q < N; ++q) {
      const int bq = q & (NRB - 1);
      mbar_wait(&S.pfull[bq], (uint32_t)((q / NRB) & 1));
      float4 sm = S.red[bq][0][lane];
#pragma unroll
      for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
      __syncwarp();
      if (lane == 0) mbar_arrive(&S.pempty[bq]);
      if (q < nA) {
        const int v = a0 + q, j = v >> 9, rb = v & 511;
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
    __threadfence();
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
"""
VMS_CPP_SRC = r"""
#include <torch/extension.h>
void run(torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor wA, torch::Tensor wB, torch::Tensor gsA,
         torch::Tensor gsB, torch::Tensor nw, torch::Tensor inter, torch::Tensor ybuf, torch::Tensor ss, torch::Tensor ctr,
         torch::Tensor out, torch::Tensor marks, int64_t m, double eps, int64_t pdl, int64_t lo);
std::vector<int64_t> attrs();
"""

VMS_ENABLE = True
VMS_MAXT = 16
VMS_UB = 36864
VMS_POS = (((6, 7, 8, 15), (22, 23, 24, 31)), ((2, 3, 4, 11), (18, 19, 20, 27)),
           ((28, 29, 30, 5), (12, 13, 14, 21)), ((0, 1, 26, 9), (16, 17, 10, 25)))
_VMS = {"mod": None, "G": None}


def _vms_build():
    from torch.utils.cpp_extension import load_inline
    G = torch.cuda.get_device_properties(0).multi_processor_count
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-Xptxas=-warn-spills", "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t1 = time.time()
    mod = load_inline(name=f"k3_vms_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags)
    _VMS["G"] = G
    _VMS["mod"] = mod
    _K3_BUILD_S["vms"] = time.time() - t1
    return mod


def _vms_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << VMS_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _vms_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


def _vms_call(st, x, local_ids, topk_weights):
    vs = st["vms"]
    ids = local_ids.to(torch.int32).contiguous()
    tw = topk_weights.to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
    vs["mod"].run(x, ids, tw, vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"],
                  out, vs["nomarks"], m, K2A_POST_NORM_EPS, 1, 0)
    return out


def _vms_prepare(st, w, dev):
    st["vms_ok"] = False
    if not VMS_ENABLE:
        return
    t1 = time.time()
    try:
        _pt1_wait("vms")
        mod = _VMS["mod"] if _VMS["mod"] is not None else _vms_build()
        if _VMS["G"] != torch.cuda.get_device_properties(0).multi_processor_count:
            print("[vms] built for a different SM count: VMS OFF (K2A at every T)", flush=True)
            return
        b13, b2 = w["w13_sf"].view(torch.uint8), w["w2_sf"].view(torch.uint8)
        if not (bool(((b13 >= 8) & (b13 <= 0x7E)).all()) and bool(((b2 >= 8) & (b2 <= 0x7E)).all())):
            print("[vms] block-scale bytes outside 0x08..0x7E (PRMT scale decode inexact): VMS OFF (K2A at every T)", flush=True)
            return
        E = w["w13"].shape[0]
        if E != K2A_E_LOC or tuple(w["w13"].shape[1:]) != (2 * K2A_D_EXPERT, K2A_D_HIDDEN // 2):
            print("[vms] expert shapes differ: VMS OFF", flush=True)
            return
        wA = torch.empty((E, 512, VMS_UB), dtype=torch.uint8, device=dev)
        wB = torch.empty((E, 256, VMS_UB), dtype=torch.uint8, device=dev)
        for e in range(E):
            wA[e] = _vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            wB[e] = _vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        two62 = float(2 ** 62)
        st["vms"] = {"mod": mod, "wA": wA, "wB": wB, "gsA": (w["w13_gs"].float().reshape(E) * two62).contiguous(),
                     "gsB": (w["w2_gs"].float().reshape(E) * two62).contiguous(),
                     "nw": (1.0 + w["post_norm_w"].float()).contiguous(),
                     "inter": torch.zeros((128, K2A_D_EXPERT), dtype=torch.bfloat16, device=dev),
                     "ybuf": torch.zeros((128, K2A_D_HIDDEN), dtype=torch.float32, device=dev),
                     "ss": torch.zeros((256,), dtype=torch.float32, device=dev),
                     "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
                     "nomarks": torch.empty((0,), dtype=torch.int64, device=dev)}
        good = True
        for bs, seed, dist in VMS_SELF_TEST_CASES:
            if bs > VMS_MAXT:
                continue
            if dist == "hot8":
                d = _k2a36_synth_inputs(bs, seed, dev, "hot8")
                ids = d["topk_ids"]
                d["topk_ids"] = torch.where(ids < K2A_E_LOC, ids, torch.full_like(ids, -1))
            else:
                d = _k2a_self_test_inputs(bs, seed, dev, dist)
            x = d["hidden_states"].contiguous()
            ref = _k2a_ref_fp32(d, w)
            st["err"].zero_()
            ok_k = _v11_k2a33_call(x, d["topk_ids"], d["topk_weights"], st).clone()
            torch.cuda.synchronize()
            for k in range(2):
                o = _vms_call(st, x, d["topk_ids"], d["topk_weights"])
                torch.cuda.synchronize()
                e2, _ = _k2a_rel_err(o, ok_k)
                tot, row = _k2a_rel_err(o, ref)
                errf = int(st["vms"]["ctr"][2])
                ok = e2 < 8e-3 and tot < 1.0e-2 and row < 5.0e-2 and not errf and bool(torch.isfinite(o.float()).all())
                if not ok or k == 0:
                    print(f"[vms] self-test bs={bs} {dist} call {k}: rel vs K2A {e2:.2e} vs FP32 {tot:.3e} worst row {row:.3e} "
                          f"err={errf} -> {'ok' if ok else 'FAIL'}", flush=True)
                good &= ok
            _k2a33_reset(st)
        st["vms_ok"] = good
        print(f"[vms] VMS {'LIVE at T <= ' + str(VMS_MAXT) if good else 'OFF: self-test failed, K2A at every T'} "
              f"(regs/local/smem/G/NS {list(mod.attrs())}, {time.time() - t1:.1f}s)", flush=True)
    except Exception as ex:
        st["vms_ok"] = False
        print(f"[vms] prepare raised {ex!r}: VMS OFF (K2A at every T)", flush=True)
        try:
            _k2a33_reset(st)
        except Exception:
            pass


VMS_SELF_TEST_CASES = ((1, 41, "normal"), (1, 42, "skewed"), (3, 43, "normal"), (5, 44, "normal"), (8, 45, "normal"),
                       (8, 46, "skewed"), (12, 47, "skewed"), (16, 48, "normal"), (16, 49, "hot8"), (16, 50, "skewed"))


_K3_BUILD_S = {}


def _pt1_wait(name):
    return None


def kernel_fn(data, state):
    st = state["st"]
    x = data["hidden_states"]
    m = x.shape[0]
    if st is not None and st.get("vms_ok") and m <= VMS_MAXT:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
        vs = st["vms"]
        out = torch.empty((m, K2A_D_HIDDEN), dtype=torch.bfloat16, device=x.device)
        vs["mod"].run(x, data["topk_ids"].to(torch.int32).contiguous(), data["topk_weights"].to(torch.float32).contiguous(),
                      vs["wA"], vs["wB"], vs["gsA"], vs["gsB"], vs["nw"], vs["inter"], vs["ybuf"], vs["ss"], vs["ctr"], out,
                      vs["nomarks"], m, K2A_POST_NORM_EPS, 0, int(data["config"]["local_start"]))
        return out
    return _sa_kernel_fn(data, state)


def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        _vms_prepare(state["st"], state["w"], weights["w13"].device)
        torch.cuda.synchronize()
    return state

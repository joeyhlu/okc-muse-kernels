# okc: problem=muse_moe_w4a16
# study-a: the shipped K2A MoE of k3_v35_pt1.py, extracted verbatim by tools/sa_moe_extract.py (AST closure of the layer's K2A prepare / call / self-tests) + a board wrapper; check only, never scored.
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


def kernel_fn(data, state):
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


import math  # CR lab
import statistics  # CR lab

CR_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace crk {
#ifndef CR_NST
#define CR_NST 4
#endif
#ifndef CR_NGMAX
#define CR_NGMAX 8
#endif
constexpr int NST = CR_NST, NGMAX = CR_NGMAX;
#ifndef CR_NCW
#define CR_NCW 16
#endif
constexpr int NCW = CR_NCW;                // 16: 1 CTA/SM, 4 consumer WGs + 1 producer WG (setmaxnreg); 8: 2 CTAs/SM, 8 + 1 warps
constexpr int NTH = 32 * (NCW + (NCW == 16 ? 4 : 1));
constexpr int MINB = NCW == 16 ? 1 : 2;
constexpr int HB = 16 / NCW;               // CR blocks per 512-row packed block
#ifndef CR_PREG
#define CR_PREG 24
#endif
#ifndef CR_CREG
#define CR_CREG 112
#endif
constexpr int CB = NCW * 2048;             // codes per slot: NCW warps x 4 parts x 32 lanes x 16 B
constexpr int WB = NCW * 2304;             // + scales NCW x 32 lanes x 8 B
constexpr int XRB = 272;                   // X row bytes per stage: 128 bf16 + 16 B pad (conflict-free B loads)
constexpr int XB = 8 * NGMAX * XRB;
constexpr int SLOT = WB + XB;
constexpr int OFF_BAR = NST * SLOT;
constexpr int SMEM = OFF_BAR + 2 * NST * 8;
constexpr int PMAX = 48;
constexpr int PART = 512 * 64;             // floats per scratch slot (16 warps x 2 tiles x 8 groups x 4 x 32 lanes)
constexpr long long WAIT_BUDGET = 1ll << 31;
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 1;
#else
constexpr int FULLCNT = 32;
#endif

struct Args {
  const uint8_t* ws;        // [32][16][32][WB]
  const uint8_t* xp;        // packed X
  const int* units;         // [nu][8]: e, b, ntok, ng, rbase, xoff, multipass, 0
  const int* cuts;          // [G + 1] global stage cut points
  const float* gsc;         // [32] w13_gs * 256
  __nv_bfloat16* inter;     // [R][4096]
  float* scratch;           // [G][2][PART]
  int* cnt;                 // [nu] self-resetting stage counters
  int* err; long long* marks;
  int G, mode;
  // full-call fields: phase 0 = W13 (+ SwiGLU, inter into xp2 in W2's X layout, zero ss), 1 = W2 (+ post-norm partials)
  int phase, nbe, R;
  uint8_t* xp2; float* ss; __nv_bfloat16* ybuf; const float* nw;
};

__device__ __forceinline__ void deq4(uint32_t w, uint32_t* o) {   // BITPOS3 words: o[j] = codes (2j, 2j + 1)
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
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ long long gtime() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
  return t;
}
__device__ __forceinline__ void mbar_init(uint32_t b, int n) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(b), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t b) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}\n" :: "r"(b) : "memory");
}
__device__ __forceinline__ bool mbar_try(uint32_t b, uint32_t par) {
  uint32_t ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(par) : "memory");
#endif
  return ok != 0;
}
__device__ __forceinline__ void set_err(int* err, int code) {
  if (atomicExch(err, 1) == 0) atomicExch(err + 1, code);
}
__device__ __forceinline__ bool mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return true;
  const long long t0 = clock64();
  while (!mbar_try(b, par)) {
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return false; }
  }
  return true;
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void bulk_g2s_h(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint64_t pol) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
#else
__device__ __forceinline__ void cpa16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpa_arrive(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
}
__device__ __forceinline__ void warp_copy(int lane, uint32_t dst, const void* src, uint32_t bytes) {
  const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
  for (uint32_t o = 16u * lane; o < bytes; o += 512u) cpa16(dst + o, s + o);
}
#endif
__device__ __forceinline__ float silu_mul(float g, float u) { return g / (1.f + __expf(-g)) * u; }
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }
__device__ __forceinline__ void pdl_launch() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

// One piece's stage loop, NG token groups (8 tokens each) compiled in: in-place accumulation, no predicated mma.
template <int NG>
__device__ __forceinline__ void consume_piece(const uint8_t* smem, uint32_t full_a, uint32_t empty_a, int* err, int s0,
                                              int s1, int& it, bool& dead, bool null_c, int warp, int lane,
                                              long long* mk, int tid, float (&acc)[2][NGMAX][4]) {
  const int g = lane >> 2, t = lane & 3;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    if (!dead && !mbar_wait(full_a + 8 * slot, (it / NST) & 1, err, 1)) dead = true;
    if (it == 0 && mk && tid == 0) mk[2] = gtime();
    if (!null_c) {
      const uint8_t* sb = smem + slot * SLOT;
      const uint4* cw = reinterpret_cast<const uint4*>(sb) + warp * 128 + lane;
      const uint2 sc = reinterpret_cast<const uint2*>(sb + CB)[warp * 32 + lane];
      const uint8_t* xs = sb + WB + g * XRB + t * 32;
#pragma unroll
      for (int kbo = 0; kbo < 2; ++kbo) {
        const uint4 q0 = cw[32 * kbo], q1 = cw[32 * (2 + kbo)];
        const uint32_t Sg0 = __byte_perm(sc.x, 0u, 0x4040u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t S80 = __byte_perm(sc.x, 0u, 0x4141u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t Sg1 = __byte_perm(sc.y, 0u, 0x4040u + 0x0202u * kbo) * 16u + 0x75807580u;
        const uint32_t S81 = __byte_perm(sc.y, 0u, 0x4141u + 0x0202u * kbo) * 16u + 0x75807580u;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          uint32_t a0g[4], a08[4], a1g[4], a18[4];
          deq4(h ? q0.y : q0.x, a0g); deq4(h ? q0.w : q0.z, a08);
          deq4(h ? q1.y : q1.x, a1g); deq4(h ? q1.w : q1.z, a18);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            a0g[j] = bmul2(a0g[j], Sg0); a08[j] = bmul2(a08[j], S80);
            a1g[j] = bmul2(a1g[j], Sg1); a18[j] = bmul2(a18[j], S81);
          }
#pragma unroll
          for (int gi = 0; gi < NG; ++gi) {
            const uint4 xv = *reinterpret_cast<const uint4*>(xs + gi * 8 * XRB + kbo * 128 + h * 16);
            mma16816(acc[0][gi], a0g[0], a08[0], a0g[1], a08[1], xv.x, xv.y);
            mma16816(acc[1][gi], a1g[0], a18[0], a1g[1], a18[1], xv.x, xv.y);
            mma16816(acc[0][gi], a0g[2], a08[2], a0g[3], a08[3], xv.z, xv.w);
            mma16816(acc[1][gi], a1g[2], a18[2], a1g[3], a18[3], xv.z, xv.w);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
}

__global__ void __launch_bounds__(NTH, MINB) cr_k1(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_pu[PMAX], s_ps0[PMAX], s_ps1[PMAX], s_pslot[PMAX];
  __shared__ int s_np, s_last;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int c = blockIdx.x;
  long long* mk = A.marks ? A.marks + (size_t)c * 8 : nullptr;
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + OFF_BAR);
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);
  if (tid == 0) {
    if (mk) mk[0] = gtime();
    if (A.phase == 0) pdl_launch();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, NCW); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    // piece list: partial pieces first (slot 0 = the piece holding cuts[c], slot 1 = the one holding cuts[c+1]-1)
    const int a = A.cuts[c], bnd = A.cuts[c + 1];
    int np = 0;
    if (bnd > a) {
      const int ua = a >> 5, ub = (bnd - 1) >> 5;
      if (ua == ub) {
        const int s0 = a & 31, s1 = bnd - 32 * ua;
        s_pu[0] = ua; s_ps0[0] = s0; s_ps1[0] = s1; s_pslot[0] = (s0 == 0 && s1 == 32) ? -1 : 0; np = 1;
      } else {
        const bool hp = (a & 31) != 0, tp = (bnd - 32 * ub) != 32;
        if (hp) { s_pu[np] = ua; s_ps0[np] = a & 31; s_ps1[np] = 32; s_pslot[np] = 0; ++np; }
        if (tp) { s_pu[np] = ub; s_ps0[np] = 0; s_ps1[np] = bnd - 32 * ub; s_pslot[np] = 1; ++np; }
        for (int u = hp ? ua + 1 : ua; u <= (tp ? ub - 1 : ub); ++u) {
          if (np >= PMAX) { set_err(A.err, 9); break; }
          s_pu[np] = u; s_ps0[np] = 0; s_ps1[np] = 32; s_pslot[np] = -1; ++np;
        }
      }
    }
    s_np = np;
  }
  __syncthreads();
  const int np = s_np;
  if (A.phase == 1) pdl_wait();                      // K1's inter (xp2) and zeroed ss are visible after this
  else if (A.ss) for (int i = c * NTH + tid; i < A.R; i += A.G * NTH) A.ss[i] = 0.f;

  if (warp >= NCW) {
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(CR_PREG) : "memory");
#endif
    if (warp != NCW) return;
    // ---------------- producer ----------------
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0;
    bool dead = false;                      // after one bounded-wait timeout: stop waiting (err is set), never hang
    for (int p = 0; p < np; ++p) {
      const int* U = A.units + 8 * s_pu[p];
      const int e = U[0], b = U[1], ng = U[3], xoff = U[5], multi = U[6];
      const uint32_t xbytes = (uint32_t)(8 * ng * XRB);
      for (int s = s_ps0[p]; s < s_ps1[p]; ++s, ++it) {
        const int slot = it % NST;
        if (it >= NST && !dead && !mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, A.err, 2)) dead = true;
        const uint8_t* wsrc = A.ws + ((size_t)((e * A.nbe + b / HB) * 32 + s)) * 36864 + (b % HB) * CB;
        const uint8_t* ssrc = A.ws + ((size_t)((e * A.nbe + b / HB) * 32 + s)) * 36864 + 32768 + (b % HB) * (NCW * 256);
        const uint8_t* xsrc = A.xp + (size_t)xoff + (size_t)s * xbytes;
        const uint32_t dst = ring_a + slot * SLOT;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          mbar_expect_tx(full_a + 8 * slot, WB + xbytes);
          if (NCW == 16) {
            if (multi) bulk_g2s(dst, wsrc, WB, full_a + 8 * slot);
            else bulk_g2s_h(dst, wsrc, WB, full_a + 8 * slot, pol);
          } else {
            if (multi) { bulk_g2s(dst, wsrc, CB, full_a + 8 * slot); bulk_g2s(dst + CB, ssrc, WB - CB, full_a + 8 * slot); }
            else { bulk_g2s_h(dst, wsrc, CB, full_a + 8 * slot, pol); bulk_g2s_h(dst + CB, ssrc, WB - CB, full_a + 8 * slot, pol); }
          }
          bulk_g2s(dst + WB, xsrc, xbytes, full_a + 8 * slot);
        }
#else
        warp_copy(lane, dst, wsrc, CB);
        warp_copy(lane, dst + CB, ssrc, WB - CB);
        warp_copy(lane, dst + WB, xsrc, xbytes);
        cpa_arrive(full_a + 8 * slot);
#endif
        if (it == 0 && mk && lane == 0) mk[1] = gtime();
      }
    }
    if (mk && lane == 0) mk[4] = gtime();
    return;
  }

  // ---------------- consumers ----------------
#if __CUDA_ARCH__ >= 900 && CR_NCW == 16
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(CR_CREG) : "memory");
#endif
  const int g = lane >> 2, t = lane & 3;
  const bool null_c = (A.mode & 1) != 0;
  int it = 0;
  bool dead = false;
  float acc[2][NGMAX][4];
  for (int p = 0; p < np; ++p) {
    const int u = s_pu[p];
    const int* U = A.units + 8 * u;
    const int e = U[0], b = U[1], ntok = U[2], ng = U[3], rbase = U[4], xoff = U[5];
#pragma unroll
    for (int ti = 0; ti < 2; ++ti)
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[ti][gi][i] = 0.f;
    {
      const int s0 = s_ps0[p], s1 = s_ps1[p];
#define CR_CP(N) consume_piece<N>(smem, full_a, empty_a, A.err, s0, s1, it, dead, null_c, warp, lane, mk, tid, acc)
      switch (ng) {
        case 1: CR_CP(1); break;
        case 2: CR_CP(2); break;
        case 3: CR_CP(3); break;
        case 4: CR_CP(4); break;
        case 5: CR_CP(5); break;
        case 6: CR_CP(6); break;
#if CR_NGMAX >= 7
        case 7: CR_CP(7); break;
#endif
#if CR_NGMAX >= 8
        default: CR_CP(8); break;
#else
        default: CR_CP(6); break;
#endif
      }
#undef CR_CP
    }
    if (null_c) continue;
    // ---------------- piece epilogue ----------------
    const int pslot = s_pslot[p];
    const float gs = A.gsc[e];
    const int d0 = 16 * NCW * b + 16 * warp;
    bool fin = pslot < 0;
    if (!fin) {
      float* my = A.scratch + ((size_t)(c * 2 + pslot)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
      for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int gi = 0; gi < NGMAX; ++gi)
          if (gi < ng) {
#pragma unroll
            for (int i = 0; i < 4; ++i) my[((ti * NGMAX + gi) * 4 + i) * 32 + lane] = acc[ti][gi][i];
          }
      bar_sync(1, 32 * NCW);
      if (tid == 0) {
        __threadfence();
        const int n = s_ps1[p] - s_ps0[p];
        const int old = atomicAdd(A.cnt + u, n);
        s_last = (old + n == 32);
        if (old + n > 32) set_err(A.err, 5);
        __threadfence();
      }
      bar_sync(1, 32 * NCW);
      fin = s_last != 0;
      if (fin) {
        // add the other contributors' partials (CTAs whose ranges intersect unit u)
        const int lo = 32 * u, hi = 32 * u + 32;
        int c0 = 0, c1 = A.G;                       // first c' with cuts[c' + 1] > lo
        while (c0 < c1) { const int m = (c0 + c1) >> 1; if (A.cuts[m + 1] > lo) c1 = m; else c0 = m + 1; }
        for (int cc = c0; cc < A.G && A.cuts[cc] < hi; ++cc) {
          if (cc == c || A.cuts[cc + 1] <= A.cuts[cc]) continue;
          const int os = ((A.cuts[cc] >> 5) == u) ? 0 : 1;
          const float* o = A.scratch + ((size_t)(cc * 2 + os)) * PART + (size_t)warp * (2 * NGMAX * 4 * 32);
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int gi = 0; gi < NGMAX; ++gi)
              if (gi < ng) {
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[ti][gi][i] += __ldcg(o + ((ti * NGMAX + gi) * 4 + i) * 32 + lane);
              }
        }
        if (tid == 0) A.cnt[u] = 0;
      }
    }
    if (fin && A.phase == 0) {
      __nv_bfloat16* x2 = A.xp2 ? reinterpret_cast<__nv_bfloat16*>(A.xp2 + xoff) : nullptr;
      const int sstr = 8 * ng * (XRB / 2);          // elements per K-stage of this pass in W2's X layout
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int tok = 8 * gi + 2 * t + (i & 1);
            const int dim = d0 + g + 8 * (i >> 1);
            if (tok < ntok) {
              const __nv_bfloat16 v = __float2bfloat16_rn(silu_mul(acc[0][gi][i] * gs, acc[1][gi][i] * gs));
              if (A.inter) A.inter[(size_t)(rbase + tok) * 4096 + dim] = v;
              if (x2) x2[(size_t)(dim >> 7) * sstr + tok * (XRB / 2) + (dim & 127)] = v;
            }
          }
        }
    }
    if (fin && A.phase == 1) {
      const int o0 = 32 * NCW * b + 32 * warp;
#pragma unroll
      for (int gi = 0; gi < NGMAX; ++gi)
        if (gi < ng) {
          float q0 = 0.f, q1 = 0.f;
#pragma unroll
          for (int ti = 0; ti < 2; ++ti)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int tok = 8 * gi + 2 * t + (i & 1);
              const int row = o0 + 16 * ti + g + 8 * (i >> 1);
              const float yb = bf16r(bf16r(acc[ti][gi][i] * gs) * A.nw[row]);
              if (i & 1) q1 += yb * yb; else q0 += yb * yb;
              if (tok < ntok) A.ybuf[(size_t)(rbase + tok) * 4096 + row] = __float2bfloat16_rn(yb);
            }
#pragma unroll
          for (int m = 4; m < 32; m <<= 1) {
            q0 += __shfl_xor_sync(0xffffffffu, q0, m);
            q1 += __shfl_xor_sync(0xffffffffu, q1, m);
          }
          const int tk = 8 * gi + 2 * t;
          if (g == 0) {
            if (tk < ntok) atomicAdd(A.ss + rbase + tk, q0);
            if (tk + 1 < ntok) atomicAdd(A.ss + rbase + tk + 1, q1);
          }
        }
    }
  }
  if (mk && tid == 0) mk[3] = gtime();
}
// K3: out[t] = sum over t's local routes of w * rsqrt(mean(yb^2) + eps) * yb (yb = bf16(bf16(y) * (1 + w_post)))
__global__ void __launch_bounds__(256) cr_fin(const __nv_bfloat16* ybuf, const float* ss, const int* tokr,
                                              const float* tokw, __nv_bfloat16* out) {
  pdl_wait();
  const int tt = blockIdx.x, d = blockIdx.y * 1024 + threadIdx.x * 4;
  float o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < 8; ++k) {
    const int r = tokr[tt * 8 + k];
    if (r < 0) continue;
    const float sc = tokw[tt * 8 + k] * rsqrtf(ss[r] * (1.f / 4096.f) + 1e-8f);
    const uint2 v = *reinterpret_cast<const uint2*>(ybuf + (size_t)r * 4096 + d);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&v.x), b2 = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
    o[0] += sc * __low2float(a); o[1] += sc * __high2float(a); o[2] += sc * __low2float(b2); o[3] += sc * __high2float(b2);
  }
  __nv_bfloat162 r0 = __floats2bfloat162_rn(o[0], o[1]), r1 = __floats2bfloat162_rn(o[2], o[3]);
  uint2 w;
  w.x = *reinterpret_cast<uint32_t*>(&r0); w.y = *reinterpret_cast<uint32_t*>(&r1);
  *reinterpret_cast<uint2*>(out + (size_t)tt * 4096 + d) = w;
}
}  // namespace crk

template <typename K, typename... Args>
static void cr_launch(K kern, dim3 grid, dim3 block, size_t smem, bool pdl, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at; cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, args...));
}

void cr_full(torch::Tensor x1, torch::Tensor ws13, torch::Tensor units1, torch::Tensor cuts1, torch::Tensor gsc13,
             torch::Tensor xp2, torch::Tensor ws2, torch::Tensor units2, torch::Tensor cuts2, torch::Tensor gsc2,
             torch::Tensor nw, torch::Tensor ss, torch::Tensor ybuf, torch::Tensor tokr, torch::Tensor tokw,
             torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, int64_t G, int64_t R,
             int64_t pdl) {
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
    attr = true;
  }
  crk::Args A = {};
  A.ws = ws13.data_ptr<uint8_t>(); A.xp = x1.data_ptr<uint8_t>(); A.units = units1.data_ptr<int>();
  A.cuts = cuts1.data_ptr<int>(); A.gsc = gsc13.data_ptr<float>(); A.inter = nullptr;
  A.scratch = scratch.data_ptr<float>(); A.cnt = cnt.data_ptr<int>(); A.err = err.data_ptr<int>(); A.marks = nullptr;
  A.G = (int)G; A.mode = 0; A.phase = 0; A.nbe = 16 / crk::HB * crk::HB == 16 ? 16 : 16; A.R = (int)R;
  A.xp2 = xp2.data_ptr<uint8_t>(); A.ss = ss.data_ptr<float>(); A.ybuf = nullptr; A.nw = nullptr;
  cr_launch(crk::cr_k1, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, false, A);
  crk::Args B = A;
  B.ws = ws2.data_ptr<uint8_t>(); B.xp = xp2.data_ptr<uint8_t>(); B.units = units2.data_ptr<int>();
  B.cuts = cuts2.data_ptr<int>(); B.gsc = gsc2.data_ptr<float>(); B.cnt = cnt.data_ptr<int>() + 4096;
  B.phase = 1; B.nbe = 8; B.xp2 = nullptr;
  B.ybuf = reinterpret_cast<__nv_bfloat16*>(ybuf.data_ptr<at::BFloat16>()); B.nw = nw.data_ptr<float>();
  cr_launch(crk::cr_k1, dim3((unsigned)G), dim3(crk::NTH), crk::SMEM, pdl != 0, B);
  const int M = (int)out.size(0);
  cr_launch(crk::cr_fin, dim3((unsigned)M, 4), dim3(256), 0, pdl != 0,
            (const __nv_bfloat16*)B.ybuf, (const float*)ss.data_ptr<float>(), (const int*)tokr.data_ptr<int>(),
            (const float*)tokw.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
}

void cr_run(torch::Tensor xp, torch::Tensor ws, torch::Tensor units, torch::Tensor cuts, torch::Tensor gsc,
            torch::Tensor inter, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks,
            int64_t G, int64_t mode) {
  crk::Args A;
  A.ws = ws.data_ptr<uint8_t>(); A.xp = xp.data_ptr<uint8_t>(); A.units = units.data_ptr<int>();
  A.cuts = cuts.data_ptr<int>(); A.gsc = gsc.data_ptr<float>();
  A.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr<at::BFloat16>());
  A.scratch = scratch.data_ptr<float>(); A.cnt = cnt.data_ptr<int>(); A.err = err.data_ptr<int>();
  A.marks = marks.numel() ? reinterpret_cast<long long*>(marks.data_ptr<int64_t>()) : nullptr;
  A.G = (int)G; A.mode = (int)mode;
  A.phase = 0; A.nbe = 16; A.R = 0; A.xp2 = nullptr; A.ss = nullptr; A.ybuf = nullptr; A.nw = nullptr;
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
    attr = true;
  }
  crk::cr_k1<<<(unsigned)G, crk::NTH, crk::SMEM, at::cuda::getCurrentCUDAStream()>>>(A);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> cr_attrs() {
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, crk::cr_k1));
  C10_CUDA_CHECK(cudaFuncSetAttribute(crk::cr_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, crk::SMEM));
  int occ = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, crk::cr_k1, crk::NTH, crk::SMEM));
  return {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)crk::SMEM, (int64_t)occ, (int64_t)crk::NGMAX,
          (int64_t)crk::NST, (int64_t)crk::NCW};
}
"""

CR_CPP = r"""
#include <torch/extension.h>
#include <vector>
void cr_run(torch::Tensor xp, torch::Tensor ws, torch::Tensor units, torch::Tensor cuts, torch::Tensor gsc,
            torch::Tensor inter, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, torch::Tensor marks,
            int64_t G, int64_t mode);
std::vector<int64_t> cr_attrs();
void cr_full(torch::Tensor x1, torch::Tensor ws13, torch::Tensor units1, torch::Tensor cuts1, torch::Tensor gsc13,
             torch::Tensor xp2, torch::Tensor ws2, torch::Tensor units2, torch::Tensor cuts2, torch::Tensor gsc2,
             torch::Tensor nw, torch::Tensor ss, torch::Tensor ybuf, torch::Tensor tokr, torch::Tensor tokw,
             torch::Tensor out, torch::Tensor scratch, torch::Tensor cnt, torch::Tensor err, int64_t G, int64_t R,
             int64_t pdl);
"""

CR_WB = 36864
CR_XRB = 272
CR_PART = 512 * 64


def cr_build(arch="sm_90a", ncw=16, ngmax=8, nst=4, verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = (["-gencode=arch=compute_90a,code=sm_90a", f"-DCR_NST={nst}"] if arch == "sm_90a"
           else ["-gencode=arch=compute_86,code=sm_86", "-DCR_NST=1"])
    gen = gen + [f"-DCR_NGMAX={ngmax}", f"-DCR_NCW={ncw}"]
    tag = hashlib.sha1((CR_CPP + CR_CUDA + arch + str((ncw, ngmax, nst))).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"cr_k1_{tag}", cpp_sources=CR_CPP, cuda_sources=CR_CUDA, functions=["cr_run", "cr_attrs", "cr_full"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills"] + gen,
                      verbose=verbose)
    return mod, time.time() - t0


# ---- generic NVFP4 decode math (BITPOS3 words + exact E4M3 re-encode; the same lossless math K2A's prepare uses) ----
CR_BITPOS3 = ((6, 7, 8, 15), (22, 23, 24, 31), (0, 1, 2, 9), (16, 17, 18, 25),
              (26, 27, 28, 3), (10, 11, 12, 19), (4, 5, 14, 13), (20, 21, 30, 29))
_CR_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def cr_dequant(packed, sf, gs):
    grid = torch.tensor(_CR_GRID, dtype=torch.float32, device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)
    return vals.view(*vals.shape[:-2], k)


def cr_reencode_scales(sf_u8):
    """E4M3 bytes -> (b', neg, zero, has_nan); b' = (e+10)<<3 | f so that bf16 bits b'*16 + 0x7580 = |sf| * 2^118."""
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


def cr_pack_words(codes, bitpos=CR_BITPOS3):
    c = codes.to(torch.int64)
    w = torch.zeros(c.shape[:-1], dtype=torch.int64, device=c.device)
    for i, (p0, p1, p2, ps) in enumerate(bitpos):
        ci = c[..., i]
        w |= ((ci & 1) << p0) | (((ci >> 1) & 1) << p1) | (((ci >> 2) & 1) << p2) | (((ci >> 3) & 1) << ps)
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w)
    return w.to(torch.int32)


def cr_prepack_rows(w, sf):
    """w uint8 [N, K/2], sf e4m3 [N, K/16] -> (wq int32 [N/16, K/64, 32, 4], sq uint8 [N/16, K/128, 32, 4]); natural row order.
    words (x, y) = row g at k = 64 kb + 16 t + (0..7, 8..15), (z, w) = row g + 8; sq bytes 2 par + hi of group 4 kb + t."""
    n, k2 = w.shape
    k = 2 * k2
    rt, nkb = n // 16, k // 64
    sfb = sf.view(torch.uint8) if sf.dtype != torch.uint8 else sf
    bp, neg, zero, bad = cr_reencode_scales(sfb)
    if bad:
        raise ValueError("NaN block scale in NVFP4 weights")
    codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(n, k)
    if bool(neg.any()) or bool(zero.any()):
        cb = codes.view(n, k // 16, 16)
        cb = torch.where(neg.unsqueeze(-1), cb ^ 8, cb)
        cb = torch.where(zero.unsqueeze(-1), torch.zeros_like(cb), cb)
        codes = cb.reshape(n, k)
    c = codes.view(rt, 2, 8, nkb, 4, 2, 8).permute(0, 3, 2, 4, 1, 5, 6).reshape(rt, nkb, 32, 4, 8)
    wq = cr_pack_words(c)
    sq = bp.view(rt, 2, 8, nkb // 2, 2, 4).permute(0, 3, 2, 5, 4, 1).reshape(rt, nkb // 2, 32, 4)
    return wq.contiguous(), sq.contiguous()


def cr_prepack(w13, w13_sf, gs13):
    """w13 uint8 [E, 8192, 2048], w13_sf e4m3 [E, 8192, 256] -> ws uint8 [E][16 b][32 s][36864] (CR slot layout),
    gsc float32 [E] (= gs * 256: the decoded weights carry 2^-8)."""
    E = w13.shape[0]
    ws = torch.empty((E, 16, 32, CR_WB), dtype=torch.uint8, device=w13.device)
    for e in range(E):
        wq, sq = cr_prepack_rows(w13[e], w13_sf[e])   # [512, 64, 32, 4] i32, [512, 32, 32, 4] u8
        wq = wq.view(2, 16, 16, 32, 2, 32, 4)            # (gate|up, b, w, s, kbo, lane, word)
        codes = wq.permute(1, 3, 2, 0, 4, 5, 6).contiguous().view(16, 32, 16 * 4 * 32 * 4)    # (b, s, w, tile, kbo, lane, word)
        sq32 = sq.contiguous().view(torch.int32).view(2, 16, 16, 32, 32)                      # (tile, b, w, s, lane)
        scl = sq32.permute(1, 3, 2, 4, 0).contiguous().view(16, 32, 16 * 32 * 2)              # (b, s, w, lane, tile)
        ws[e, :, :, :32768] = codes.view(torch.uint8).view(16, 32, 32768)
        ws[e, :, :, 32768:] = scl.view(torch.uint8).view(16, 32, 4096)
    return ws, (gs13.float() * 256.0).contiguous()


def cr_prepack_w2(w2, w2_sf, gs2):
    """w2 uint8 [E, 4096, 2048] -> ws2 uint8 [E][8 b][32 s][36864]: warp w of block b owns output rows 512 b + 32 w .. + 31
    as two m16 tiles (K = the 4096 intermediate dims, same K permutation and scale words as W13)."""
    E = w2.shape[0]
    ws = torch.empty((E, 8, 32, CR_WB), dtype=torch.uint8, device=w2.device)
    for e in range(E):
        wq, sq = cr_prepack_rows(w2[e], w2_sf[e])   # [256, 64, 32, 4], [256, 32, 32, 4]
        codes = wq.view(8, 16, 2, 32, 2, 32, 4).permute(0, 3, 1, 2, 4, 5, 6).contiguous().view(8, 32, 16 * 4 * 32 * 4)
        sq32 = sq.contiguous().view(torch.int32).view(8, 16, 2, 32, 32)
        scl = sq32.permute(0, 3, 1, 4, 2).contiguous().view(8, 32, 16 * 32 * 2)
        ws[e, :, :, :32768] = codes.view(torch.uint8).view(8, 32, 32768)
        ws[e, :, :, 32768:] = scl.view(torch.uint8).view(8, 32, 4096)
    return ws, (gs2.float() * 256.0).contiguous()


def cr_plan_w2(d, P1, G, kappa=0.22, nb2=8):
    """W2 units from the W13 plan's (expert, pass) table (same token passes, X = K1's packed inter), equal-cost cuts,
    and the finalize's per-token route table."""
    dev = P1["units"].device
    u1 = P1["units"].cpu().tolist()
    seen, order = set(), []
    for e, b, nt, ng, rb, xo, multi, _ in u1:
        if (e, rb) not in seen:
            seen.add((e, rb))
            order.append((e, nt, ng, rb, xo, multi))
    units = []
    for e in dict.fromkeys(o[0] for o in order):
        ps = [o for o in order if o[0] == e]
        for b in range(nb2):
            for (_, nt, ng, rb, xo, multi) in ps:
                units.append([e, b, nt, ng, rb, xo, multi, 0])
    cost = [max(1.0, u[3] * kappa) for u in units]
    tot = 32.0 * sum(cost)
    cuts, cum, ui = [0], 0.0, 0
    for i in range(1, G):
        target = tot * i / G
        while ui < len(units) and cum + 32 * cost[ui] <= target:
            cum += 32 * cost[ui]
            ui += 1
        s_ = 0 if ui >= len(units) else min(32, int((target - cum) / cost[ui] + 0.5))
        cuts.append(max(cuts[-1], 32 * ui + s_))
    cuts.append(32 * len(units))
    rmap = {te: r for r, te in enumerate(P1["route_tok"])}
    ids = d["topk_ids"].long().cpu()
    wts = d["topk_weights"].float().cpu()
    M = ids.shape[0]
    tokr = torch.full((M, 8), -1, dtype=torch.int32)
    tokw = torch.zeros((M, 8), dtype=torch.float32)
    for tt in range(M):
        for k in range(8):
            e = int(ids[tt, k])
            if e < 32:
                tokr[tt, k] = rmap[(tt, e)]
                tokw[tt, k] = wts[tt, k]
    return {"units": torch.tensor(units, dtype=torch.int32, device=dev).reshape(-1, 8),
            "cuts": torch.tensor(cuts, dtype=torch.int32, device=dev), "tokr": tokr.to(dev), "tokw": tokw.to(dev)}


def cr_ref_full(d, w, dequant):
    """FP32 reference of the whole MoE (the problem's math; post-norm on bf16(y * (1 + w)))."""
    x = d["hidden_states"].float()
    ids = d["topk_ids"].long()
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not 0 <= e < 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y.to(torch.bfloat16).float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, tok, y * d["topk_weights"][tok, slot].float().unsqueeze(1))
    return out


class CrFull:
    """CR full call: K1 (W13 + SwiGLU -> packed inter) -> PDL -> K2 (W2 + post-norm partials) -> PDL -> K3 finalize.
    Plans / X pack are host-built (lab only)."""

    def __init__(self, st, ws2, gsc2, nw, pdl=True):
        self.st, self.ws2, self.gsc2, self.nw, self.pdl = st, ws2, gsc2, nw, pdl

    def prep(self, d, P1, P2):
        dev = self.ws2.device
        R = max(P1["R"], 1)
        bufs = {"xp2": torch.zeros_like(P1["xp"]), "ss": torch.zeros((R,), dtype=torch.float32, device=dev),
                "ybuf": torch.zeros((R, 4096), dtype=torch.bfloat16, device=dev),
                "out": torch.empty((d["hidden_states"].shape[0], 4096), dtype=torch.bfloat16, device=dev)}
        return bufs

    def fn(self, P1, P2, B):
        st = self.st

        def f():
            st.mod.cr_full(P1["xp"], st.ws, P1["units"], P1["cuts"], st.gsc, B["xp2"], self.ws2, P2["units"], P2["cuts"],
                           self.gsc2, self.nw, B["ss"], B["ybuf"], P2["tokr"], P2["tokw"], B["out"], st.scratch, st.cnt,
                           st.err, st.G, max(P1["R"], 1), int(self.pdl))
        return f


def cr_check_full(full, d, w, P1, P2, dequant):
    B = full.prep(d, P1, P2)
    full.st.err.zero_()
    full.fn(P1, P2, B)()
    torch.cuda.synchronize()
    ref = cr_ref_full(d, w, dequant)
    got = B["out"].float()
    diff = got - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-30))
    rn = ref.norm(dim=1)
    worst = float((diff.norm(dim=1) / torch.maximum(rn, 0.1 * rn.mean()).clamp(min=1e-30)).max())
    return rel, worst, [int(v) for v in full.st.err.tolist()], int(full.st.cnt.abs().sum()), bool(torch.isnan(got).any()), B


def cr_case(bs, seed, dev):
    """The board's normal / ep8 / busiest routing law (our own copy; seeds disjoint from the judge's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    z = torch.randn((256,), generator=g, device=dev)
    u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


# decode_layer routings (tools/box/m32_run.py, box_out/m32/run.json: 5 router seeds x 2 inputs): local expert token counts
CR_LAYER_COUNTS = {
    32: ([26, 11, 7, 6, 3, 3, 2, 1, 1, 1], [24, 16, 10, 6, 4, 3, 3], [22, 19, 11, 4, 3, 3, 3, 1, 1, 1], [23, 13, 11, 10]),
    16: ([14, 7, 5, 5, 3, 2, 1], [14, 12, 4, 2, 1, 1, 1, 1], [9, 4, 4, 4, 3, 1, 1, 1, 1, 1, 1]),
}


def cr_case_counts(bs, counts, seed, dev):
    """A routing with the given local-expert token counts (layer-shaped): random local experts, random distinct tokens per
    expert (<= 8 slots per token), the remaining slots filled with distinct remote experts (>= 32)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((bs, 4096), generator=torch.Generator(device=dev).manual_seed(seed), device=dev).to(torch.bfloat16)
    experts = torch.randperm(32, generator=g)[:len(counts)].tolist()
    slots = [[] for _ in range(bs)]
    for e, cn in zip(experts, counts):
        order = torch.randperm(bs, generator=g).tolist()
        order.sort(key=lambda tt: len(slots[tt]))
        free = [tt for tt in order if len(slots[tt]) < 8][:cn]
        assert len(free) == cn, "counts exceed 8 slots per token"
        for tt in free:
            slots[tt].append(e)
    ids = torch.empty((bs, 8), dtype=torch.int32)
    for tt in range(bs):
        rem = (32 + torch.randperm(224, generator=g)[:8 - len(slots[tt])]).tolist()
        row = slots[tt] + rem
        perm = torch.randperm(8, generator=g).tolist()
        ids[tt] = torch.tensor([row[i] for i in perm], dtype=torch.int32)
    wts = torch.rand((bs, 8), generator=g) + 0.05
    wts = wts / wts.sum(-1, keepdim=True)
    return {"hidden_states": x, "topk_ids": ids.to(dev).contiguous(), "topk_weights": wts.float().to(dev).contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def cr_plan(d, G, kappa=0.22, tpass=64, nb=16):
    """Host plan (lab only): units in hot-first expert order, (expert, block, pass); contiguous equal-cost stage ranges."""
    ids = d["topk_ids"].long().cpu()
    x = d["hidden_states"]
    dev = x.device
    M = ids.shape[0]
    toks = {e: [] for e in range(32)}
    for tt in range(M):
        for k in range(8):
            e = int(ids[tt, k])
            if e < 32:
                toks[e].append(tt)
    experts = sorted((e for e in range(32) if toks[e]), key=lambda e: (-len(toks[e]), e))
    passes, units, route_tok, xparts = {}, [], [], []
    rbase = xoff = 0
    for e in experts:
        T = toks[e]
        npass = math.ceil(len(T) / tpass)
        for p in range(npass):
            tl = T[tpass * p:tpass * (p + 1)]
            ng = math.ceil(len(tl) / 8)
            passes[(e, p)] = (len(tl), ng, rbase, xoff, npass > 1)
            xparts.append((tl, 8 * ng))
            route_tok.extend((tt, e) for tt in tl)
            rbase += len(tl)
            xoff += 32 * 8 * ng * CR_XRB
    for e in experts:
        npass = math.ceil(len(toks[e]) / tpass)
        for b in range(nb):
            for p in range(npass):
                nt, ng, rb, xo, multi = passes[(e, p)]
                units.append([e, b, nt, ng, rb, xo, int(multi), 0])
    cost = [max(1.0, u[3] * kappa) for u in units]
    tot = 32.0 * sum(cost)
    cuts, cum, ui = [0], 0.0, 0
    for i in range(1, G):
        target = tot * i / G
        while ui < len(units) and cum + 32 * cost[ui] <= target:
            cum += 32 * cost[ui]
            ui += 1
        s = 0 if ui >= len(units) else min(32, int((target - cum) / cost[ui] + 0.5))
        cuts.append(max(cuts[-1], 32 * ui + s))
    cuts.append(32 * len(units))
    # X pack: per (expert, pass) [32 stages][tpad][136] bf16 (128 K + 8 pad), zero-padded tokens
    chunks = []
    for tl, tpad in xparts:
        X = torch.zeros((tpad, 4096), dtype=torch.bfloat16, device=dev)
        X[:len(tl)] = x[torch.tensor(tl, dtype=torch.long, device=dev)]
        X = torch.nn.functional.pad(X.view(tpad, 32, 128).permute(1, 0, 2), (0, 8)).contiguous()
        chunks.append(X.view(torch.uint8).reshape(-1))
    xp = torch.cat(chunks) if chunks else torch.zeros((16,), dtype=torch.uint8, device=dev)
    w13_bytes = len(experts) * 16 * 32 * CR_WB
    return {"units": torch.tensor(units, dtype=torch.int32, device=dev).reshape(-1, 8),
            "cuts": torch.tensor(cuts, dtype=torch.int32, device=dev), "xp": xp, "R": rbase, "route_tok": route_tok,
            "experts": experts, "cnts": [len(toks[e]) for e in experts], "nunits": len(units), "bytes": w13_bytes,
            "cost": tot, "max_cta_cost": max((cuts[i + 1] - cuts[i]) for i in range(G))}


def cr_ref(d, w, P, dequant):
    """FP32 reference inter per planned route: silu(x Wg^T) * (x Wu^T)."""
    x = d["hidden_states"].float()
    R = P["R"]
    ref = torch.zeros((max(R, 1), 4096), dtype=torch.float32, device=x.device)
    by_e = {}
    for r, (tt, e) in enumerate(P["route_tok"]):
        by_e.setdefault(e, []).append((r, tt))
    for e, lst in by_e.items():
        rr = torch.tensor([a for a, _ in lst], dtype=torch.long, device=x.device)
        tk = torch.tensor([b for _, b in lst], dtype=torch.long, device=x.device)
        W = dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e])
        h = x[tk] @ W.T
        ref[rr] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        del W
    return ref


class CrTimer:
    """The judge clone's timer: 256 MB read before every replay (cold L2), events around graph.replay()."""

    def __init__(self):
        self.flush_buf = torch.ones((256 << 20) // 4, dtype=torch.float32, device="cuda")
        self.sink = torch.empty((), dtype=torch.float32, device="cuda")
        self.stream = torch.cuda.Stream()

    def capture(self, fn):
        with torch.cuda.stream(self.stream):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                fn()
        torch.cuda.synchronize()
        return gr

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=60):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


class CrState:
    """Device buffers + module for the CR-K1 lab."""

    def __init__(self, mod, ws, gsc, G, dev):
        self.mod, self.ws, self.gsc, self.G = mod, ws, gsc, G
        self.scratch = torch.zeros((G * 2 * CR_PART,), dtype=torch.float32, device=dev)
        self.cnt = torch.zeros((8192,), dtype=torch.int32, device=dev)
        self.err = torch.zeros((4,), dtype=torch.int32, device=dev)
        self.marks = torch.zeros((G * 8,), dtype=torch.int64, device=dev)
        self.nomark = torch.zeros((0,), dtype=torch.int64, device=dev)

    def fn(self, P, inter, mode, mk=False):
        def f():
            self.mod.cr_run(P["xp"], self.ws, P["units"], P["cuts"], self.gsc, inter, self.scratch, self.cnt, self.err,
                            self.marks if mk else self.nomark, self.G, mode)
        return f


def cr_check(st, d, w, P, dequant):
    """Run once (real consumer), compare inter with the FP32 reference; returns (rel, worst_row_rel, err, cnt_clean)."""
    inter = torch.full((max(P["R"], 1), 4096), float("nan"), dtype=torch.bfloat16, device=st.ws.device)
    st.err.zero_()
    st.fn(P, inter, 0)()
    torch.cuda.synchronize()
    ref = cr_ref(d, w, P, dequant)
    got = inter.float()
    diff = got - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-30))
    rown = ref.norm(dim=1)
    floor = 0.1 * rown.mean()
    worst = float((diff.norm(dim=1) / torch.maximum(rown, floor)).max())
    nan = bool(torch.isnan(got).any())
    return rel, worst, [int(v) for v in st.err.tolist()], int(st.cnt.abs().sum()), nan, inter


def cr_rate_lab(st, G, timer, ov, log, tpass=64, nb=16, tag="R", ntoks=(8, 16, 32, 48, 64), rounds=12):
    """Consumer-rate constant (R1 rule): every unit is expert 0 / block 0 (1.15 MB, L2-resident after first touch), 384
    units x 32 stages = 453 MB of slot copies, ntok tokens on the expert. Prints S (null) and real us and GB/s per ng."""
    dev = st.ws.device
    out = {}
    for ntok in ntoks:
        if ntok > tpass:
            continue
        ids = torch.full((ntok, 8), 100, dtype=torch.int32, device=dev)
        ids[:, 0] = 0
        ids[:, 1:] = (100 + torch.arange(7, device=dev)).int()
        d = {"hidden_states": torch.randn((ntok, 4096), device=dev).to(torch.bfloat16), "topk_ids": ids,
             "topk_weights": torch.full((ntok, 8), 0.125, device=dev), "config": {}}
        P = cr_plan(d, G, tpass=tpass, nb=nb)
        nun = 384 * (16 // nb) if nb <= 16 else 384 * 2
        u = P["units"][:1].repeat(nun, 1).contiguous()
        u[:, 1] = 0
        P["units"] = u
        tot = 32 * nun
        P["cuts"] = torch.tensor([tot * i // G for i in range(G)] + [tot], dtype=torch.int32, device=dev)
        inter = torch.zeros((max(P["R"], 1), 4096), dtype=torch.bfloat16, device=dev)
        st.cnt.zero_()
        gS, gR = timer.capture(st.fn(P, inter, 1)), timer.capture(st.fn(P, inter, 0))
        tS, tR = [], []
        for _ in range(rounds):
            tS.append(timer.time(gS) - ov)
            tR.append(timer.time(gR) - ov)
        uS, uR = statistics.median(tS), statistics.median(tR)
        byt = tot * (CR_WB if nb == 16 else CR_WB // 2)
        out[ntok] = (uS, uR)
        log(f"[cr] rate {tag} ntok {ntok} (ng {u[0, 3].item()}): S {uS:.1f} us {byt / uS / 1e6:.2f} TB/s | real {uR:.1f} us "
            f"{byt / uR / 1e6:.2f} TB/s | real/S {uR / uS:.3f} | err {st.err.tolist()}")
        st.err.zero_()
        st.cnt.zero_()
        del gS, gR
    return out


def cr_lab(w, dequant=cr_dequant, k_full=None, k1_only=None, k_reset=None, log=print, rounds=20,
           cases=((256, 4), (128, 3), (32, 2), (16, 2), (8, 2), (1, 3), ("L32", 4), ("L16", 3)), seed0=770001,
           kappas=(0.22, 0.30)):
    """Arms per routing: K (shipped K2A full call), K1 (shipped K2A W13 kernel(s) alone), S (CR null consumers),
    R (CR real, kappa[0]), Rk (CR real, kappa[1]), V (2 CTAs/SM x 8 consumer warps, 256-row slots, <= 48 tokens/pass,
    kappa[0]). Correctness of R / Rk / V on every routing. Prints the pre-registered gate."""
    dev = w["w13"].device
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as ex:                      # both builds at once (prepare-time budget on the judge)
        f16, f8 = ex.submit(cr_build), ex.submit(cr_build, ncw=8, ngmax=6, nst=3)
        mod, dt = f16.result()
        mod8, dt8 = f8.result()
    a = list(mod.cr_attrs())
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    log(f"[cr] built in {dt:.0f}s; attrs (regs, local, smem, occ, ngmax, nst, ncw) {a}; {sms} SMs")
    a8 = list(mod8.cr_attrs())
    log(f"[cr] V built in {dt8:.0f}s; attrs {a8}")
    use_v = a8[3] >= 2
    if not use_v:
        log("[cr] V variant: occupancy < 2, arm dropped")
    if a[3] < 1 or a[0] != 96:
        log(f"[cr] ABORT: occupancy {a[3]} or launch registers {a[0]} != 96 (setmaxnreg 24 / 112 assumes a 96-reg CTA pool)")
        return []
    t0 = time.time()
    ws, gsc = cr_prepack(w["w13"][:32], w["w13_sf"][:32], w["w13_gs"][:32])
    torch.cuda.synchronize()
    log(f"[cr] prepack {time.time() - t0:.1f}s")
    ws2, gsc2 = cr_prepack_w2(w["w2"][:32], w["w2_sf"][:32], w["w2_gs"][:32])
    nw = (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()
    G = sms
    st = CrState(mod, ws, gsc, G, dev)
    full = CrFull(st, ws2, gsc2, nw, pdl=True)
    st8 = CrState(mod8, ws, gsc, 2 * G, dev) if use_v else None
    timer = CrTimer()
    ov = timer.overhead()
    log(f"[cr] graph overhead {ov:.2f} us")
    try:
        cr_rate_lab(st, G, timer, ov, log)
        if use_v:
            cr_rate_lab(st8, 2 * G, timer, ov, log, tpass=48, nb=32, tag="V", ntoks=(8, 16, 32, 48))
    except Exception as ex:  # noqa: BLE001
        log(f"[cr] rate lab failed: {ex!r}")
    rows = []
    for bs, ncase in cases:
        for r in range(ncase):
            if isinstance(bs, str):
                d = cr_case_counts(int(bs[1:]), CR_LAYER_COUNTS[int(bs[1:])][r], seed0 + 500 + r, dev)
            else:
                d = cr_case(bs, seed0 + 1000 * bs + r, dev)
            Ps = [cr_plan(d, G, kappa=k) for k in kappas]
            P = Ps[0]
            P8 = cr_plan(d, 2 * G, kappa=kappas[0], tpass=48, nb=32) if use_v else None
            log(f"[cr] bs={bs} r={r}: {len(P['experts'])} experts, tokens {P['cnts'][:8]}{'...' if len(P['cnts']) > 8 else ''}; "
                f"{P['nunits']} units, W13 {P['bytes'] / 1e6:.0f} MB, stages/CTA max {P['max_cta_cost']}")
            rel, worst, e0, cl, nan, _ = cr_check(st, d, w, P, dequant)
            rel2, worst2, e2, cl2, nan2, _ = cr_check(st, d, w, Ps[1], dequant)
            ok = (rel <= 1e-2 and worst <= 5e-2 and e0[0] == 0 and cl == 0 and not nan and
                  rel2 <= 1e-2 and e2[0] == 0 and cl2 == 0 and not nan2)
            relv, ev = float("nan"), [0]
            if use_v:
                relv, worstv, ev, clv, nanv, _ = cr_check(st8, d, w, P8, dequant)
                ok = ok and relv <= 1e-2 and worstv <= 5e-2 and ev[0] == 0 and clv == 0 and not nanv
            log(f"[cr]   check: rel {rel:.2e} worst-row {worst:.2e} err {e0} cnt-clean {cl == 0} nan {nan} | "
                f"k2: rel {rel2:.2e} err {e2} | V: rel {relv:.2e} err {ev} | {'OK' if ok else 'FAIL'}")
            if not ok:
                rows.append({"bs": bs, "bad": True})
                st.err.zero_(); st.cnt.zero_()
                continue
            P2 = cr_plan_w2(d, P, G, kappa=kappas[0])
            relF, worstF, eF, clF, nanF, BF = cr_check_full(full, d, w, P, P2, dequant)
            okF = relF <= 1.2e-2 and worstF <= 6e-2 and eF[0] == 0 and clF == 0 and not nanF
            log(f"[cr]   full check: rel {relF:.2e} worst-row {worstF:.2e} err {eF} cnt-clean {clF == 0} nan {nanF} "
                f"-> {'OK' if okF else 'FAIL'}")
            st.err.zero_(); st.cnt.zero_()
            inter = torch.zeros((max(P["R"], 1), 4096), dtype=torch.bfloat16, device=dev)
            arms = {"S": st.fn(P, inter, 1), "R": st.fn(P, inter, 0), "Rk": st.fn(Ps[1], inter, 0)}
            if okF:
                arms["F"] = full.fn(P, P2, BF)
            if use_v:
                arms["V"] = st8.fn(P8, inter, 0)
                arms["SV"] = st8.fn(P8, inter, 1)
            if k_full is not None:
                arms["K"] = (lambda d=d: k_full(d))
            graphs = {nm: timer.capture(fn) for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            del graphs
            if k1_only is not None:
                try:
                    g1 = timer.capture(lambda d=d: k1_only(d))
                    us["K1"] = statistics.median(max(timer.time(g1) - ov, 1e-3) for _ in range(max(8, rounds // 2)))
                    del g1
                except Exception as ex:  # noqa: BLE001
                    log(f"[cr]   K1 arm failed: {ex!r}")
                    us["K1"] = float("nan")
                if k_reset is not None:
                    k_reset()
            # one marked run of R: per-CTA consumer ends
            st.marks.zero_()
            with torch.cuda.stream(timer.stream):
                torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                st.fn(P, inter, 0, mk=True)()
            torch.cuda.synchronize()
            mv = st.marks.view(G, 8).tolist()
            t00 = min(m[0] for m in mv)
            ends = sorted((m[3] - t00) / 1e3 for m in mv if m[3] > 0)
            first = sorted((m[2] - t00) / 1e3 for m in mv if m[2] > 0)
            e1 = [int(v) for v in st.err.tolist()]
            row = {"bs": bs, "us": us, "B": P["bytes"], "cnts": P["cnts"], "err": e1, "rel": rel, "worst": worst, "relF": relF,
                   "end": (statistics.median(ends), ends[-1]) if ends else (0, 0),
                   "first": (statistics.median(first), first[-1]) if first else (0, 0)}
            rows.append(row)
            log(f"[cr]   us: " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items()) +
                f" | W13 rate S {P['bytes'] / us['S'] / 1e6:.2f} R {P['bytes'] / us['R'] / 1e6:.2f}"
                f"{(' V %.2f SV %.2f' % (P['bytes'] / us['V'] / 1e6, P['bytes'] / us['SV'] / 1e6)) if 'V' in us else ''} TB/s | "
                f"R/K1 {us['R'] / us.get('K1', float('nan')):.3f} | marks first p50/max {row['first'][0]:.1f}/"
                f"{row['first'][1]:.1f} end p50/max {row['end'][0]:.1f}/{row['end'][1]:.1f} us | err {e1}")
            st.err.zero_()
            del inter
    log("[cr] GATE (pre-registered R-cr-1 = R-cr-0's rule, medians): GO if (min(R,Rk)+2)/K1 <= 0.80 at bs256 AND <= 0.88 at bs128 AND "
        "R stream rate at bs8/bs32 >= 3.6 TB/s AND layer L16 / L32 <= 0.95; KILL if bs256 > 0.90 or stream rate < 3.4; "
        "else REVISE")
    res = {}
    for bs in sorted({x["bs"] for x in rows}, key=str):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad")]
        if not rr or any(x.get("bad") for x in rows if x["bs"] == bs):
            log(f"[cr]   bs={bs}: INVALID ({len(rr)} valid)")
            res[bs] = None
            continue
        best = [min(x["us"]["R"], x["us"]["Rk"], x["us"].get("V", 9e9)) for x in rr]
        g = statistics.median((b_ + 2.0) / x["us"].get("K1", float("nan")) for b_, x in zip(best, rr))
        rate = statistics.median(x["B"] / min(x["us"]["R"], x["us"].get("V", 9e9)) / 1e6 for x in rr)
        res[bs] = (g, rate)
        log(f"[cr]   bs={bs}: (best R + 2)/K1 {g:.3f} | R {statistics.median(x['us']['R'] for x in rr):.1f} "
            f"Rk {statistics.median(x['us']['Rk'] for x in rr):.1f} V {statistics.median(x['us'].get('V', float('nan')) for x in rr):.1f} "
            f"SV {statistics.median(x['us'].get('SV', float('nan')) for x in rr):.1f} S {statistics.median(x['us']['S'] for x in rr):.1f} "
            f"K1 {statistics.median(x['us'].get('K1', float('nan')) for x in rr):.1f} us | R W13 rate {rate:.2f} TB/s")
    # full call (diagnostic, pre-registered in R-cr-1): (F + 3) / K per board bs, projected MoE board vs K2A's 1.93
    lnsum, nsh = 0.0, 0
    for bs in (1, 8, 16, 32, 128, 256, "L16", "L32"):
        rr = [x for x in rows if x["bs"] == bs and not x.get("bad") and "F" in x["us"] and "K" in x["us"]]
        if not rr:
            continue
        fk = statistics.median((x["us"]["F"] + 3.0) / x["us"]["K"] for x in rr)
        log(f"[cr]   full bs={bs}: F {statistics.median(x['us']['F'] for x in rr):.1f} K {statistics.median(x['us']['K'] for x in rr):.1f} us "
            f"| (F + 3) / K {fk:.3f} | rel {max(x['relF'] for x in rr):.2e}")
        if not isinstance(bs, str):
            lnsum += math.log(fk)
            nsh += 1
    if nsh == 6:
        log(f"[cr] FULL-CALL projection: MoE board 1.93 x {math.exp(-lnsum / 6):.3f} = {1.93 * math.exp(-lnsum / 6):.3f} "
            f"(host plan + X pack outside timing; +3 us allowance included)")
    ok = all(v is not None for v in res.values()) and 256 in res and 128 in res
    if not ok:
        dec = "INVALID (fix, rerun under the same rule)"
    else:
        srate = min(res[b][1] for b in (8, 32) if res.get(b))
        lay = max((res[b][0] for b in ("L32", "L16") if res.get(b)), default=9.0)
        if res[256][0] > 0.90 or srate < 3.4:
            dec = "KILL (CR uniform-consumer structure; add trap)"
        elif res[256][0] <= 0.80 and res[128][0] <= 0.88 and srate >= 3.6 and lay <= 0.95:
            dec = "GO (build CR K2 + finalize; board check file)"
        else:
            dec = f"REVISE (one revision guided by S / marks, one rerun; layer worst {lay:.3f})"
    log(f"[cr] DECISION {dec}")
    del timer
    torch.cuda.empty_cache()
    return rows


# ======== CR-K1 (clean-room MoE step 0, notes/study/moe_cleanroom.md): lab in prepare(), kernel_fn = K2A ========
def _cr_d2(d):
    x = d["hidden_states"].contiguous()
    return {"hidden_states": x, "topk_ids": d["topk_ids"], "topk_weights": d["topk_weights"],
            "config": {"local_start": int(d["config"]["local_start"])}}


def _cr_k1_only(st, d):
    d2 = _cr_d2(d)
    rep = _k2a_bucket_of(d2["hidden_states"].shape[0])
    r37 = st["r37"].get(rep) if (K2A37_ENABLE and st.get("r37")) else None
    if r37 is not None:
        return _k2a37_call(st, d2, r37[0], r37[1], r37[2], mask=1)
    return _k2a33_call(st, d2, K2A33_PLAN[rep], mask=1, hot_t=K2A33_HOT_T_BUCKET.get(rep),
                       hot_w4=(76 if rep == 256 else None))


def prepare(weights):
    state = _sa_prepare(weights)
    st = state.get("st")
    if st is not None and state["ok"].get(128) and state["ok"].get(256):
        try:
            cr_lab(state["w"], _k2a_dequant, lambda d: kernel_fn(d, state), lambda d: _cr_k1_only(st, d),
                   lambda: _k2a33_reset(st), log=lambda s: print(s, flush=True))
        except Exception as ex:  # noqa: BLE001
            print("[cr] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
        _k2a33_reset(st)
        st["err"].zero_()
    else:
        print("[cr] K2A state not ready: lab skipped", flush=True)
    return state

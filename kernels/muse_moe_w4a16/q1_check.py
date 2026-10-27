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


# nonce: q1-step0-a
import statistics  # Q1 lab

Q1_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace q1 {
constexpr int KDIM = 4096;
constexpr int SW = 18432;            // weight bytes per stage: 128 rows x 256 K (codes 16384 + scales 2048)
constexpr int XROW = 528;            // staged B row: 256 bf16 + 16 B pad (conflict-free LDS.128)
constexpr int XST = XROW / 2;
constexpr int MMAX = 8;              // tokens per call
constexpr int PMAX = 8 * MMAX;       // ids per call = max routes
constexpr int RS = PMAX;             // route stride of the inter buffer [16 K-stages][RS routes][XST]
constexpr int STG = SW + 8 * XROW;   // one slot: weights + up to 8 B rows
constexpr unsigned FULL = 0xffffffffu;
constexpr long long WAIT_BUDGET = 1ll << 30;   // clock64 cycles per wait (~0.5 s) -> error flag
#if __CUDA_ARCH__ >= 900
constexpr int FULLCNT = 1;           // lane 0's arrive.expect_tx
#else
constexpr int FULLCNT = 32;          // 32 producer lanes' cp.async noinc arrivals
#endif

struct Args {
  const __nv_bfloat16* x; const int* ids; const float* wts;
  const uint8_t* ws13; const uint8_t* ws2;
  const float* gsc13; const float* gsc2; const float* nw;
  float* h; __nv_bfloat16* inter; float* y; int* cnt; unsigned* sync; int* err;
  __nv_bfloat16* out; long long* marks;
  int M, lo, mode;
};

struct Plan {
  int n, nA, R;
  int grp[32], cnt[32], off[32];
};

// ---------------- math (K2A v3.3, unchanged) ----------------
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
// One kb (16 B of codes per lane) of this warp's 16-row tile; B from the staged rows (K2A consume_kb<1>).
__device__ __forceinline__ void consume_kb1(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow,
                                            int kloc, int t, float* acc) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int j = 0; j < 8; ++j) { ag[j] = bmul2(ag[j], Sg); a8[j] = bmul2(a8[j], S8); }
  const uint4* xp = reinterpret_cast<const uint4*>(xrow + kloc + 16 * t);
  const uint4 x0 = xp[0], x1 = xp[1];
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}

// ---------------- barriers / copies ----------------
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
__device__ __forceinline__ void mbar_wait(uint32_t b, uint32_t par, int* err, int code) {
  if (mbar_try(b, par)) return;
  const long long t0 = clock64();
  int n = 0;
  while (!mbar_try(b, par)) {
    if ((++n & 255) == 0) {
      if (*reinterpret_cast<volatile int*>(err) != 0) return;
      if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return; }
    }
  }
}
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ bool reached(unsigned v, unsigned tgt) { return (int)(v - tgt) >= 0; }
// bounded spin until *p >= tgt (monotonic counter); false on timeout / error
__device__ __forceinline__ bool spin_ge(const unsigned* p, unsigned tgt, int* err, int code) {
  if (reached(ld_acq(p), tgt)) return true;
  const long long t0 = clock64();
  while (true) {
    __nanosleep(32);
    if (reached(ld_acq(p), tgt)) return true;
    if (*reinterpret_cast<volatile int*>(err) != 0) return false;
    if (clock64() - t0 > WAIT_BUDGET) { set_err(err, code); return false; }
  }
}
__device__ __forceinline__ void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void fence_proxy_async() {
#if __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async;\n" ::: "memory");
#endif
}
#if __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_expect_tx(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(b), "r"(bytes) : "memory");
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

// ---------------- plan ----------------
// phase p: 0 W13(A), 1 W13(B), 2 W2(A), 3 W2(B); this CTA's static Stream-K share [beg, end) of the phase's stages
__device__ __forceinline__ void phase_range(const Plan& S, int p, int b, int G, int& g0, int& nrb, int& beg, int& end) {
  const bool a = (p == 0 || p == 2);
  g0 = a ? 0 : S.nA;
  const int ng = a ? S.nA : S.n - S.nA;
  nrb = (p < 2) ? 64 : 32;
  const long long sp = (long long)ng * nrb * 16;
  beg = (int)(sp * b / G);
  end = (int)(sp * (b + 1) / G);
}

// ---------------- the kernel ----------------
template <int NST>
__global__ void __launch_bounds__(320, 2) q1_kernel(const Args A) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_ids[PMAX];
  __shared__ int16_t s_list[PMAX];   // route c -> id position p (token p >> 3)
  __shared__ int16_t s_rt[PMAX];     // id position p -> route c, or -1
  __shared__ Plan S;
  __shared__ int s_gen;
  __shared__ float s_red[8][8];
  __shared__ float s_sc[8];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = blockIdx.x, G = gridDim.x;
  const int M = A.M, NP = 8 * A.M;
  const int mode = A.mode;
  long long* mk = A.marks ? A.marks + (size_t)b * 16 : nullptr;
  const uint32_t ring_a = sa(smem), full_a = sa(bars), empty_a = sa(bars + NST);

  // ---------------- prologue ----------------
  if (tid == 0) {
    if (mk) mk[0] = gtime();
    for (int i = 0; i < NST; ++i) { mbar_init(full_a + 8 * i, FULLCNT); mbar_init(empty_a + 8 * i, 8); }
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#endif
    s_gen = -1;
  }
  if (tid < NP) {
    const int v = __ldg(A.ids + tid) - A.lo;
    s_ids[tid] = (v >= 0 && v < 32) ? v : -1;
  }
  __syncthreads();
  if (warp == 0) {
    // lane = local expert: count, exclusive prefix, route list in increasing id position, touched order
    const int e = lane;
    int c = 0;
    for (int p = 0; p < NP; ++p) c += (s_ids[p] == e);
    int v = c;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    const int off = v - c;
    const unsigned msk = __ballot_sync(FULL, c > 0);
    S.cnt[e] = c; S.off[e] = off;
    s_rt[lane] = -1; s_rt[lane + 32] = -1;
    __syncwarp();
    int r = 0;
    for (int p = 0; p < NP; ++p)
      if (s_ids[p] == e) { s_list[off + r] = (int16_t)p; s_rt[p] = (int16_t)(off + r); ++r; }
    const int n = __popc(msk);
    if (c > 0) S.grp[__popc(msk & ((1u << lane) - 1u))] = e;
    if (lane == 31) {
      S.n = n;
      S.nA = n - ((n >= 2) ? max(1, n / 3) : 0);
      S.R = v;
    }
  }
  __syncthreads();

  if (warp == 8) {
    // ---------------- producer ----------------
    const bool noX = (mode & 2) != 0, nodep = (mode & 4) != 0;
#if __CUDA_ARCH__ >= 900
    const uint64_t pol = policy_evict_first();
#endif
    int it = 0;
    int nd = 0;
    uint32_t dDst[NST], dBar[NST], dB[NST];
    const __nv_bfloat16* dSrc[NST];
    int dDep[NST];
    bool dok0 = nodep, dok1 = nodep;
    unsigned tgt = 0;
    bool have_tgt = false;
    auto dep_target = [&]() -> unsigned {
      if (!have_tgt) {
        const long long t0 = clock64();
        while (*reinterpret_cast<volatile int*>(&s_gen) < 0) {
          if (clock64() - t0 > WAIT_BUDGET) { if (lane == 0) set_err(A.err, 5); break; }
        }
        tgt = (unsigned)(*reinterpret_cast<volatile int*>(&s_gen) + 1) * (unsigned)G;
        have_tgt = true;
      }
      return tgt;
    };
    // blocking flush of the deferred inter copies, in order
    auto flush = [&]() {
#pragma unroll
      for (int d = 0; d < NST; ++d) {
        if (d < nd) {
          const int dp = dDep[d];
          if (!(dp ? dok1 : dok0)) {
            const unsigned tg = dep_target();
            bool ok = true;
            if (lane == 0) ok = spin_ge(A.sync + 1 + dp, tg, A.err, 3);
            __syncwarp();
            if (dp) dok1 = true; else dok0 = true;
            if (mk && lane == 0) mk[8 + dp] = gtime();
            fence_proxy_async();
            (void)ok;
          }
#if __CUDA_ARCH__ >= 900
          if (lane == 0) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);
#else
          warp_copy(lane, dDst[d], dSrc[d], dB[d]);
          cpa_arrive(dBar[d]);
#endif
        }
      }
      nd = 0;
    };
    bool first = true;
    for (int p = 0; p < 4; ++p) {
      int g0, nrb, beg, end;
      phase_range(S, p, b, G, g0, nrb, beg, end);
      const int md = (p < 2) ? 1 : 2;
      const int dp = (p == 2) ? 0 : 1;
      const uint8_t* wsrc = (md == 1) ? A.ws13 : A.ws2;
      const int per = nrb * 16;
      for (int x = beg; x < end; ++x, ++it) {
        const int ge = x / per, l = x - ge * per;
        const int e = S.grp[g0 + ge], rb = l >> 4, s = l & 15;
        const int ntok = S.cnt[e], c0 = S.off[e];
        const int slot = it % NST;
        const uint32_t par = ((it / NST) & 1) ^ 1;
        if (it >= NST) {
          if (nd > 0) flush();   // never block on a slot while copies are deferred
          mbar_wait(empty_a + 8 * slot, par, A.err, 2);
        }
        const uint32_t dst = ring_a + slot * STG, fb = full_a + 8 * slot;
        const uint32_t xbytes = noX ? 0u : (uint32_t)((md == 1) ? ntok * 512 : ntok * XROW);
        const uint8_t* src = wsrc + ((size_t)(e * nrb + rb) * 16 + s) * SW;
        bool defer = false;
        const __nv_bfloat16* isrc = A.inter + ((size_t)s * RS + c0) * XST;
        if (md == 2 && !noX) {
          bool ok = dp ? dok1 : dok0;
          if (!ok) {   // non-blocking probe
            if (mk && lane == 0 && mk[12 + dp] == 0) mk[12 + dp] = gtime();   // first time this dependency is needed
            unsigned v = 0;
            if (lane == 0) v = ld_acq(A.sync + 1 + dp);
            v = __shfl_sync(FULL, v, 0);
            ok = reached(v, dep_target());
            if (ok) {
              if (dp) dok1 = true; else dok0 = true;
              if (mk && lane == 0) mk[8 + dp] = gtime();
              fence_proxy_async();
            }
          }
          defer = !ok || nd > 0;
        }
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          mbar_expect_tx(fb, SW + xbytes);
          bulk_g2s_ef(dst, src, SW, fb, pol);
        }
        __syncwarp();
        if (!noX) {
          if (md == 1) {
            if (lane >= 1 && lane <= ntok) {
              const int tok = ((int)s_list[c0 + lane - 1]) >> 3;
              bulk_g2s(dst + SW + (lane - 1) * XROW, A.x + (size_t)tok * KDIM + s * 256, 512, fb);
            }
          } else if (!defer) {
            if (lane == 0) bulk_g2s(dst + SW, isrc, xbytes, fb);
          }
        }
#else
        warp_copy(lane, dst, src, SW);
        if (!noX && md == 1) {
          for (int n = 0; n < ntok; ++n) {
            const int tok = ((int)s_list[c0 + n]) >> 3;
            warp_copy(lane, dst + SW + n * XROW, A.x + (size_t)tok * KDIM + s * 256, 512);
          }
        } else if (!noX && !defer) {
          warp_copy(lane, dst + SW, isrc, xbytes);
        }
        if (!defer) cpa_arrive(fb);
#endif
        if (defer) {
#pragma unroll
          for (int d = 0; d < NST; ++d)
            if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = isrc; dB[d] = xbytes; dDep[d] = dp; }
          ++nd;
        }
        if (first) { first = false; if (mk && lane == 0) mk[1] = gtime(); }
      }
    }
    if (nd > 0) flush();
    if (mk && lane == 0) mk[10] = gtime();
    return;
  }

  if (warp == 9) {
    // ---------------- epilogue warp ----------------
    if (lane == 0) {
      const unsigned old = atomicAdd(A.sync + 0, 1u);   // generation ticket: every CTA once per call
      *reinterpret_cast<volatile int*>(&s_gen) = (int)(old / (unsigned)G);
    }
    const bool nullc = (mode & 1) != 0;
    int nsplit = 0;
    for (int p = 0; p < 2; ++p) {
      int g0, nrb, beg, end;
      phase_range(S, p, b, G, g0, nrb, beg, end);
      const int per = nrb * 16;
      for (int x = beg; x < end;) {
        const int ge = x / per, l = x - ge * per;
        const int e = S.grp[g0 + ge], rb = l >> 4, s0 = l & 15;
        const int s1 = min(16, s0 + (end - x));
        x += s1 - s0;
        if ((s0 == 0 && s1 == 16) || nullc) continue;
        const int ntok = S.cnt[e], c0 = S.off[e];
        bar_sync(4 + nsplit, 288);
        ++nsplit;
        __threadfence();   // cumulative over the consumers' h atomics (ordered before us by the barrier)
        int last = 0;
        if (lane == 0) {
          int* cp = A.cnt + e * 64 + rb;
          const int old = atomicAdd(cp, s1 - s0);
          last = (old + (s1 - s0) == 16);
          if (last) atomicExch(cp, 0);
        }
        last = __shfl_sync(FULL, last, 0);
        if (last) {
          __threadfence();
          const int col = rb * 64 + 2 * lane;
          float2 gv[MMAX], uv[MMAX];
#pragma unroll
          for (int n = 0; n < MMAX; ++n) {
            if (n < ntok) {
              const float* hp = A.h + (size_t)(c0 + n) * (2 * KDIM) + col;
              gv[n] = __ldcg(reinterpret_cast<const float2*>(hp));
              uv[n] = __ldcg(reinterpret_cast<const float2*>(hp + KDIM));
            }
          }
#pragma unroll
          for (int n = 0; n < MMAX; ++n) {
            if (n < ntok) {
              const int c = c0 + n;
              float* hp = A.h + (size_t)c * (2 * KDIM) + col;
              *reinterpret_cast<uint32_t*>(A.inter + ((size_t)(col >> 8) * RS + c) * XST + (col & 255)) =
                  pack_bf2(swiglu(gv[n].x, uv[n].x), swiglu(gv[n].y, uv[n].y));
              *reinterpret_cast<float2*>(hp) = make_float2(0.f, 0.f);
              *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);
            }
          }
        }
      }
      bar_sync(1 + p, 288);   // the consumers are done with this W13 wave (inter stores / h atomics issued)
      __threadfence();
      if (lane == 0) atomicAdd(A.sync + 1 + p, 1u);
    }
    bar_sync(3, 288);         // the consumers are done with W2 (y stores / atomics issued)
    __threadfence();
    if (lane == 0) {
      atomicAdd(A.sync + 3, 1u);
      if (mk) mk[6] = gtime();
    }
    return;
  }

  // ---------------- consumers (warps 0-7: warp w = tile w of the 128-row item) ----------------
  const bool nullc = (mode & 1) != 0;
  {
    // y rows [0, R) -> 0 (W2 split items add into them; W2 starts only after every CTA passed its W13(A))
    const size_t n4 = (size_t)S.R * KDIM / 4;
    float4* yp = reinterpret_cast<float4*>(A.y);
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (size_t i = (size_t)b * 256 + tid; i < n4; i += (size_t)G * 256) yp[i] = z;
  }
  const int g = lane >> 2, t = lane & 3;
  int it = 0, nsplit = 0;
  for (int p = 0; p < 4; ++p) {
    int g0, nrb, beg, end;
    phase_range(S, p, b, G, g0, nrb, beg, end);
    const int md = (p < 2) ? 1 : 2;
    const int per = nrb * 16;
    for (int x = beg; x < end;) {
      const int ge = x / per, l = x - ge * per;
      const int e = S.grp[g0 + ge], rb = l >> 4, s0 = l & 15;
      const int s1 = min(16, s0 + (end - x));
      const int ntok = S.cnt[e], c0 = S.off[e];
      float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
      for (int s = s0; s < s1; ++s, ++it) {
        const int slot = it % NST;
        mbar_wait(full_a + 8 * slot, (it / NST) & 1, A.err, 1);
        if (!nullc) {
          const uint8_t* st = smem + slot * STG;
          const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
          const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
          const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
          uint4 wv[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
          const uint32_t sv0 = sp[0], sv1 = sp[32];
          consume_kb1(wv[0], sv0, 0, xrow, 0, t, acc);
          consume_kb1(wv[1], sv0, 1, xrow, 64, t, acc);
          consume_kb1(wv[2], sv1, 0, xrow, 128, t, acc2);
          consume_kb1(wv[3], sv1, 1, xrow, 192, t, acc2);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty_a + 8 * slot);
      }
      const bool whole = (s0 == 0 && s1 == 16);
      if (!nullc) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += acc2[i];
        const int tile = rb * 8 + warp;
        if (md == 1) {
          const float gs = __ldg(A.gsc13 + e);
          const int col = 8 * tile + g;
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              const int c = c0 + n;
              const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
              if (whole) {
                A.inter[((size_t)(col >> 8) * RS + c) * XST + (col & 255)] = __float2bfloat16_rn(swiglu(v0, v1));
              } else {
                atomicAdd(A.h + (size_t)c * (2 * KDIM) + col, v0);
                atomicAdd(A.h + (size_t)c * (2 * KDIM) + KDIM + col, v1);
              }
            }
          }
          if (!whole) { bar_arrive(4 + nsplit, 288); ++nsplit; }
        } else {
          const float gs = __ldg(A.gsc2 + e);
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int n = 2 * t + hh;
            if (n < ntok) {
              float* yp = A.y + (size_t)(c0 + n) * KDIM + 16 * tile + g;
              const float v0 = acc[hh] * gs, v1 = acc[2 + hh] * gs;
              if (whole) { yp[0] = v0; yp[8] = v1; }
              else { atomicAdd(yp, v0); atomicAdd(yp + 8, v1); }
            }
          }
        }
      }
      x += s1 - s0;
    }
    if (p < 2) bar_arrive(1 + p, 288);
    if (p == 3) bar_arrive(3, 288);
    if (mk && tid == 0) mk[2 + p] = gtime();
  }

  // ---------------- finalize: post-expert norm + weighted sum, item (token b / 4, quarter b % 4) ----------------
  if ((mode & 8) || b >= 4 * M) return;
  {
    if (tid == 0) {
      const long long t0 = clock64();
      while (*reinterpret_cast<volatile int*>(&s_gen) < 0) {
        if (clock64() - t0 > WAIT_BUDGET) { set_err(A.err, 5); break; }
      }
      const unsigned tg = (unsigned)(*reinterpret_cast<volatile int*>(&s_gen) + 1) * (unsigned)G;
      spin_ge(A.sync + 3, tg, A.err, 4);
      if (mk) mk[11] = gtime();
    }
    bar_sync(8, 256);
    const int tk = b >> 2, q = b & 3;
    // local routes of the token, compacted (route id, slot)
    int nl = 0;
    int lr[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int c = s_rt[tk * 8 + j];
      lr[j] = 0;
      if (c >= 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) if (i == nl) lr[i] = (c << 3) | j;
        ++nl;
      }
    }
    // pass 1: sums of squares of z = bf16(y) * (1 + w) over the full rows, 2 routes per batch
    float ssp[8];
#pragma unroll
    for (int j0 = 0; j0 < 8; j0 += 2) {
      ssp[j0] = 0.f; ssp[j0 + 1] = 0.f;
      if (j0 < nl) {
        float4 v[2][4];
#pragma unroll
        for (int jj = 0; jj < 2; ++jj)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            v[jj][k] = (j0 + jj < nl)
                ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)(lr[j0 + jj] >> 3) * KDIM + 1024 * k + 4 * tid))
                : make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float4 nk = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * k + 4 * tid));
#pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            float zz, a = ssp[j0 + jj];
            zz = bf16r(v[jj][k].x) * nk.x; a += zz * zz;
            zz = bf16r(v[jj][k].y) * nk.y; a += zz * zz;
            zz = bf16r(v[jj][k].z) * nk.z; a += zz * zz;
            zz = bf16r(v[jj][k].w) * nk.w; a += zz * zz;
            ssp[j0 + jj] = a;
          }
        }
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float a = ssp[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(FULL, a, o);
      if (lane == 0) s_red[warp][j] = a;
    }
    bar_sync(8, 256);
    if (tid < 8) {
      float a = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) a += s_red[w][tid];
      int lv = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) if (i == tid) lv = lr[i];
      s_sc[tid] = (tid < nl) ? rsqrtf(a * (1.f / 4096.f) + 1e-8f) * __ldg(A.wts + tk * 8 + (lv & 7)) : 0.f;
    }
    bar_sync(8, 256);
    // pass 2: this quarter's columns, all local routes in one batch
    const float4 w = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * q + 4 * tid));
    float4 vq[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      vq[j] = (j < nl) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)(lr[j] >> 3) * KDIM + 1024 * q + 4 * tid))
                       : make_float4(0.f, 0.f, 0.f, 0.f);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float r = s_sc[j];
      a0 += bf16r(vq[j].x) * w.x * r; a1 += bf16r(vq[j].y) * w.y * r;
      a2 += bf16r(vq[j].z) * w.z * r; a3 += bf16r(vq[j].w) * w.w * r;
    }
    uint2 o;
    o.x = pack_bf2(a0, a1); o.y = pack_bf2(a2, a3);
    *reinterpret_cast<uint2*>(A.out + (size_t)tk * KDIM + 1024 * q + 4 * tid) = o;
    if (mk && tid == 0) mk[7] = gtime();
  }
}

template <int NST>
constexpr int smem_bytes() { return NST * STG + 2 * NST * 8; }

template <int NST>
void launch(const Args& a, int G) {
  constexpr int sm = smem_bytes<NST>();
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  q1_kernel<NST><<<G, 320, sm, at::cuda::getCurrentCUDAStream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template <int NST>
std::vector<int64_t> attrs() {
  constexpr int sm = smem_bytes<NST>();
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  cudaFuncSetAttribute(q1_kernel<NST>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  cudaFuncAttributes fa;
  cudaFuncGetAttributes(&fa, q1_kernel<NST>);
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, q1_kernel<NST>, 320, sm);
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), occ};
}
}  // namespace q1

void q1_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor inter,
            torch::Tensor y, torch::Tensor cnt, torch::Tensor sync, torch::Tensor err, torch::Tensor out,
            torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode) {
  TORCH_CHECK(M >= 1 && M <= q1::MMAX, "q1: M out of range");
  q1::Args a;
  a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  a.ids = ids.data_ptr<int>(); a.wts = wts.data_ptr<float>();
  a.ws13 = reinterpret_cast<const uint8_t*>(ws13.data_ptr()); a.ws2 = reinterpret_cast<const uint8_t*>(ws2.data_ptr());
  a.gsc13 = gsc13.data_ptr<float>(); a.gsc2 = gsc2.data_ptr<float>(); a.nw = nw.data_ptr<float>();
  a.h = h.data_ptr<float>(); a.inter = reinterpret_cast<__nv_bfloat16*>(inter.data_ptr()); a.y = y.data_ptr<float>();
  a.cnt = cnt.data_ptr<int>(); a.sync = reinterpret_cast<unsigned*>(sync.data_ptr<int>()); a.err = err.data_ptr<int>();
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  a.marks = marks.numel() >= G * 16 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  a.M = (int)M; a.lo = (int)lo; a.mode = (int)mode;
  switch (cfg) {
    case 3: q1::launch<3>(a, (int)G); break;
    case 4: q1::launch<4>(a, (int)G); break;
    default: TORCH_CHECK(false, "q1: bad cfg");
  }
}
std::vector<int64_t> q1_attrs(int64_t cfg) {
  switch (cfg) {
    case 3: return q1::attrs<3>();
    case 4: return q1::attrs<4>();
    default: return {-1};
  }
}
"""

Q1_CPP = r"""
#include <torch/extension.h>
#include <vector>
void q1_run(int64_t cfg, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13, torch::Tensor ws2,
            torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h, torch::Tensor inter,
            torch::Tensor y, torch::Tensor cnt, torch::Tensor sync, torch::Tensor err, torch::Tensor out,
            torch::Tensor marks, int64_t M, int64_t lo, int64_t G, int64_t mode);
std::vector<int64_t> q1_attrs(int64_t cfg);
"""

Q1_EXPERT_BYTES = 768 * 36864    # 28.31 MB per touched expert (W13 + W2, codes + scales)
Q1_CFGS = (3, 4)                 # ring stages


def q1_build(arch="sm_90a", verbose=False):
    from torch.utils.cpp_extension import load_inline
    gen = "-gencode=arch=compute_90a,code=sm_90a" if arch == "sm_90a" else "-gencode=arch=compute_86,code=sm_86"
    tag = hashlib.sha1((Q1_CPP + Q1_CUDA + arch).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"q1_moe_{tag}", cpp_sources=Q1_CPP, cuda_sources=Q1_CUDA, functions=["q1_run", "q1_attrs"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", gen], verbose=verbose)
    return mod, time.time() - t0


def q1_bufs(dev, G):
    return {
        "h": torch.zeros((64, 8192), dtype=torch.float32, device=dev),
        "inter": torch.zeros((16 * 64 * 264,), dtype=torch.bfloat16, device=dev),
        "y": torch.zeros((64, 4096), dtype=torch.float32, device=dev),
        "cnt": torch.zeros((32 * 64,), dtype=torch.int32, device=dev),
        "sync": torch.zeros((8,), dtype=torch.int32, device=dev),
        "err": torch.zeros((8,), dtype=torch.int32, device=dev),
        "marks": torch.zeros((G * 16,), dtype=torch.int64, device=dev),
        "nomark": torch.zeros((0,), dtype=torch.int64, device=dev),
    }


def q1_fn(mod, st, bufs, data, cfg, G, mode=0, mark=False):
    """Closure for one call (graph-capturable): returns out [M, 4096] bf16."""
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    lo = int(data["config"]["local_start"])
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    mk = bufs["marks"] if mark else bufs["nomark"]

    def f():
        mod.q1_run(cfg, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], bufs["h"],
                   bufs["inter"], bufs["y"], bufs["cnt"], bufs["sync"], bufs["err"], out, mk, m, lo, G, mode)
        return out
    return f


def q1_case(bs, seed, dev, dist="normal"):
    """generate_input's normal / ep8 / busiest routing math (our own copy; seeds disjoint from the board's)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=g, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=g, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=g, device=dev)
        u = torch.rand((256,), generator=g, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((256,), generator=g, device=dev).pow(6) * 4
    routed = scores.topk(8, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": 256, "local_start": 0,
                       "local_count": 32, "top_k": 8, "sharding": "ep8"}}


def q1_touched(data):
    ids = data["topk_ids"].long() - int(data["config"]["local_start"])
    loc = ids[(ids >= 0) & (ids < 32)]
    return int(torch.unique(loc).numel())


def q1_rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


class Q1Timer:
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
                out = fn()
        torch.cuda.synchronize()
        return gr, out

    def time(self, gr):
        with torch.cuda.stream(self.stream):
            torch.sum(self.flush_buf, dim=0, out=self.sink)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gr.replay()
            e.record()
        e.synchronize()
        return s.elapsed_time(e) * 1e3

    def overhead(self, n=100):
        tiny = torch.zeros(1, device="cuda")
        with torch.cuda.stream(self.stream):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=self.stream):
                tiny.add_(0)
        torch.cuda.synchronize()
        return statistics.median(self.time(gr) for _ in range(n))


def q1_marks_summary(bufs, G, t_base=None):
    mv = bufs["marks"].view(-1, 16)[:G].tolist()
    t0 = min(r[0] for r in mv) if t_base is None else t_base

    def col(k, only=None):
        v = [(r[k] - t0) / 1e3 for r in mv if r[k] > 0 and (only is None or only(r))]
        return sorted(v)

    def p50max(v):
        return (statistics.median(v), v[-1]) if v else (float("nan"), float("nan"))
    out = {}
    for name, k in (("first_copy", 1), ("w13a_end", 2), ("w13b_end", 3), ("w2a_end", 4), ("w2_end", 5),
                    ("sync2_pub", 6), ("fin_end", 7), ("depA", 8), ("depB", 9), ("prod_done", 10), ("fin_seen", 11),
                    ("needA", 12), ("needB", 13)):
        out[name] = p50max(col(k))
    # producer stall on a dependency: satisfied - first needed (0 when it was met at the first probe)
    for nm, a, b2 in (("stallA", 12, 8), ("stallB", 13, 9)):
        v = sorted((r[b2] - r[a]) / 1e3 for r in mv if r[a] > 0 and r[b2] > 0)
        out[nm] = p50max(v) if v else (0.0, 0.0)
    return out


def q1_lab(st, k2a_fn, k2a_state, log=print, rounds=30, inputs=3, seeds0=919001, sizes=(1, 8), arch="sm_90a"):
    """Step-0 gate (notes/memory/gpu_requests.md R-moe1-0): arms K (shipped K2A), Q3 / Q4 (one-launch two-wave, 3 / 4
    ring stages), Q0 (Q3 with null consumers), S (Q3 pure stream: null consumers, no X / inter, no deps, no finalize)."""
    dev = st["ws13"].device
    mod, dt = q1_build(arch)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    occ = {}
    for cfg in Q1_CFGS:
        a = list(mod.q1_attrs(cfg))
        occ[cfg] = a[3]
        log(f"[q1] cfg {cfg} attrs (regs, local, smem, occ) {a}")
    log(f"[q1] built in {dt:.0f}s, {sms} SMs")
    timer = Q1Timer()
    ov = timer.overhead()
    log(f"[q1] graph overhead {ov:.2f} us")
    rows = []
    for bs in sizes:
        for r in range(inputs):
            d = q1_case(bs, seeds0 + 1000 * bs + r, dev)
            ne = q1_touched(d)
            B = ne * Q1_EXPERT_BYTES
            spec = {}
            for cfg in Q1_CFGS:
                if occ[cfg] >= 1:   # every CTA must be resident (in-kernel waits): G = SMs x min(occ, 2)
                    spec[f"Q{cfg}"] = (cfg, min(occ[cfg], 2) * sms, 0)
            if "Q3" in spec:
                spec["Q0"] = (3, spec["Q3"][1], 1)
                spec["S"] = (3, spec["Q3"][1], 15)
            bufs = {nm: q1_bufs(dev, v[1]) for nm, v in spec.items()}
            arms = {"K": lambda d=d: k2a_fn(d, k2a_state)}
            arms.update({nm: q1_fn(mod, st, bufs[nm], d, *v) for nm, v in spec.items()})
            ref = arms["K"]().float().clone()
            torch.cuda.synchronize()
            rel, errs = {}, {}
            for nm in spec:
                if spec[nm][2] != 0:
                    continue
                bufs[nm]["err"].zero_()
                o1 = arms[nm]().float().clone()
                o2 = arms[nm]().float().clone()
                torch.cuda.synchronize()
                rel[nm] = (q1_rel(o1, ref), q1_rel(o2, ref), bool(torch.isfinite(o1).all()))
                errs[nm] = [int(v) for v in bufs[nm]["err"].tolist()[:2]]
            graphs = {nm: timer.capture(fn)[0] for nm, fn in arms.items()}
            names = list(graphs)
            ts = {nm: [] for nm in names}
            for rd in range(rounds):
                k = rd % len(names)
                for nm in names[k:] + names[:k]:
                    ts[nm].append(max(timer.time(graphs[nm]) - ov, 1e-3))
            us = {nm: statistics.median(v) for nm, v in ts.items()}
            tl = {}
            for nm in ("Q3", "Q0"):
                if nm not in spec:
                    continue
                cfg, G, md = spec[nm]
                bufs[nm]["marks"].zero_()
                f = q1_fn(mod, st, bufs[nm], d, cfg, G, md, mark=True)
                with torch.cuda.stream(timer.stream):
                    torch.sum(timer.flush_buf, dim=0, out=timer.sink)
                    f()
                torch.cuda.synchronize()
                tl[nm] = q1_marks_summary(bufs[nm], G)
            e_all = {nm: [int(v) for v in bufs[nm]["err"].tolist()[:2]] for nm in spec}
            rows.append({"bs": bs, "ne": ne, "B": B, "us": us, "rel": rel, "err": e_all, "tl": tl})
            log(f"[q1] bs={bs} r={r} experts {ne} ({B / 1e6:.1f} MB): " + " ".join(f"{nm} {v:.1f}" for nm, v in us.items())
                + " | " + " ".join(f"{nm}/K {us[nm] / us['K']:.3f}" for nm in us if nm != "K")
                + f" | B/4.4TB/s {B / 4.4e6:.1f}")
            log("[q1]   rel vs K (call 1, call 2, finite): " + " ".join(f"{nm} {a:.1e}/{b2:.1e}/{fi}" for nm, (a, b2, fi) in rel.items())
                + f" | err {e_all}")
            for nm, s in tl.items():
                log(f"[q1]   TL {nm}: " + " ".join(f"{k} {a:.1f}/{b2:.1f}" for k, (a, b2) in s.items()))
            del graphs, bufs
            torch.cuda.empty_cache()
    # gate (pre-registered R-moe1-0)
    log("[q1] GATE (median over routings of best(Q3, Q4) / K): bs1 <= 0.85 and bs8 <= 0.92, rel <= 1e-3, err 0")
    verdict = True
    for bs in sizes:
        rr = [x for x in rows if x["bs"] == bs]
        if not rr:
            continue
        ratios, ok_c = [], True
        for x in rr:
            qs = [nm for nm in ("Q3", "Q4") if nm in x["us"]]
            if not qs:
                ok_c = False
                continue
            best = min(qs, key=lambda nm: x["us"][nm])
            ratios.append(x["us"][best] / x["us"]["K"])
            for nm in qs:
                a, b2, fi = x["rel"].get(nm, (1.0, 1.0, False))
                ok_c &= (a <= 1e-3 and b2 <= 1e-3 and fi and x["err"][nm][0] == 0)
        mr = statistics.median(ratios) if ratios else float("nan")
        lim = 0.85 if bs == 1 else 0.92
        okb = ok_c and mr <= lim
        if bs in (1, 8):
            verdict &= okb
        else:
            ok_c and log(f"[q1]   (bs={bs} is information only)")
        log(f"[q1]   bs={bs}: median best/K {mr:.3f} (limit {lim}), correctness {'ok' if ok_c else 'FAIL'} -> "
            f"{'pass' if okb else 'FAIL'}")
    log(f"[q1] GATE {'GO' if verdict else 'NO-GO'}")
    del timer
    torch.cuda.empty_cache()
    return rows


# ======== Q1 step 0 (notes/study/moe_structure_fresh.md, request R-moe1-0): lab in prepare(), kernel_fn = K2A ========
def prepare(weights):
    state = _sa_prepare(weights)
    if state.get("st") is not None:
        try:
            q1_lab(state["st"], kernel_fn, state, log=lambda s: print(s, flush=True), rounds=60, inputs=4,
                   sizes=(1, 4, 8))
        except Exception as ex:  # noqa: BLE001
            print("[q1] lab FAILED: " + repr(ex)[-3000:], flush=True)
        torch.cuda.synchronize()
    return state

"""PX: K2 (W2) as a dynamic, finalize-free queue that starts while K1's younger CTAs still run (MoE fresh designer for
bs 16 / 32, notes/study/moe_tails_fresh.md). LAB FILE: box runs only (tools/box/px_run.py).

Diagnosis this tests (box_out/sa5/ship, fb4, setup): on every SM the CTA that started first finishes K1 (K2) 8-20 us
(3-9 us) before its co-resident partner with IDENTICAL work (132 / 132 SMs on a cold-only routing): warp-scheduler age
priority, not work. K2's CTAs land in the freed slots and sleep in griddepcontrol.wait; the late K2 landers then carry a
full static share. Under that, a static schedule cannot balance; a dynamic K2 that may start on experts whose W13 is
already complete can fill K1's tail with W2 work.

PX = k2a_v37 vid 3 (HK2 NCH 4, NST 3, 2 CTAs/SM) with:
  * K1 (k1_kernel36, shipped code; probe bit PXB): skips the y / ss zeroing (y is self-reset by K3) and, after every warp
    of the CTA is done (inter / h writes, split finalizes), thread 0 adds the CTA's stage count per (expert, W13 quarter)
    to rdy[e * 4 + q] (fence + red); a quarter = W13 row blocks 16q..16q+15 = inter columns 1024q.. (target 256 np).
  * K2 (k2_px, PDL secondary): units (expert, W2 row block, stages [s0, s0 + D)) in K1-readiness order (hot experts
    first, then cold, K1's stage-sequence order; within an expert K-chunk major); the first unit is blockIdx, later units
    by ticket (rdy[128], requested one unit ahead); a unit's X copies are issued once its quarters are complete (relaxed
    polls, value prefetched one unit ahead, then fence.proxy.async); the consumers are K2A's cold_item / hc_item stage
    loops verbatim with a flush that adds the unit's partial y by red.global.add.v4.f32 (4x4 lane transpose); no epilogue
    work (ss is computed in K3). griddepcontrol.wait only at the end (the chain then implies K1 complete before K3).
    probe bit PXW: griddepcontrol.wait before the first X copy instead (dynamic queue without the early start).
  * K3 (k3_px, PDL, 4-CTA cluster per token, grid 4 M): SS3's k3_kernel_ss (ss from y over DSMEM in rank order) + zeroes
    the y entries it read + resets rdy[0..129].
Bounded waits set err (no trap). Numerics: K2A's consumer; y is a sum of D-stage partials in arbitrary order (FP32).
"""
import hashlib
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "kernels/muse_moe_w4a16",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PXB = 28   # probe bit: PX mode (K1 skips y / ss zeroing, signals rdy)
PXW = 29   # probe bit: PX_WAIT arm (K2 waits for K1 before its first X copy)
PXV = 30   # probe bit: v2 producer pipeline (R-tail-0b): next ticket + readiness load issued one unit ahead

DEV = r"""
// ======================================================================================================================
// PX lab (kernels/lab/px_moe_lab.py)
// ======================================================================================================================
__device__ __forceinline__ long long px_gt() {
  long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int px_smid() {
  int s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}
__device__ __forceinline__ long long* px_rec(const Params& P, int k) {
  return (P.pxmk != nullptr && blockIdx.x < 1024) ? P.pxmk + ((size_t)k * 1024 + blockIdx.x) * 16 : nullptr;
}
__device__ __forceinline__ int px_ld(const int* p) {
  int v;
  asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void px_red_v4(float* p, float a, float b, float c, float d) {
  asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n" :: "l"(p), "f"(a), "f"(b), "f"(c), "f"(d)
               : "memory");
}

// K1 end of CTA (thread 0, after a __syncthreads that follows gemm36): this CTA's stages per (expert, W13 quarter).
__device__ void px_k1_signal(const Params& P, const Route& S, const int* pr) {
  __threadfence();   // cumulative over every warp's inter / h writes (ordered before us by the barrier)
  for (int sg = 0; sg < 4; ++sg) {
    const int xb = pr[3 * sg], xe = pr[3 * sg + 1];
    if (xe <= xb) continue;
    const int* sb = pr[3 * sg + 2] ? S.sbh : S.sbc;
    int e = 0;
    for (int x = xb; x < xe;) {
      while (sb[e + 1] <= x) ++e;
      const int qs = 256 * S.np[e];
      const int q = (x - sb[e]) / qs;
      const int x1 = min(xe, sb[e] + qs * (q + 1));
      atomicAdd(P.rdy + e * 4 + q, x1 - x);
      x = x1;
    }
  }
}

// unit u -> (expert, W2 row block, pass, stages [s0, s1), tokens). Full warp (ballot over the 32 order positions).
struct PXU { int e, rb, s0, s1, ntok, c0; };
template <int NCH>
__device__ __forceinline__ PXU px_decode(const Params& P, const Route& S, const int* ord, const int* ub, int D, int u) {
  const int lane = threadIdx.x & 31;
  const int i = __popc(__ballot_sync(FULL, ub[lane + 1] <= u));
  PXU w;
  w.e = ord[i];
  const int np = S.np[w.e];
  const int ue = u - ub[i];
  const int per = 32 * np;
  const int kc = ue / per, r = ue - kc * per;
  w.rb = r / np;
  const int j = r - w.rb * np;
  w.s0 = kc * D;
  w.s1 = w.s0 + D;
  const int TP = (S.cnt[w.e] > P.hot_t) ? 8 * NCH : 8;
  w.ntok = min(TP, S.cnt[w.e] - j * TP);
  w.c0 = S.off[w.e] + j * TP;
  return w;
}

// The unit's partial y (NQ 8-token chunks) by red.v4: lanes 16h + 4j + t hold rows 4h + j (and + 8) of the warp's 16-row
// tile for tokens 2t, 2t + 1; a 4x4 transpose over j (lane xor 8, then xor 4) gives each lane 4 consecutive rows of one
// (token, row half): lane j -> token 2t + (j & 1), rows 8 (j >> 1) + 4h .. + 3.
template <int NQ>
__device__ __forceinline__ void px_flush2(const Params& P, float (*acc)[4], float gs, int rb, int c0, int ntok) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int j = g & 3, h = g >> 2;
  const int tile = rb * 8 + warp;
  int z = 0;
  asm volatile("" : "+r"(z));
  float* yb = P.y + z;
#pragma unroll
  for (int q = 0; q < NQ; ++q) {
    float x0 = acc[q][0] * gs, x1 = acc[q][1] * gs, x2 = acc[q][2] * gs, x3 = acc[q][3] * gs;
    {   // swap the off-diagonal 2x2 blocks between j and j ^ 2 (lane ^ 8)
      const bool lo = j < 2;
      const float s0 = lo ? x2 : x0, s1 = lo ? x3 : x1;
      const float r0 = __shfl_xor_sync(FULL, s0, 8), r1 = __shfl_xor_sync(FULL, s1, 8);
      if (lo) { x2 = r0; x3 = r1; } else { x0 = r0; x1 = r1; }
    }
    {   // transpose the 2x2 blocks between j and j ^ 1 (lane ^ 4)
      const bool ev = (j & 1) == 0;
      const float s0 = ev ? x1 : x0, s1 = ev ? x3 : x2;
      const float r0 = __shfl_xor_sync(FULL, s0, 4), r1 = __shfl_xor_sync(FULL, s1, 4);
      if (ev) { x1 = r0; x3 = r1; } else { x0 = r0; x2 = r1; }
    }
    const int n = q * 8 + 2 * t + (j & 1);
    if (n < ntok) px_red_v4(yb + (size_t)(c0 + n) * KDIM + 16 * tile + 8 * (j >> 1) + 4 * h, x0, x1, x2, x3);
  }
}

// K2A's cold_item / hc_item stage loops verbatim (MODE 2), flush = px_flush2.
template <int NST, int STG>
__device__ __forceinline__ void px_cold_item2(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                              int& it_ref, int s0, int s1, float gs, int rb, int c0, int ntok) {
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
  px_flush2<1>(P, acc, gs, rb, c0, ntok);
}

template <int NQ, int NST, int STG>
__device__ __forceinline__ void px_hc_item2(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                            int& it_ref, int s0, int s1, float gs, int rb, int c0, int ntok) {
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
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        uint32_t a[4], b[4];
        deq4(hf ? w.y : w.x, a);
        deq4(hf ? w.w : w.z, b);
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) { a[jj] = bmul2(a[jj], Sg); b[jj] = bmul2(b[jj], S8); }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
          const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);
          mma16816(acc[q], a[0], b[0], a[1], b[1], xv.x, xv.y);
          mma16816(acc[q], a[2], b[2], a[3], b[3], xv.z, xv.w);
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty_a + 8 * slot);
  }
  it_ref = it;
  px_flush2<NQ>(P, acc, gs, rb, c0, ntok);
}

// K2: dynamic finalize-free W2 queue. Marks (rec 1): [0] smid [1] entry [2] route done [3] first X issued [4] END
// [5] units [6] readiness spin ns [7] first unit [8] polls that were not ready at first look.
template <int NCH, int NST>
__global__ void __launch_bounds__(320, 2) k2_px(const Params P) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int STG = SW + 8 * NCH * XROW;
  __shared__ Route S;
  __shared__ int s_tag[NST];
  __shared__ int s_ord[32];
  __shared__ int s_ub[33];
  __shared__ int s_tu[32];
  __shared__ float s_gs[32];
  __shared__ unsigned s_rq[4];
  long long* mk = px_rec(P, 1);
  if (mk != nullptr && threadIdx.x == 0) { mk[0] = px_smid(); mk[1] = px_gt(); }
  pdl_launch();
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + NST * STG);
  setup_bars<1, NST>(bars);
  route<1, 8 * NCH, true, false>(P, S, nullptr, nullptr, 32, false);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int D = P.pxd;
  if (warp == 0) {   // expert order: hot (index order) then cold (index order); unit prefix over the order
    const int c = S.cnt[lane];
    const bool hot = c > P.hot_t;
    const unsigned mh = __ballot_sync(FULL, c > 0 && hot), mc = __ballot_sync(FULL, c > 0 && !hot);
    const unsigned lt = (1u << lane) - 1u;
    const int pos = (c <= 0) ? -1 : (hot ? __popc(mh & lt) : __popc(mh) + __popc(mc & lt));
    s_tu[lane] = 0;
    __syncwarp();
    if (pos >= 0) { s_ord[pos] = lane; s_tu[pos] = S.np[lane] * 32 * (16 / D); }
    __syncwarp();
    int v = s_tu[lane];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) { const int nb = __shfl_up_sync(FULL, v, d); if (lane >= d) v += nb; }
    s_ub[lane + 1] = v;
    if (lane == 0) s_ub[0] = 0;
    s_gs[lane] = __ldg(P.gsc2 + lane);
    if (mk != nullptr && lane == 0) mk[2] = px_gt();
  }
  __syncthreads();
  const int U = s_ub[32];
  const uint32_t ring_a = sa(smem);
  const uint32_t full_a = sa(bars), empty_a = sa(bars + NST);

  if (warp == 8) {
    // ---------------- producer ----------------
    const bool hint = (P.probe & 16) != 0;
    const bool pwait = ((P.probe >> PXW) & 1) != 0;
    const uint64_t pol = policy_evict_first();
    const int G = (int)gridDim.x;
    int it = 0, u = (int)blockIdx.x, tnext = 0, nunits = 0, nslow = 0;
    long long spin = 0, first_x = 0;
    if (lane == 0) tnext = atomicAdd(P.rdy + 128, 1);
    unsigned* rq = s_rq;                 // known-complete quarters (bit e), lane 0 only
    if (lane == 0) { rq[0] = 0u; rq[1] = 0u; rq[2] = 0u; rq[3] = 0u; }
    // v2 pipeline (R-tail-0b): at the top of unit u the ticket for the next unit (requested one unit ago) is consumed and
    // the next unit's readiness load is issued; that load is consumed one unit later. v1 issued the load at the END of a
    // unit and used it at the START of the next, exposing one loaded round trip per unit boundary (R-tail-0).
    int pv = 0x40000000, pidx = -1;      // readiness value loaded one unit ago for the current unit (lane 0)
    bool waited = false;
    while (u < U) {
      const PXU w = px_decode<NCH>(P, S, s_ord, s_ub, D, u);
      if (lane == 0) {
        const int need = 256 * S.np[w.e];
        bool fresh = false;
        for (int q = w.s0 >> 2; q <= ((w.s1 - 1) >> 2); ++q) {
          if ((rq[q] >> w.e) & 1u) continue;
          const int idx = w.e * 4 + q;
          int v = (idx == pidx) ? pv : px_ld(P.rdy + idx);
          if (v < need) {
            ++nslow;
            const long long t0 = clock64(), g0 = px_gt();
            while ((v = px_ld(P.rdy + idx)) < need) {
              __nanosleep(64);
              if (clock64() - t0 > (1ll << 28)) { atomicExch(P.err, 1); atomicExch(P.err + 1, 7); break; }
            }
            spin += px_gt() - g0;
          }
          rq[q] |= 1u << w.e;
          fresh = true;
        }
        if (pwait && !waited) { pdl_wait(); waited = true; fresh = true; }
        // before any new load / atomic is in flight, so the fence never waits on one
        if (fresh) asm volatile("fence.proxy.async.global;\n" ::: "memory");   // generic -> bulk-copy view, no MEMBAR
      }
      int un = 0;
      if ((P.probe >> PXV) & 1) {   // v2: next unit + its readiness load now, consumed one unit later
        if (lane == 0) {
          un = G + tnext;
          tnext = atomicAdd(P.rdy + 128, 1);
        }
        un = __shfl_sync(FULL, un, 0);
        if (un < U) {
          const PXU wn = px_decode<NCH>(P, S, s_ord, s_ub, D, un);
          if (lane == 0) {
            pidx = -1;
            for (int q = wn.s0 >> 2; q <= ((wn.s1 - 1) >> 2); ++q)
              if (!((rq[q] >> wn.e) & 1u)) { pidx = wn.e * 4 + q; pv = px_ld(P.rdy + pidx); break; }
          }
        }
      }
      __syncwarp();
      const uint8_t* wb = P.ws2 + ((size_t)(w.e * 32 + w.rb) * 16) * SW;
      const uint32_t xbytes = w.ntok * XROW;
      for (int s = w.s0; s < w.s1; ++s, ++it) {
        const int slot = it % NST;
        mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
        const uint32_t dst = ring_a + slot * STG, fb = full_a + 8 * slot;
        if (lane == 0) {
          s_tag[slot] = u;
          mbar_expect_tx(fb, SW + xbytes);
          const uint8_t* src = wb + (size_t)s * SW;
          if (hint) bulk_g2s_ef(dst, src, SW, fb, pol);
          else bulk_g2s(dst, src, SW, fb);
          bulk_g2s(dst + SW, P.inter + ((size_t)s * (8 * M3) + w.c0) * XST, xbytes, fb);
          if (first_x == 0) first_x = px_gt();
        }
        mbar_arrive(fb);   // 1 expect_tx + 32 lane arrivals
        __syncwarp();
      }
      ++nunits;
      if (!((P.probe >> PXV) & 1)) {   // v1 (as measured in R-tail-0): next unit + its readiness load after the copies
        if (lane == 0) {
          un = G + tnext;                         // ticket requested one unit ago
          tnext = atomicAdd(P.rdy + 128, 1);
        }
        un = __shfl_sync(FULL, un, 0);
        if (un < U) {
          const PXU wn = px_decode<NCH>(P, S, s_ord, s_ub, D, un);
          if (lane == 0) {
            pidx = -1;
            for (int q = wn.s0 >> 2; q <= ((wn.s1 - 1) >> 2); ++q)
              if (!((rq[q] >> wn.e) & 1u)) { pidx = wn.e * 4 + q; pv = px_ld(P.rdy + pidx); break; }
          }
        }
      }
      u = un;
    }
    {   // END: tag -1, completes without tx (1 + 32 arrivals)
      const int slot = it % NST;
      mbar_wait(empty_a + 8 * slot, ((it / NST) & 1) ^ 1, P.err);
      if (lane == 0) { s_tag[slot] = -1; mbar_arrive(full_a + 8 * slot); }
      mbar_arrive(full_a + 8 * slot);
    }
    if (!waited) pdl_wait();   // K2 completes only after K1: K3's wait then covers K1's writes (meta)
    if (mk != nullptr && lane == 0) { mk[3] = first_x; mk[4] = px_gt(); mk[5] = nunits; mk[6] = spin; mk[8] = nslow;
                                      mk[7] = (int)blockIdx.x; }
    return;
  }
  if (warp == 9) return;
  // ---------------- consumers (warps 0-7) ----------------
  int it = 0;
  while (true) {
    const int slot = it % NST;
    if (!mbar_wait_ok(full_a + 8 * slot, (it / NST) & 1, P.err)) break;
    const int u = s_tag[slot];
    if (u < 0) break;
    const PXU w = px_decode<NCH>(P, S, s_ord, s_ub, D, u);
    const int nch = (w.ntok + 7) >> 3;
    const float gs = s_gs[w.e];
    if (nch <= 1) {
      px_cold_item2<NST, STG>(P, smem, full_a, empty_a, it, w.s0, w.s1, gs, w.rb, w.c0, w.ntok);
    } else if constexpr (NCH >= 4) {
      if (nch >= 4) px_hc_item2<4, NST, STG>(P, smem, full_a, empty_a, it, w.s0, w.s1, gs, w.rb, w.c0, w.ntok);
      else if (nch == 3) px_hc_item2<3, NST, STG>(P, smem, full_a, empty_a, it, w.s0, w.s1, gs, w.rb, w.c0, w.ntok);
      else px_hc_item2<2, NST, STG>(P, smem, full_a, empty_a, it, w.s0, w.s1, gs, w.rb, w.c0, w.ntok);
    } else {
      px_hc_item2<2, NST, STG>(P, smem, full_a, empty_a, it, w.s0, w.s1, gs, w.rb, w.c0, w.ntok);
    }
  }
}

// K3: SS3's k3_kernel_ss (ss from y, 4-CTA cluster per token, DSMEM in rank order) + y self-reset + counter reset.
__device__ __forceinline__ void px_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ float px_peer(const float* p, int rank) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  uint32_t r;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];\n" : "=f"(v) : "r"(r) : "memory");
  return v;
}
__global__ void __launch_bounds__(256) k3_px(const Params P) {
  __shared__ int s_fc[8];
  __shared__ float s_fw[8];
  __shared__ float s_wp[8][8];
  __shared__ float s_q[8];
  __shared__ float s_fr[8];
  long long* mk = px_rec(P, 2);
  pdl_wait();
  if (mk != nullptr && threadIdx.x == 0) mk[1] = px_gt();
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int it = blockIdx.x;
  const int tok = it >> 2;
  if (tid < 8) {
    s_fc[tid] = __ldcg(P.meta + MH + tok * 8 + tid);
    s_fw[tid] = __ldg(P.wts + tok * 8 + tid);
  }
  if (blockIdx.x == 0 && tid < 130) P.rdy[tid] = 0;
  __syncthreads();
  const int col = (it & 3) * 1024 + 4 * tid;
  const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 v[8];
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int c = s_fc[s];
    v[s] = (c >= 0) ? __ldcg(reinterpret_cast<const float4*>(P.y + (size_t)c * KDIM + col)) : z4;
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const int c = s_fc[s];
    if (c >= 0) *reinterpret_cast<float4*>(P.y + (size_t)c * KDIM + col) = z4;   // self-reset for the next call
  }
  const float4 w = __ldg(reinterpret_cast<const float4*>(P.nw + col));
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    float z, q = 0.f;
    z = bf16r(v[s].x) * w.x; q += z * z;  z = bf16r(v[s].y) * w.y; q += z * z;
    z = bf16r(v[s].z) * w.z; q += z * z;  z = bf16r(v[s].w) * w.w; q += z * z;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    if (lane == 0) s_wp[warp][s] = q;
  }
  __syncthreads();
  if (tid < 8) {
    float q = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) q += s_wp[i][tid];
    s_q[tid] = q;
  }
  px_cluster_sync();
  if (tid < 8) {
    float ss = 0.f;
#pragma unroll
    for (int r = 0; r < 4; ++r) ss += px_peer(&s_q[tid], r);
    s_fr[tid] = (s_fc[tid] >= 0) ? rsqrtf(ss * (1.f / 4096.f) + 1e-8f) * s_fw[tid] : 0.f;
  }
  __syncthreads();
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
  px_cluster_sync();
  if (mk != nullptr && threadIdx.x == 0) mk[2] = px_gt();
}
"""

K1_HOOK = r"""  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);
  if ((P.probe >> PXB) & 1) {   // PX: readiness signal once every warp of this CTA is done
    long long* mk = px_rec(P, 0);
    __syncthreads();
    if (threadIdx.x == 0) {
      if (mk != nullptr) { mk[0] = px_smid(); mk[2] = px_gt(); }
      px_k1_signal(P, S, s_pr);
      if (mk != nullptr) mk[3] = px_gt();
    }
  }
}
"""

HOST = r"""
template <int NCH, int NST>
static void launch_px(const k3::Params& p, int G, int M, int mask) {
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  constexpr int SM = smem_bytes36<NCH, NST>();
  if (mask & 1) {
    k3::k1_kernel36<NCH, NST><<<G, 320, SM, st>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = st;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  if (mask & 2) {
    cfg.gridDim = dim3(G); cfg.blockDim = dim3(320); cfg.dynamicSmemBytes = SM;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, k3::k2_px<NCH, NST>, p));
  }
  if (mask & 4) {
    cudaLaunchAttribute a2[2];
    a2[0] = at[0];
    a2[1].id = cudaLaunchAttributeClusterDimension;
    a2[1].val.clusterDim.x = 4; a2[1].val.clusterDim.y = 1; a2[1].val.clusterDim.z = 1;
    cudaLaunchConfig_t c2 = cfg;
    c2.attrs = a2; c2.numAttrs = 2;
    c2.gridDim = dim3(4 * M); c2.blockDim = dim3(256); c2.dynamicSmemBytes = 0;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&c2, k3::k3_px, p));
  }
}

std::vector<int64_t> pxattrs() {
  constexpr int SM = smem_bytes36<4, 3>();
  std::vector<int64_t> r;
  auto one = [&](const void* f, int nt, int sm) {
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncAttributes fa;
    int occ = 0;
    if (cudaFuncGetAttributes(&fa, f) != cudaSuccess) { r.insert(r.end(), {-1, -1, -1, -1}); return; }
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, nt, sm) != cudaSuccess) occ = -1;
    r.insert(r.end(), {(int64_t)fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)(fa.sharedSizeBytes + sm), (int64_t)occ});
  };
  one((const void*)k3::k1_kernel36<4, 3>, 320, SM);
  one((const void*)k3::k2_px<4, 3>, 320, SM);
  one((const void*)k3::k3_px, 256, 0);
  return r;
}

void pxrun(int64_t D, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
           torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
           torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
           torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor rdy,
           torch::Tensor marks, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t probe) {
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
  p.xw = nullptr;
  p.xw2 = nullptr;
  p.hot_t = (int)(hot_t % 1000);
  p.hot_w4 = (hot_t >= 1000) ? (int)((hot_t / 1000) % 100) : 8;
  p.hot_mode = (int)(hot_t / 100000);
  p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe | (1 << PXB_HOST);
  p.rdy = rdy.data_ptr<int>();
  p.pxmk = marks.numel() >= 3 * 1024 * 16 ? reinterpret_cast<long long*>(marks.data_ptr()) : nullptr;
  p.pxd = (int)D;
  TORCH_CHECK(D == 2 || D == 4 || D == 8 || D == 16, "px: D in {2, 4, 8, 16}");
  TORCH_CHECK(M >= 1 && M <= 48, "px: HK2 serves M <= 48");
  launch_px<4, 3>(p, (int)G, (int)M, (int)mask);
}
"""

CPP = r"""
#include <torch/extension.h>
#include <vector>
void run3(int64_t vid, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
          torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
          torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
          torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor xw,
          torch::Tensor xw2, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t G3, int64_t probe);
std::vector<int64_t> attrs3v(int64_t vid);
std::vector<int64_t> pxattrs();
void pxrun(int64_t D, int64_t mask, torch::Tensor x, torch::Tensor ids, torch::Tensor wts, torch::Tensor ws13,
           torch::Tensor ws2, torch::Tensor gsc13, torch::Tensor gsc2, torch::Tensor nw, torch::Tensor h,
           torch::Tensor inter, torch::Tensor y, torch::Tensor meta, torch::Tensor cnt, torch::Tensor err,
           torch::Tensor out, torch::Tensor sink, torch::Tensor ss, torch::Tensor cnt2, torch::Tensor rdy,
           torch::Tensor marks, int64_t hot_t, int64_t M, int64_t lo, int64_t G, int64_t probe);
"""

VARIANTS = [(1, 3, 2, 2, 4)]   # vid 3 only (run3 index 0)


def _once(src, old, new, name, region=None):
    a, b = (0, len(src)) if region is None else (src.index(region[0]), src.index(region[1], src.index(region[0])))
    n = src[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"px: anchor '{name}' found {n} times")
    i = src.index(old, a)
    return src[:i] + new + src[i + len(old):]


def lab_source():
    import k2a_v37 as D
    cases, attrs = D.variant_cases(VARIANTS)
    src = D.KERNEL_SRC_V3.replace("@CASES3@", cases).replace("@ATTRS3@", attrs).replace("@WGSPECS@", D._wg_specs())
    src = _once(src, "  int M, lo, probe;\n};", "  int M, lo, probe;\n  int* rdy; long long* pxmk; int pxd;   // PX lab\n};\n"
                f"constexpr int PXB = {PXB};   // PX lab probe bits\nconstexpr int PXW = {PXW};\nconstexpr int PXV = {PXV};\n", "params")
    src = _once(src, "p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;",
                "p.M = (int)M; p.lo = (int)lo; p.probe = (int)probe;\n  p.rdy = nullptr; p.pxmk = nullptr; p.pxd = 4;",
                "run3 fields")
    zero = ("  if (MODE == 1) {\n    // y rows [0, R) and ss[0, R) -> 0 while the first stages are in flight (K2 accumulates "
            "into both after its wait)\n")
    src = _once(src, zero, "  if (MODE == 1 && !((P.probe >> PXB) & 1)) {   // PX: y is self-reset by K3\n"
                + zero.split("\n", 1)[1], "gemm36 zeroing", ("__device__ void gemm36(", "k1_kernel36(const Params P) {"))
    src = _once(src, "  gemm36<NCH, NST, 1>(P, S, s_list, smem, bars, false, s_pr);\n}\n", K1_HOOK, "k1_36 hook")
    src = _once(src, "  pdl_launch();\n", "  pdl_launch();\n  if (P.pxmk != nullptr && threadIdx.x == 0 && blockIdx.x < 1024) P.pxmk[blockIdx.x * 16 + 1] = px_gt();\n",
                "k1_36 entry mark", ("k1_kernel36(const Params P) {", "k2_kernel36(const Params P) {"))
    # PX device code before the v3.6 kernels (needs Params, Route, cold/hc consumers, setup_bars, route); k3_px after K3
    dev = DEV
    head, k2part = dev.split("// K2: dynamic finalize-free W2 queue.", 1)
    src = _once(src, "// K1 / K2 of the HK2 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.\n",
                head + "// K1 / K2 of the HK2 variants (cold NC1, hot up to 8*NCH tokens per pass, NST ring stages); 2 CTAs/SM.\n",
                "px device helpers")
    src = _once(src, "}  // namespace k3\n", "// K2: dynamic finalize-free W2 queue." + k2part + "}  // namespace k3\n",
                "px kernels")
    src = _once(src, "void run3(int64_t vid", HOST.replace("PXB_HOST", str(PXB)) + "\nvoid run3(int64_t vid", "px host")
    return src, CPP


def build(name="px_moe_lab", verbose=False):
    from torch.utils.cpp_extension import load_inline
    cuda, cpp = lab_source()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["run3", "attrs3v", "pxattrs", "pxrun"],
                      extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills",
                                         "-gencode=arch=compute_90a,code=sm_90a"], verbose=verbose)
    return mod, time.time() - t0


def px_state(st, torch):
    """PX buffers of its own (y must start at zero and is self-reset; rdy / tickets reset by K3); weights shared."""
    dev = st["ws13"].device
    z = lambda t: torch.zeros_like(t)  # noqa: E731
    return dict(h=z(st["h"]), inter=z(st["inter"]), y=z(st["y"]), meta=z(st["meta"]), cnt=z(st["cnt"]),
                err=torch.zeros((4,), dtype=torch.int32, device=dev), sink=z(st["sink"]), ss=z(st["ss"]),
                cnt2=z(st["cnt2"]), rdy=torch.zeros((256,), dtype=torch.int32, device=dev),
                nomk=torch.zeros((1,), dtype=torch.int64, device=dev))


def px_call(mod, st, ps, data, D, hot_code, probe, marks=None, mask=7):
    import torch
    x = data["hidden_states"].contiguous()
    ids = data["topk_ids"].to(torch.int32).contiguous()
    wts = data["topk_weights"].to(torch.float32).contiguous()
    m = x.shape[0]
    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
    mod.pxrun(D, mask, x, ids, wts, st["ws13"], st["ws2"], st["gsc13"], st["gsc2"], st["nw"], ps["h"], ps["inter"], ps["y"],
              ps["meta"], ps["cnt"], ps["err"], out, ps["sink"], ps["ss"], ps["cnt2"], ps["rdy"],
              ps["nomk"] if marks is None else marks, hot_code, m, data["config"]["local_start"], st["sms"] * 2, probe)
    return out


def device_source():
    """The CUDA device code only (no torch host code): for the local sm_90a ptxas / SASS check (tools/px_sass.py)."""
    src, _ = lab_source()
    head = src[:src.index("namespace k3 {")]
    head = "\n".join(l for l in head.splitlines() if "torch/" not in l and "ATen" not in l and "c10/" not in l)
    body = src[src.index("namespace k3 {"):src.index("}  // namespace k3\n") + len("}  // namespace k3\n")]
    inst = ("template __global__ void k3::k1_kernel36<4, 3>(const k3::Params P);\n"
            "template __global__ void k3::k2_kernel36<4, 3>(const k3::Params P);\n"
            "template __global__ void k3::k2_px<4, 3>(const k3::Params P);\n")
    return head + "\n" + body + inst


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "px_moe_lab_gen.cu")
    out.write_text(device_source(), encoding="utf-8", newline="\n")
    print(f"wrote {out}")

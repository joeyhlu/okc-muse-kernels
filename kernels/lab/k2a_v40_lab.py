"""K2A v4.0 lab (T = 256 / 128 consumer anatomy + hot-loop fixes): K2A v3.7 as shipped in k3_v18/v20
(kernels/muse_moe_w4a16/k2a_v37.py, KERNEL_SRC_V3) with hot_p1's patches (kernels/lab/hot_p1_lab.py: clock / role
marks, producer bits) plus the v4.0 patches below, emitted once per COPY in its own namespace `v40_<name>`.

LAB FILE. Box runs only, through tools/box/k2a_v40.py. Never submit it, and never import it from a scored file.

Per copy (compile-time):
  HP_LAB    hot_p1 consumer mask (0 here: real math).
  V40_PROF  1 = per-warp phase counters (clock deltas, smem atomics by lane 0) for every warp class of the v3.3 gemm
            (HOTCAP hot role, its cold role, producer, epilogue warp), dumped per (kernel, CTA, warp) into Params.v40p:
            [((k * 1024 + cta) * 10 + warp) * 8 + c], k 0 = K1, 1 = K2. Counters c:
              hot consumer  0 full-wait  1 LDS+decode+wgmma issue  2 wait_group<1> stall  3 drain (wait_group<0>) + release
                            4 flush  5 #stages  6 #items  7 gemm total
              cold consumer 0 full-wait  1 LDS+decode+mma  3 release  4 flush  5 #stages  6 #items  7 total
              producer      0 empty-wait  1 issue  2 lazy griddepcontrol.wait  3 tail  5 #stages  7 total
              epilogue      0 bar.sync wait  1 work  6 #split items  7 total
            gemm37 (vid 5) records only counter 7 (per-warp totals).
  V40_VAR   bits (hot_item of the HOTCAP role only; cold role, gemm37 and all numerics unchanged):
            1 LEAN  wgmma descriptor = {lo + offset, hi} formed inside the asm (32-bit add on the low word, constant high
                    word) instead of a 64-bit add per wgmma: 1 IADD + 1 R2UR per wgmma instead of IADD3 + IMAD.X + 2 MOV +
                    2 R2UR. The low word's 14-bit address field never carries (smem < 228 KB, offsets <= 0xF0), so the
                    descriptor values are identical.
            2 DEFER no wait_group<0> per stage: wait_group<1> before every kb, the previous stage's slot is released
                    after the wait before kb 1 (its last group is then complete); wait_group<0> + release once per item.
                    Same wgmmas in the same order: bitwise-identical accumulators. (ptxas SERIALIZES it: DEPBAR per HGMMA.)
            4 FLUSH2 hot_item flush with affine addresses: one base pointer per lane, compile-time offsets per
                    (chunk, hh) (xw_elem(n) = xw_elem(2t) + 2048 (n >> 3) + 8 hh); same stores / atomics: bitwise.
            8 COLD1 cold role (NC 2): stage loop specialised on nch (1 / 2 chunks, no dead predicated chunk, no runtime
                    probe paths): same mma order, bitwise.
           16 EARLY with COLD1, nch 1: the stage's codes, scales and X fragments into registers, slot released, then
                    decode + mma (v3.1's V3_EARLY idea for NC 2): same mma order, bitwise.
           32 DUAL  with COLD1, nch 1: kb 0-1 and kb 2-3 on two accumulator chains (acc[0], acc[1]), summed before the
                    flush: FP32 summation order changes (not bitwise; tolerance checks).
           64 HANDOFF (with FLUSH2) per-(expert, pass) K1 -> K2 hand-off instead of K2's grid-wide griddepcontrol.wait:
                    Params.rdy[e * 64 + j] counts K1 warps done with pass j of expert e's SwiGLU rows (whole items: each of
                    the 8 consumer warps +1 after its xw2 / inter stores; split items: the finalizing epilogue warp +8), target
                    64 row blocks x 8 = 512; rdy[2048] counts K1 consumer warps done zeroing y / ss (target 8 x grid). Release
                    = __syncwarp + __threadfence + atomicAdd by lane 0 (the shipped split-item pattern). K2's producer lane 0
                    polls with ld.acquire.gpu (bounded, error flag err[0] / err[3], no trap) before the first X copy of a pass,
                    then fence.proxy.async; the first NST stages' weight copies are issued before the first poll (the shipped
                    deferral). K2 never takes griddepcontrol.wait; K3 (after its wait on K2) zeroes rdy for the next call.
          128 WSK   Stream-K ranges of the v3.3 gemm (both roles, K1 and K2) cut at equal *byte-model weight*
                    (pick_role's model: a stage of a pass with n token rows weighs 36 + n (hot: n rounded up to 8)) instead
                    of equal stage counts: a 64-token hot stage costs more than a 32-token one. Only cut points move.
Runtime (all copies, pick_role): hot_w4 >= 50 = fine role-split weight (hot_w4 - 50) / 32 (ship: 8 / 8 = 1.0); K2 uses
  Params.hot_mode as its own weight when > 0 (run3 hot_t = hot_t + 1000 * w4(K1) + 100000 * w4(K2)).
Waits stay v3.7's bounded mbar_wait (R9); no new wait loop.
"""
import hashlib
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hot_p1_lab as H  # noqa: E402

V40_LEAN, V40_DEFER, V40_FLUSH2, V40_COLD1, V40_EARLY, V40_DUAL = 1, 2, 4, 8, 16, 32
V40_HANDOFF = 64   # needs FLUSH2 (the hot whole-item signal lives in FLUSH2's flush)
V40_WSK = 128      # byte-weighted Stream-K ranges inside a role
V40_WSKP = 256     # WSK weights A + B * rows from Params.rdy[3000 + (MODE - 1) * 4 + (cold ? 2 : 0) + {0, 1}] (runtime)
# name -> (HP_LAB, V40_PROF, V40_VAR, vids built)
COPIES = {
    "base":    (0, 0, 0, (2, 5)),
    "lf":      (0, 0, V40_LEAN | V40_FLUSH2, (2,)),
    "lp":      (0, 0, V40_LEAN | V40_FLUSH2 | V40_WSK | V40_WSKP, (2,)),   # = v4.0 + v4.2 with runtime WSK weights
}
# earlier batches also built: prof (PROF), lean, c1, ce, cd, alle, alled, profe, ho, hoe, lw, aw, hw (see notes/lab.md 10-26 K2A v4.0 rows)
NWARP = 10
NCNT = 8


def _wg_specs2():
    """wgmma m64nNk16 RS with the descriptor formed in the asm from two 32-bit halves (N = 16, 32, 48, 64)."""
    out = ["template <int N> __device__ __forceinline__ void wgmma_rs2(float* d, uint32_t a0, uint32_t a1, uint32_t a2,"
           " uint32_t a3, uint32_t lo, uint32_t hi);"]
    for n in (16, 32, 48, 64):
        k = n // 2
        regs = ",".join("%%%d" % i for i in range(k))
        dl = ", ".join('"+f"(d[%d])' % i for i in range(k))
        out.append(
            "template <> __device__ __forceinline__ void wgmma_rs2<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
            "uint32_t a3, uint32_t lo, uint32_t hi) {\n"
            "  asm volatile(\"{\\n.reg .pred p;\\n.reg .b64 dd;\\nmov.b64 dd, {%%%d, %%%d};\\nsetp.ne.b32 p, 1, 0;\\n"
            "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%%%d,%%%d,%%%d,%%%d}, dd, p, 1, 1, 0;\\n}\\n\"\n"
            "    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(lo), \"r\"(hi));\n}"
            % (n, k + 4, k + 5, n, regs, k, k + 1, k + 2, k + 3, dl))
    return "\n".join(out)


HELPERS = r"""
// ======== k2a_v40 lab helpers (kernels/lab/k2a_v40_lab.py) ========
__shared__ uint32_t v40s[10 * 8];
__device__ __forceinline__ uint32_t v40_clk() {
  uint32_t c;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
  return c;
}
// counter k of this warp += now - t; t = now (lane 0 adds; smem atomic, no return value used)
#define V40_ACC(k, t) do { if constexpr (V40_PROF != 0) { const uint32_t _n = v40_clk(); \
  if ((threadIdx.x & 31) == 0) atomicAdd(&v40s[(threadIdx.x >> 5) * 8 + (k)], _n - (t)); (t) = _n; } } while (0)
#define V40_CNT(k) do { if constexpr (V40_PROF != 0) { \
  if ((threadIdx.x & 31) == 0) atomicAdd(&v40s[(threadIdx.x >> 5) * 8 + (k)], 1u); } } while (0)
__device__ __forceinline__ void v40_init() {
  if constexpr (V40_PROF != 0) { if (threadIdx.x < 80) v40s[threadIdx.x] = 0u; }
}
__device__ __forceinline__ void v40_dump(const Params& P, int k) {   // each warp after its own gemm returned
  if constexpr (V40_PROF != 0) {
    if (P.v40p == nullptr || blockIdx.x >= 1024) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncwarp();
    if (lane < 8) P.v40p[(((size_t)k * 1024 + blockIdx.x) * 10 + warp) * 8 + lane] = v40s[warp * 8 + lane];
  }
}
"""
HO_HELPERS = r"""
// v4.0 HANDOFF: per-(expert, pass) readiness of K1's SwiGLU rows for K2 (lab)
__device__ __forceinline__ int v40_ld_acq(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void v40_poll(const int* p, int target, int* err) {   // bounded (R9), flag, no trap
  if (v40_ld_acq(p) >= target) return;
  const long long t0 = clock64();
  while (v40_ld_acq(p) < target) {
    __nanosleep(64);
    if (clock64() - t0 > WAIT_BUDGET) { atomicExch(err, 1); atomicExch(err + 3, 1); return; }
  }
}
// K2 producer lane 0, before an X copy of (expert, pass) key: K1's y / ss zeroing (once), then the key's 512 warps
__device__ __forceinline__ void v40_ready(const Params& P, int key, int& rkey, bool& zok) {
  if (!zok) { v40_poll(P.rdy + 2048, 8 * (int)gridDim.x, P.err); zok = true; }
  if (key != rkey) {
    v40_poll(P.rdy + key, 512, P.err);
    asm volatile("fence.proxy.async;\n" ::: "memory");   // generic-proxy K1 stores -> async-proxy bulk copy reads
    rkey = key;
  }
}
// all lanes, after this warp's stores: release n counts on rdy[key] (cumulative over the warp's stores via __syncwarp)
__device__ __forceinline__ void v40_signal(const Params& P, int key, int n) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) { __threadfence(); atomicAdd(P.rdy + key, n); }
}
"""
WSK_HELPERS = r"""
// v4.0 WSK: stage index of this role's sequence where the byte-model weight prefix reaches T
// weight of a stage of pass jj = A + B * rows (rows = ntok, hot: rounded up to 8); wm = {A, B} (runtime, WSKP) or null (36, 1)
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ long long v40_wpass(int c, int jj, const int* wm) {
  constexpr int TP = HOT ? 64 : 8 * NC;
  const int nt = min(TP, c - jj * TP);
  const int rows = HOT ? ((nt + 7) & ~7) : nt;
  return wm ? (long long)wm[0] + (long long)wm[1] * rows : 36 + rows;
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ int v40_wpos(const Route& S, long long T, const int* wm) {
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    const int np = S.np[e], c = S.cnt[e];
    long long wrb = 0;
    for (int jj = 0; jj < np; ++jj) wrb += 16 * v40_wpass<HOT, NC, NRB>(c, jj, wm);
    if (T >= wrb * NRB) { T -= wrb * NRB; continue; }
    const int rb = (int)(T / wrb);
    T -= rb * wrb;
    for (int jj = 0; jj < np; ++jj) {
      const long long w = v40_wpass<HOT, NC, NRB>(c, jj, wm);
      if (T >= 16 * w) { T -= 16 * w; continue; }
      return S.sb[e] + (rb * np + jj) * 16 + (int)(T / w);
    }
    return S.sb[e] + (rb + 1) * np * 16;
  }
  return S.sb[32];
}
template <bool HOT, int NC, int NRB>
__device__ __forceinline__ void v40_wrange(const Route& S, int cta, int ncta, int& beg, int& end, const int* wm = nullptr) {
  long long W = 0;
  for (int e = 0; e < 32; ++e) {
    if (S.sb[e + 1] == S.sb[e]) continue;
    for (int jj = 0; jj < S.np[e]; ++jj) W += 16LL * NRB * v40_wpass<HOT, NC, NRB>(S.cnt[e], jj, wm);
  }
  beg = v40_wpos<HOT, NC, NRB>(S, W * cta / ncta, wm);
  end = (cta + 1 == ncta) ? S.sb[32] : v40_wpos<HOT, NC, NRB>(S, W * (cta + 1) / ncta, wm);
}
"""
COLD_HELPERS = r"""
// COLD1: one kb of this warp's tile for NCH (compile-time) chunks; a0 = chunk 0's accumulator, a1 = chunk 1's
template <int NCH>
__device__ __forceinline__ void v40_ckb(const uint4 w, const uint32_t sw2, int par, const __nv_bfloat16* xrow, int kloc,
                                        int t, float* a0, float* a1) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) { ag[jj] = bmul2(ag[jj], Sg); a8[jj] = bmul2(a8[jj], S8); }
#pragma unroll
  for (int q = 0; q < NCH; ++q) {
    float* acc = q ? a1 : a0;
    const uint4* xp = reinterpret_cast<const uint4*>(xrow + q * 8 * XST + kloc + 16 * t);
    const uint4 x0 = xp[0], x1 = xp[1];
    mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
    mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
    mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
    mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
  }
}
__device__ __forceinline__ void v40_ckb_r(const uint4 w, const uint32_t sw2, int par, const uint4 x0, const uint4 x1,
                                          float* acc) {
  const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
  const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
  uint32_t ag[8], a8[8];
  deq4(w.x, ag); deq4(w.y, ag + 4); deq4(w.z, a8); deq4(w.w, a8 + 4);
#pragma unroll
  for (int jj = 0; jj < 8; ++jj) { ag[jj] = bmul2(ag[jj], Sg); a8[jj] = bmul2(a8[jj], S8); }
  mma16816(acc, ag[0], a8[0], ag[1], a8[1], x0.x, x0.y);
  mma16816(acc, ag[2], a8[2], ag[3], a8[3], x0.z, x0.w);
  mma16816(acc, ag[4], a8[4], ag[5], a8[5], x1.x, x1.y);
  mma16816(acc, ag[6], a8[6], ag[7], a8[7], x1.z, x1.w);
}
// COLD1 stage loop of one cold item (NC 2 role), NCH = the item's 8-token chunks (1 or 2)
template <int NCH, int NST, int STG, bool EARLY, bool DUAL>
__device__ __forceinline__ void v40_cold_loop(const Params& P, const uint8_t* ring, uint32_t full_a, uint32_t empty_a,
                                              int& it_ref, int s0, int s1, float (*acc)[4], uint32_t& vt) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  int it = it_ref;
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    V40_ACC(0, vt);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;
    uint4 wv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) wv[i] = wp[i * 32];
    const uint32_t sv0 = sp[0], sv1 = sp[32];
    if constexpr (EARLY && NCH == 1) {
      uint4 x0[4], x1[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint4* xp = reinterpret_cast<const uint4*>(xrow + i * 64 + 16 * t);
        x0[i] = xp[0]; x1[i] = xp[1];
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v40_ckb_r(wv[i], (i < 2) ? sv0 : sv1, i & 1, x0[i], x1[i], (DUAL && i >= 2) ? acc[1] : acc[0]);
      V40_ACC(1, vt);
    } else {
#pragma unroll
      for (int i = 0; i < 4; ++i)
        v40_ckb<NCH>(wv[i], (i < 2) ? sv0 : sv1, i & 1, xrow, i * 64, t,
                     (NCH == 1 && DUAL && i >= 2) ? acc[1] : acc[0], acc[1]);
      V40_ACC(1, vt);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
    }
    V40_ACC(3, vt);
    V40_CNT(5);
  }
  if constexpr (DUAL && NCH == 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc[0][i] += acc[1][i]; acc[1][i] = 0.f; }
  }
  it_ref = it;
}
"""

# hot_item's stage loop, rewritten (V40_VAR LEAN / DEFER; PROF marks). Replaces the text from OLD_A to OLD_B (inclusive)
# inside hot_item.
HOT_OLD_A = "  for (int s = s0; s < s1; ++s, ++it) {\n    const int slot = it % NST;\n"
HOT_OLD_B = "    __syncwarp();\n    if (lane == 0) mbar_arrive(empty_a + 8 * slot);\n  }\n  it_ref = it;\n"
HOT_NEW = r"""  uint32_t vt = V40_PROF ? v40_clk() : 0u;
  constexpr bool LEAN = (V40_VAR & 1) != 0, DEFER = (V40_VAR & 2) != 0;
  int pslot = -1;   // DEFER: previous stage's slot, released once its last commit group is complete
  for (int s = s0; s < s1; ++s, ++it) {
    const int slot = it % NST;
    mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);
    V40_ACC(0, vt);
    const uint8_t* st = ring + slot * STG;
    const uint4* wp = reinterpret_cast<const uint4*>(st + warp * 2048) + lane;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + warp * 256) + lane;
    const uint64_t d0 = wg_desc_sa(sa(st + SW), 4096u);
    const uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);
    uint32_t hb[2][16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 w = wp[i * 32];
      const uint32_t sw2 = sp[(i >> 1) * 32];
      const int par = i & 1;
      const uint32_t Sg = __byte_perm(sw2, 0u, 0x4040u + 0x0202u * par) * 16u + 0x75807580u;
      const uint32_t S8 = __byte_perm(sw2, 0u, 0x4141u + 0x0202u * par) * 16u + 0x75807580u;
      if (i >= 2 || DEFER) {   // DEFER: kb 0 / 1 wait for the previous stage's kb 2 / 3 (a no-op when none is pending)
        V40_ACC(1, vt);
        wg_wait<1>();
        V40_ACC(2, vt);
      }
      if constexpr (DEFER) {
        if (i == 1 && pslot >= 0) {
          __syncwarp();
          if (lane == 0) mbar_arrive(empty_a + 8 * pslot);
          pslot = -1;
        }
      }
      uint32_t* h = hb[i & 1];
      deq4(w.x, h); deq4(w.y, h + 4); deq4(w.z, h + 8); deq4(w.w, h + 12);
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) { h[jj] = bmul2(h[jj], Sg); h[8 + jj] = bmul2(h[8 + jj], S8); }
      wg_fence();
      if constexpr (LEAN) {
        wgmma_rs2<N>(hacc, h[0], h[8], h[1], h[9], dlo + 64 * i, dhi);
        wgmma_rs2<N>(hacc, h[2], h[10], h[3], h[11], dlo + 64 * i + 16, dhi);
        wgmma_rs2<N>(hacc, h[4], h[12], h[5], h[13], dlo + 64 * i + 32, dhi);
        wgmma_rs2<N>(hacc, h[6], h[14], h[7], h[15], dlo + 64 * i + 48, dhi);
      } else {
        wgmma_rs<N>(hacc, h[0], h[8], h[1], h[9], d0 + 64 * i);
        wgmma_rs<N>(hacc, h[2], h[10], h[3], h[11], d0 + 64 * i + 16);
        wgmma_rs<N>(hacc, h[4], h[12], h[5], h[13], d0 + 64 * i + 32);
        wgmma_rs<N>(hacc, h[6], h[14], h[7], h[15], d0 + 64 * i + 48);
      }
      wg_commit();
    }
    V40_ACC(1, vt);
    if constexpr (DEFER) {
      pslot = slot;
    } else {
      wg_wait<0>();
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * slot);
      V40_ACC(3, vt);
    }
    V40_CNT(5);
  }
  if constexpr (DEFER) {
    wg_wait<0>();
    if (pslot >= 0) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty_a + 8 * pslot);
    }
    V40_ACC(3, vt);
  }
  it_ref = it;
"""

FLUSH_A = "  // ---- flush (same as the cold path's, 8-token chunks q < ceil(ntok / 8)) ----\n"
FLUSH_B = "  V40_ACC(4, vt);\n  V40_CNT(6);\n  if (!whole)"
FLUSH2 = r"""  const bool whole = (s0 == 0 && s1 == 16);
  if constexpr ((V40_VAR & 4) != 0) {
    // FLUSH2: affine addresses (one base per lane, compile-time offsets per (q, hh)); same values / stores / atomics
    const int tile = rb * 8 + warp;
    const float gs = __ldg(((MODE == 1) ? P.gsc13 : P.gsc2) + e);
    if constexpr (MODE == 1) {
      const int col = 8 * tile + g;
      if (whole) {
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
      if (whole) {
        const float w0 = __ldg(P.nw + 16 * tile + g), w1 = __ldg(P.nw + 16 * tile + g + 8);
#pragma unroll
        for (int q = 0; q < N / 8; ++q) {
          if (q * 8 < ntok) {
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
    }
  } else {
"""

HOT_REGION = ("__device__ __forceinline__ void hot_item(", "// HM 0: cold items (mma.sync)")
GEMM_REGION = ("__device__ void gemm(const Params& P, Route& S,", "__device__ __forceinline__ void setup_bars(")
K1_REGION = ("k1_kernel(const Params P) {", "// K2: route + W2.")
K2_REGION = ("k2_kernel(const Params P) {", "// P0 (v3.3)")
K137_REGION = ("k1_kernel37(const Params P) {", "k2_kernel37(const Params P) {")
K237_REGION = ("k2_kernel37(const Params P) {", "// K1: route + W13 (+ SwiGLU).")

# (name, region, old, new): old must occur exactly once in the region (applied after hot_p1's PATCHES)
PATCHES = [
    ("v40 constants", None, "constexpr int HP_LAB = @HPLAB@;\n",
     "constexpr int HP_LAB = @HPLAB@;\nconstexpr int V40_PROF = @V40PROF@;\nconstexpr int V40_VAR = @V40VAR@;\n"
     "constexpr bool HANDOFF = (V40_VAR & 64) != 0;\nconstexpr bool WSK = (V40_VAR & 128) != 0;\n"),
    ("v40 params", None, "  int hp; long long* hpclk;",
     "  int hp; long long* hpclk;\n  unsigned* v40p;   // k2a_v40 lab: per-warp phase counters\n"
     "  int* rdy;   // v4.0 HANDOFF: [32 experts][64 passes] K1 readiness, [2048] K1 y / ss zeroing count"),
    ("v40 helpers", None, "// v3.3 warm-expert item (all of its stages s0..s1)",
     HELPERS + HO_HELPERS + WSK_HELPERS + "@WGSPECS2@\n" + COLD_HELPERS + "// v3.3 warm-expert item (all of its stages s0..s1)"),
    ("hot flush end", HOT_REGION,
     "  if (!whole) asm volatile(\"bar.arrive %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n}\n",
     "  V40_ACC(4, vt);\n  V40_CNT(6);\n"
     "  if (!whole) asm volatile(\"bar.arrive %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n}\n"),
    # ---- producer ----
    ("prod init", GEMM_REGION, "    int it = 0, e = 0;\n    bool waited = !pdl",
     "    uint32_t vt = V40_PROF ? v40_clk() : 0u;\n    int it = 0, e = 0;\n    bool waited = !pdl"),
    ("prod pdl", GEMM_REGION,
     "        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
     "          pdl_wait();\n",
     "        if (!waited && nd > 0 && it >= NST) {   // a deferred slot is about to be reused: take the wait now\n"
     "          V40_ACC(1, vt);\n          pdl_wait();\n          V40_ACC(2, vt);\n"),
    ("prod empty", GEMM_REGION,
     "        mbar_wait(empty_a + 8 * slot, par, P.err);\n        const uint32_t dst = ring_a + slot * STG;\n",
     "        V40_ACC(1, vt);\n        mbar_wait(empty_a + 8 * slot, par, P.err);\n        V40_ACC(0, vt);\n"
     "        V40_CNT(5);\n        const uint32_t dst = ring_a + slot * STG;\n"),
    ("prod tail a", GEMM_REGION, "    if (waited && !pdl) pdl_wait();",
     "    V40_ACC(1, vt);\n    if (waited && !pdl) pdl_wait();"),
    ("prod tail b", GEMM_REGION, "    return;\n  }\n\n  if (warp == 9) {\n",
     "    V40_ACC(3, vt);\n    return;\n  }\n\n  if (warp == 9) {\n    uint32_t vt = V40_PROF ? v40_clk() : 0u;\n"),
    # ---- epilogue warp ----
    ("epi bar", GEMM_REGION,
     "          asm volatile(\"bar.sync %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n          ++npart;\n",
     "          V40_ACC(1, vt);\n"
     "          asm volatile(\"bar.sync %0, 288;\\n\" :: \"r\"(2 + npart) : \"memory\");\n"
     "          V40_ACC(0, vt);\n          V40_CNT(6);\n          ++npart;\n"),
    ("epi end", GEMM_REGION, "    return;\n  }\n\n  // ---------------- consumers",
     "    V40_ACC(1, vt);\n    return;\n  }\n\n  // ---------------- consumers"),
    # ---- cold consumer ----
    ("cold vt", GEMM_REGION, "    float acc[QMAX][4], acc2[NC][4];   // acc2: second mma chain (NC == 1 only)\n",
     "    uint32_t vt = V40_PROF ? v40_clk() : 0u;\n"
     "    float acc[QMAX][4], acc2[NC][4];   // acc2: second mma chain (NC == 1 only)\n"),
    ("cold full", GEMM_REGION,
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n      const uint8_t* st = ring + slot * STG;\n",
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n      V40_ACC(0, vt);\n"
     "      const uint8_t* st = ring + slot * STG;\n"),
    ("cold release", GEMM_REGION,
     "      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(empty_a + 8 * slot);\n    }\n    }\n    // ---- flush ----\n",
     "      }\n      V40_ACC(1, vt);\n      __syncwarp();\n      if (lane == 0) mbar_arrive(empty_a + 8 * slot);\n"
     "      V40_ACC(3, vt);\n      V40_CNT(5);\n    }\n    }\n    // ---- flush ----\n"),
    ("cold flush end", GEMM_REGION,
     "    if (!(s0 == 0 && s1 == 16)) ++npart;\n    x += s1 - s0;\n  }\n  }\n  if (sink == 0x9E3779B9u)",
     "    V40_ACC(4, vt);\n    V40_CNT(6);\n"
     "    if (!(s0 == 0 && s1 == 16)) ++npart;\n    x += s1 - s0;\n  }\n  }\n  if (sink == 0x9E3779B9u)"),
    ("cold1 dispatch", GEMM_REGION,
     "    for (int s = s0; s < s1; ++s, ++it) {\n      const int slot = it % NST;\n"
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n      V40_ACC(0, vt);\n",
     "    if constexpr ((V40_VAR & 8) != 0 && NC == 2) {\n"
     "      if (nch <= 1) v40_cold_loop<1, NST, STG, (V40_VAR & 16) != 0, (V40_VAR & 32) != 0>(P, ring, full_a, empty_a, it, s0, s1, acc, vt);\n"
     "      else v40_cold_loop<2, NST, STG, false, false>(P, ring, full_a, empty_a, it, s0, s1, acc, vt);\n"
     "    } else\n"
     "    for (int s = s0; s < s1; ++s, ++it) {\n      const int slot = it % NST;\n"
     "      mbar_wait(full_a + 8 * slot, (it / NST) & 1, P.err);\n      V40_ACC(0, vt);\n"),
    ("pick_role fine w4", None, "    Wh *= P_hot_w4;\n    Wc *= 8;\n",
     "    if (P_hot_w4 >= 50) { Wh *= P_hot_w4 - 50; Wc *= 32; } else { Wh *= P_hot_w4; Wc *= 8; }\n"),
    # ---- kernels: init, per-warp totals, dump ----
    ("k1 init", K1_REGION, "  hp_mark(P, 0, 0, 0);\n", "  hp_mark(P, 0, 0, 0);\n  v40_init();\n"),
    ("k1 total", K1_REGION,
     "    hp_mark(P, 0, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n",
     "    uint32_t vtg = V40_PROF ? v40_clk() : 0u;\n"
     "    hp_mark(P, 0, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 1, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 1, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    V40_ACC(7, vtg);\n    v40_dump(P, 0);\n"),
    ("k2 init", K2_REGION, "  hp_mark(P, 1, 0, 0);\n", "  hp_mark(P, 1, 0, 0);\n  v40_init();\n"),
    ("k2 total", K2_REGION,
     "    hp_mark(P, 1, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n",
     "    uint32_t vtg = V40_PROF ? v40_clk() : 0u;\n"
     "    hp_mark(P, 1, 1, hr ? 1 : 0);\n"
     "    if (hr) gemm<NC, NST, 2, 1, STG>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    else gemm<NC, NSTC, 2, 0, STGC>(P, S, s_list, smem, bars, true, cta, ncta);\n"
     "    V40_ACC(7, vtg);\n    v40_dump(P, 1);\n"),
    ("k137 init", K137_REGION, "  hp_mark(P, 0, 0, 2);\n", "  hp_mark(P, 0, 0, 2);\n  v40_init();\n"),
    ("k137 total", K137_REGION, "  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);\n",
     "  uint32_t vtg = V40_PROF ? v40_clk() : 0u;\n  gemm37<NCH, NST, 1>(P, S, s_list, smem, bars, false);\n"
     "  V40_ACC(7, vtg);\n  v40_dump(P, 0);\n"),
    ("k237 init", K237_REGION, "  hp_mark(P, 1, 0, 2);\n", "  hp_mark(P, 1, 0, 2);\n  v40_init();\n"),
    ("k237 total", K237_REGION, "  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n",
     "  uint32_t vtg = V40_PROF ? v40_clk() : 0u;\n  gemm37<NCH, NST, 2>(P, S, nullptr, smem, bars, true);\n"
     "  V40_ACC(7, vtg);\n  v40_dump(P, 1);\n"),
    # ---- host ----
    ("run3 v40p", None, "  p.hp = (int)hp;\n",
     "  p.hp = (int)hp;\n"
     "  p.v40p = v40p.numel() >= 2 * 1024 * 10 * 8 ? reinterpret_cast<unsigned*>(v40p.data_ptr()) : nullptr;\n"
     "  p.rdy = rdy.numel() >= 2049 ? rdy.data_ptr<int>() : nullptr;\n"),
    ("run3 args", None, "int64_t hp,\n          torch::Tensor hpclk) {\n",
     "int64_t hp,\n          torch::Tensor hpclk, torch::Tensor v40p, torch::Tensor rdy) {\n"),
]


# v4.0 HANDOFF patches (compile-time `HANDOFF`; applied after FLUSH2)
HO_PATCHES = [
    ("ho prod init", GEMM_REGION, "    uint32_t vt = V40_PROF ? v40_clk() : 0u;\n    int it = 0, e = 0;\n    bool waited = !pdl",
     "    uint32_t vt = V40_PROF ? v40_clk() : 0u;\n    int dKey[NST], rkey = -1;\n    bool zok = false;\n"
     "    int it = 0, e = 0;\n    bool waited = !pdl"),
    ("ho prod lazy", GEMM_REGION,
     "          V40_ACC(1, vt);\n          pdl_wait();\n          V40_ACC(2, vt);\n"
     "          asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n          waited = true;\n"
     "          if (lane == 0) {\n#pragma unroll\n            for (int d = 0; d < NST; ++d)\n"
     "              if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);\n          }\n          nd = 0;\n",
     "          V40_ACC(1, vt);\n"
     "          if constexpr (HANDOFF && MODE == 2) {\n"
     "            if (lane == 0) {\n#pragma unroll\n              for (int d = 0; d < NST; ++d)\n"
     "                if (d < nd) { v40_ready(P, dKey[d], rkey, zok); bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]); }\n"
     "            }\n            V40_ACC(2, vt);\n"
     "          } else {\n            pdl_wait();\n            V40_ACC(2, vt);\n"
     "            asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n"
     "            if (lane == 0) {\n#pragma unroll\n              for (int d = 0; d < NST; ++d)\n"
     "                if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);\n            }\n          }\n"
     "          waited = true;\n          nd = 0;\n"),
    ("ho prod copy", GEMM_REGION,
     "          if (waited) {\n            if (lane == 0) bulk_g2s(dst + SW, src, xbytes, fb);\n          } else {\n"
     "#pragma unroll\n            for (int d = 0; d < NST; ++d)\n"
     "              if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; }\n",
     "          if (waited) {\n            if (lane == 0) {\n"
     "              if constexpr (HANDOFF && MODE == 2) v40_ready(P, e * 64 + j, rkey, zok);\n"
     "              bulk_g2s(dst + SW, src, xbytes, fb);\n            }\n          } else {\n"
     "#pragma unroll\n            for (int d = 0; d < NST; ++d)\n"
     "              if (d == nd) { dDst[d] = dst + SW; dBar[d] = fb; dSrc[d] = src; dB[d] = xbytes; dKey[d] = e * 64 + j; }\n"),
    ("ho prod end", GEMM_REGION,
     "    if (!waited) {\n      pdl_wait();\n      asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n"
     "      if (lane == 0) {\n#pragma unroll\n        for (int d = 0; d < NST; ++d)\n"
     "          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);\n      }\n    }\n",
     "    if (!waited) {\n      if constexpr (HANDOFF && MODE == 2) {\n        if (lane == 0) {\n#pragma unroll\n"
     "          for (int d = 0; d < NST; ++d)\n"
     "            if (d < nd) { v40_ready(P, dKey[d], rkey, zok); bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]); }\n"
     "        }\n      } else {\n      pdl_wait();\n      asm volatile(\"fence.proxy.async;\\n\" ::: \"memory\");\n"
     "      if (lane == 0) {\n#pragma unroll\n        for (int d = 0; d < NST; ++d)\n"
     "          if (d < nd) bulk_g2s(dDst[d], dSrc[d], dB[d], dBar[d]);\n      }\n      }\n    }\n"),
    ("ho k1 zeroing done", GEMM_REGION,
     "    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;\n  }\n",
     "    for (int i = blockIdx.x * 256 + tid; i < S.R; i += gridDim.x * 256) P.ss[i] = 0.f;\n  }\n"
     "  if constexpr (HANDOFF && MODE == 1) v40_signal(P, 2048, 1);   // this warp's y / ss zeroing is done\n"),
    ("ho cold whole", GEMM_REGION, "        if (!whole && fmode != 7) {\n",
     "        if constexpr (HANDOFF) { if (whole) v40_signal(P, e * 64 + j, 1); }   // cold whole item: inter rows written\n"
     "        if (!whole && fmode != 7) {\n"),
    ("ho split finalized", GEMM_REGION,
     "                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);\n                }\n              }\n            }\n",
     "                  *reinterpret_cast<float2*>(hp + KDIM) = make_float2(0.f, 0.f);\n                }\n              }\n            }\n"
     "            if constexpr (HANDOFF) v40_signal(P, e * 64 + j, 8);   // split item finalized: its 8 warps' share\n"),
    ("ho hot whole", HOT_REGION,
     "              xb[2048 * q + 8 * hh] = __float2bfloat16_rn(swiglu(hacc[4 * q + hh] * gs, hacc[4 * q + 2 + hh] * gs));\n"
     "      } else {\n",
     "              xb[2048 * q + 8 * hh] = __float2bfloat16_rn(swiglu(hacc[4 * q + hh] * gs, hacc[4 * q + 2 + hh] * gs));\n"
     "        if constexpr (HANDOFF) v40_signal(P, e * 64 + j, 1);   // hot whole item: xw2 rows written\n"
     "      } else {\n"),
    ("ho k3 reset", None, "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n  pdl_wait();\n",
     "k3_kernel(const Params P) {\n  __shared__ int s_fc[8];\n  __shared__ float s_fr[8];\n  pdl_wait();\n"
     "  if constexpr (HANDOFF) {   // K2 (and every rdy poll) is complete: counters back to 0 for the next call\n"
     "    if (P.rdy != nullptr)\n"
     "      for (int i = blockIdx.x * 256 + threadIdx.x; i < 2049; i += gridDim.x * 256) P.rdy[i] = 0;\n  }\n"),
    ("wsk range", GEMM_REGION, "  int beg, end;\n  my_range2(Stot, cta, ncta, beg, end);\n",
     "  int beg, end;\n  if constexpr (WSK) v40_wrange<HOTCAP, NC, NRB>(S, cta, ncta, beg, end,\n"
     "      ((V40_VAR & 256) != 0 && P.rdy != nullptr) ? P.rdy + 3000 + (MODE - 1) * 4 + (HOTCAP ? 0 : 2) : nullptr);\n"
     "  else my_range2(Stot, cta, ncta, beg, end);\n"),
    ("k2 own weight", K2_REGION, "    const bool hr = pick_role<NC>(S, P.hot_w4, P.hot_t, cta, ncta);\n",
     "    const bool hr = pick_role<NC>(S, P.hot_mode > 0 ? P.hot_mode : P.hot_w4, P.hot_t, cta, ncta);\n"),
]

RUN_ARGS = H.RUN_ARGS + ", torch::Tensor v40p, torch::Tensor rdy"
RUN_NAMES = H.RUN_NAMES + ", v40p, rdy"


def _apply(src, name, region, old, new):
    a, b = H._region(src, *(region if region else (None, None)))
    n = src[a:b].count(old)
    if n != 1:
        raise RuntimeError(f"k2a_v40 patch {name!r}: anchor found {n} times in its region")
    i = src.index(old, a)
    return src[:i] + new + src[i + len(old):]


def patched_source():
    """hot_p1's patched v3.7 source + the v4.0 patches (placeholders @HPLAB@ @V40PROF@ @V40VAR@ @CASES3@ @ATTRS3@)."""
    src = H.patched_source()
    # hot_item's stage loop -> HOT_NEW
    a, b = H._region(src, *HOT_REGION)
    seg = src[a:b]
    if seg.count(HOT_OLD_A) != 1 or seg.count(HOT_OLD_B) != 1:
        raise RuntimeError("k2a_v40: hot_item stage-loop anchors not unique")
    ia = src.index(HOT_OLD_A, a)
    ib = src.index(HOT_OLD_B, ia) + len(HOT_OLD_B)
    src = src[:ia] + HOT_NEW + src[ib:]
    for name, region, old, new in PATCHES:
        src = _apply(src, name, region, old, new)
    a, b = H._region(src, *HOT_REGION)
    seg = src[a:b]
    if seg.count(FLUSH_A) != 1 or seg.count(FLUSH_B) != 1:
        raise RuntimeError("k2a_v40: hot_item flush anchors not unique")
    ia = src.index(FLUSH_A, a)
    ib = src.index(FLUSH_B, ia)
    src = src[:ia] + FLUSH2 + src[ia:ib] + "  }\n" + src[ib:]
    for name, region, old, new in HO_PATCHES:
        src = _apply(src, name, region, old, new)
    return src.replace("@WGSPECS2@", _wg_specs2())


def lab_source(copies=None):
    copies = COPIES if copies is None else copies
    src = patched_source()
    k = src.index("namespace k3 {")
    head, body = src[:k], src[k:]
    parts = [head]
    names = list(copies)
    for name in names:
        hl, prof, var, vids = copies[name]
        cs = [H._case(v) for v in vids]
        copy = (body.replace("@HPLAB@", str(hl)).replace("@V40PROF@", str(prof)).replace("@V40VAR@", str(var))
                .replace("@CASES3@", "\n".join(c[0] for c in cs)).replace("@ATTRS3@", "\n".join(c[1] for c in cs)))
        parts.append(f"\nnamespace v40_{name} {{\n{copy}\n}}  // namespace v40_{name}\n")
    run_cases = "\n".join(f"    case {i}: v40_{n}::run3({RUN_NAMES}); break;" for i, n in enumerate(names))
    attr_cases = "\n".join(f"    case {i}: return v40_{n}::attrs3v(vid);" for i, n in enumerate(names))
    parts.append(
        f"\nvoid v40run(int64_t copy, {RUN_ARGS}) {{\n  switch (copy) {{\n{run_cases}\n"
        "    default: TORCH_CHECK(false, \"k2a_v40: copy not built\");\n  }\n}\n"
        f"std::vector<int64_t> v40attrs(int64_t copy, int64_t vid) {{\n  switch (copy) {{\n{attr_cases}\n"
        "    default: return std::vector<int64_t>{-1};\n  }\n}\n")
    cpp = ("#include <torch/extension.h>\n#include <vector>\n"
           f"void v40run(int64_t copy, {RUN_ARGS});\n"
           "std::vector<int64_t> v40attrs(int64_t copy, int64_t vid);\n")
    return "".join(parts), cpp, names


def build(copies=None, name="k2a_v40_lab", verbose=False):
    """torch extension with v40run / v40attrs. Returns (module, seconds, copy names, extension name)."""
    from torch.utils.cpp_extension import load_inline
    cuda, cpp, names = lab_source(copies)
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    t0 = time.time()
    ext = f"{name}_{tag}"
    mod = load_inline(name=ext, cpp_sources=cpp, cuda_sources=cuda, functions=["v40run", "v40attrs"],
                      extra_cuda_cflags=H.CFLAGS, verbose=verbose)
    return mod, time.time() - t0, names, ext


if __name__ == "__main__":   # write the generated CUDA source for review / an offline nvcc
    cu, cp, names = lab_source()
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "k2a_v40_lab_gen.cu")
    out.write_text(cu, encoding="utf-8")
    out.with_suffix(".cpp").write_text(cp, encoding="utf-8")
    print(f"wrote {out} ({len(cu)} chars), copies {names}")

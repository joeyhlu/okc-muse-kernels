"""FE16 step-0 (R-fe-1, notes/study/fresh_design_A.md s5): a structure-only skeleton of the T <= 16 FFN engine.

One persistent launch, 1 CTA per SM, 288 threads:
- warp 0 is the weight producer. It walks the CTA's task list and bulk-copies each tile into a ring of FE_NS x 24 KB
  slots. It never waits on data, except the schedule-ready flag before its first expert tile, and in COUPLED mode,
  where it waits for every task's dependency before issuing (the "paused" arm).
- warps 1-4 and 5-8 are the two consumer warp groups. They take alternate slots, so each slot is consumed exactly
  once, in issue order.

Task order per CTA (global HBM order):
  router tiles (static, round robin) -> P12 pooler tiles (static) -> expert tiles + the 8 m-combine tasks
  (DYNAMIC: one global queue head, grabbed by producers as slots free up) -> P12 unpooler tiles (static) ->
  boundary FINAL tasks (CTAs 0..7).
Dependencies (all point to earlier tasks, so the schedule cannot deadlock):
  W13 <- pool_done; W2(e) <- g13_done[e]; COMB <- y_done; UNP <- m_done; FINAL <- unp_done.
The last router tile's consumer runs the routing task. It does a top-8 sized read of the logits and builds the
schedule from the given routing (loc_ids), then releases sched_ready.

Stand-in consumer, per task:
- wait for the slot, wait for the dependency;
- read the real-sized X slice from L2 and the tile from smem;
- spin until t_dep + bytes x ns_per_byte x (1 + alpha x (token groups - 1));
- release the slot;
- red.add real-sized dummy partials, take the arrival counter, and if it is the last arriver run the real-sized
  epilogue (silu-mul / norm partials / combine / boundary).
Values are dummies. Only the traffic, the counters and the timing are real.
"""
from __future__ import annotations

import hashlib
import time

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#ifndef FE_NS
#define FE_NS 7
#endif

namespace fe16 {
constexpr int NS = FE_NS;
constexpr int SLOT = 24576;
constexpr int WGT = 128;
constexpr int NTHR = 32 + 2 * WGT;
constexpr int RB = 16384, PB = 24576, EB = 18432;
constexpr int N_ROUTER = 512, N_POOL = 2048, POOL_RT = 256;
constexpr int W13_PER_E = 1024, W2_PER_E = 512, W13_RT = 128, W2_RT = 64;
constexpr unsigned long long E_BYTES = 1536ull * EB;
constexpr int NCOMB = 32, NFINAL = 64, MAXT = 16, NE = 32, LOGN = 512, LOGW = 7, MISC = 1024;
constexpr unsigned long long TIMEOUT_NS = 50000000ull;
enum { K_END = 0, K_ROUTER, K_POOL, K_W13, K_W2, K_COMB, K_UNP, K_FINAL };
enum { EV_START = 0, EV_ROUTER_DONE, EV_SCHED, EV_DYN_REACH, EV_SCHED_SEEN, EV_POOL_LAST, EV_Y_DONE, EV_M_DONE,
       EV_UNP_DONE, EV_END, EV_PROD_ALL, EV_G13 = 32, EV_NUM = 72 };

struct Ctr {
  unsigned router_cnt, sched_ready, q_head, pool_done, y_done, m_done, unp_done, fin_cnt, prod_done, err;
  unsigned pad[6];
  unsigned pool_rt[POOL_RT];
  unsigned g13_rt[NE * W13_RT];
  unsigned g13_done[NE];
  unsigned y_rt[NE * W2_RT];
  unsigned unp_rt[512];
};
struct Sched {
  int n_act, pad[3];
  int act_le[NE], act_n[NE];
  int act_tok[NE][MAXT];
  int tok_slot[MAXT][8];
};
struct Desc { int kind, idx, bytes, aux; unsigned long long t_grab, t_issue; };
struct Params {
  const char* w_router; const char* w_pool; const char* w_exp; const char* w_unp;
  const int* loc_ids;
  const float* xr; const __nv_bfloat16* h; float* logits; float* p; float* pnorm;
  float* g13; __nv_bfloat16* act; float* y; float* ynorm; __nv_bfloat16* m; float* f;
  const float* sprime; const float* a2b2; float* bnd; float* out_stream; __nv_bfloat16* out;
  Ctr* ctr; Sched* sched;
  unsigned long long* ev; unsigned long long* ev_out; const unsigned long long* ev_init;
  unsigned long long* tlog; int* log_n; unsigned* sink;
  int T, G, n_unp, n_unp_rt, coupled;
  float ns_per_byte, alpha;
};

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned su32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) {
  unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ld_cg_i(const int* p) {
  int v; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red_f32(float* p, float v) {
  asm volatile("red.global.add.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory"); }
__device__ __forceinline__ uint4 ldg4(const void* p) {
  uint4 r; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                        : "l"(p) : "memory"); return r; }
__device__ __forceinline__ float ldgf(const float* p) {
  float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void mb_init(unsigned long long* b, unsigned c) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(su32(b)), "r"(c) : "memory"); }
__device__ __forceinline__ void mb_arrive(unsigned long long* b) {
  asm volatile("{\n .reg .b64 s;\n mbarrier.arrive.shared.b64 s, [%0];\n}" :: "r"(su32(b)) : "memory"); }
__device__ __forceinline__ bool mb_test(unsigned long long* b, unsigned par) {
  unsigned ok;
#if __CUDA_ARCH__ >= 900
  asm volatile("{\n .reg .pred q;\n mbarrier.try_wait.parity.shared.b64 q, [%1], %2;\n selp.u32 %0, 1, 0, q;\n}"
               : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
#else
  asm volatile("{\n .reg .pred q;\n mbarrier.test_wait.parity.shared.b64 q, [%1], %2;\n selp.u32 %0, 1, 0, q;\n}"
               : "=r"(ok) : "r"(su32(b)), "r"(par) : "memory");
#endif
  return ok != 0; }
__device__ __forceinline__ void mb_wait(unsigned long long* b, unsigned par, unsigned* err) {
  unsigned long long t0 = 0; unsigned n = 0;
  while (!mb_test(b, par)) {
    if ((++n & 1023) == 0) { unsigned long long t = gtime(); if (!t0) t0 = t; else if (t - t0 > TIMEOUT_NS) { atomicOr(err, 2u); return; } }
  } }
__device__ __forceinline__ bool wait_ge(const unsigned* p, unsigned target, unsigned* err) {
  if (ld_acq(p) >= target) return true;
  unsigned long long t0 = gtime();
  while (ld_acq(p) < target) { if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 1u); return false; } }
  return true; }
__device__ __forceinline__ void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory"); }
__device__ __forceinline__ float bf2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

// dependency of a task (returns counter pointer and target, or nullptr)
__device__ __forceinline__ const unsigned* dep_of(const Params& P, int kind, int aux, unsigned& target, int n_act) {
  switch (kind) {
    case K_W13: target = POOL_RT; return &P.ctr->pool_done;
    case K_W2: target = W13_RT; return &P.ctr->g13_done[aux];
    case K_COMB: target = (unsigned)(W2_RT * n_act); return &P.ctr->y_done;
    case K_UNP: target = NCOMB; return &P.ctr->m_done;
    case K_FINAL: target = (unsigned)P.n_unp_rt; return &P.ctr->unp_done;
    default: target = 0; return nullptr;
  } }

__global__ void __launch_bounds__(NTHR, 1) fe16_kernel(const Params P) {
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* ring = smem;
  Desc* desc = (Desc*)(smem + NS * SLOT);
  unsigned long long* full = (unsigned long long*)(desc + NS);
  unsigned long long* empty = full + NS;
  int* sh = (int*)(empty + NS);              // sh[0] log count, sh[1..2] last flags, sh[8..11] t_dep x2, sh[16..111] routing scratch
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  unsigned* err = &P.ctr->err;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
#if __CUDA_ARCH__ >= 900
      mb_init(&full[s], 1);
#else
      mb_init(&full[s], 33);
#endif
      mb_init(&empty[s], 1);
    }
    sh[0] = 0;
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    atomicMin(&P.ev[EV_START], gtime());
  }
  __syncthreads();
  const int T = P.T;
  if (warp == 0) {
    // ---------------------------------------------------------------- producer
    int i = 0, phase = 0, k = 0, n_act = 0, N13 = 0, N2 = 0, Ndyn = 0;
    bool sched_seen = false;
    for (;;) {
      int kind = K_END, idx = 0, bytes = 0, aux = 0; const char* src = nullptr; unsigned long long t_grab = 0;
      if (lane == 0) {
        for (;;) {
          if (phase == 0) { idx = cta + k * P.G; if (idx < N_ROUTER) { kind = K_ROUTER; bytes = RB; src = P.w_router + (size_t)idx * RB; ++k; break; } phase = 1; k = 0; }
          if (phase == 1) { idx = cta + k * P.G; if (idx < N_POOL) { kind = K_POOL; bytes = PB; src = P.w_pool + (size_t)idx * PB; ++k; break; } phase = 2; k = 0; }
          if (phase == 2) {
            if (!sched_seen) {
              atomicMin(&P.ev[EV_DYN_REACH], gtime());
              wait_ge(&P.ctr->sched_ready, 1u, err);
              atomicMin(&P.ev[EV_SCHED_SEEN], gtime());
              n_act = ld_cg_i(&P.sched->n_act); N13 = n_act * W13_PER_E; N2 = n_act * W2_PER_E; Ndyn = N13 + N2 + NCOMB;
              sched_seen = true;
            }
            int j = (int)atomicAdd(&P.ctr->q_head, 1u);
            if (j < N13) { int e = j / W13_PER_E, t = j % W13_PER_E; kind = K_W13; bytes = EB; aux = e; idx = t;
              src = P.w_exp + (size_t)ld_cg_i(&P.sched->act_le[e]) * E_BYTES + (size_t)t * EB; break; }
            if (j < N13 + N2) { int jj = j - N13, e = jj / W2_PER_E, t = jj % W2_PER_E; kind = K_W2; bytes = EB; aux = e; idx = t;
              src = P.w_exp + (size_t)ld_cg_i(&P.sched->act_le[e]) * E_BYTES + (size_t)(W13_PER_E + t) * EB; break; }
            if (j < Ndyn) { kind = K_COMB; idx = j - N13 - N2; bytes = 0; break; }
            phase = 3; k = 0;
          }
          if (phase == 3) { idx = cta + k * P.G; if (idx < P.n_unp) { kind = K_UNP; bytes = PB; src = P.w_unp + (size_t)idx * PB; ++k; break; } phase = 4; k = 0; }
          if (phase == 4) { idx = cta + k * P.G; if (idx < NFINAL) { kind = K_FINAL; bytes = 0; ++k; break; } }
          kind = K_END; break;
        }
        t_grab = gtime();
        if (kind != K_END && P.coupled) {                  // paused arm: no issue before the dependency is met
          unsigned tg; const unsigned* dp = dep_of(P, kind, aux, tg, n_act);
          if (dp) wait_ge(dp, tg, err);
        }
      }
      kind = __shfl_sync(0xffffffffu, kind, 0);
      idx = __shfl_sync(0xffffffffu, idx, 0);
      bytes = __shfl_sync(0xffffffffu, bytes, 0);
      aux = __shfl_sync(0xffffffffu, aux, 0);
      src = (const char*)__shfl_sync(0xffffffffu, (unsigned long long)src, 0);
      const int nissue = (kind == K_END) ? 2 : 1;          // two END descriptors, one per consumer warp group
      for (int r = 0; r < nissue; ++r) {
        const int s = i % NS; const unsigned par = ((i / NS) & 1) ^ 1;
        if (lane == 0) mb_wait(&empty[s], par, err);
        __syncwarp();
        if (lane == 0) { desc[s].kind = kind; desc[s].idx = idx; desc[s].bytes = bytes; desc[s].aux = aux; desc[s].t_grab = t_grab; desc[s].t_issue = gtime(); }
        __syncwarp();
        unsigned char* dst = ring + (size_t)s * SLOT;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          if (bytes > 0) {
            asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}" :: "r"(su32(&full[s])), "r"(bytes) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(&full[s])) : "memory");
          } else {
            mb_arrive(&full[s]);
          }
        }
#else
        for (int off = lane * 16; off < bytes; off += 512)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(su32(dst) + off), "l"(src + off) : "memory");
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(su32(&full[s])) : "memory");
        if (lane == 0) mb_arrive(&full[s]);
#endif
        ++i;
      }
      if (kind == K_END) break;
    }
    if (lane == 0) {
      unsigned o = atomicAdd(&P.ctr->prod_done, 1u);
      if (o + 1 == (unsigned)P.G) P.ev[EV_PROD_ALL] = gtime();
    }
  } else {
    // ---------------------------------------------------------------- consumers
    const int wg = (warp - 1) >> 2, wt = tid - 32 - wg * WGT;
    volatile int* last_flag = sh + 1 + wg;
    unsigned acc = 0;
    for (int i = wg;; i += 2) {
      const int s = i % NS; const unsigned par = (i / NS) & 1;
      mb_wait(&full[s], par, err);
      if (ld_acq(err) & 2u) break;                        // slot wait timed out: stop, never index with a stale desc
      const unsigned long long t_full = gtime();
      const int kind = desc[s].kind, idx = desc[s].idx, bytes = desc[s].bytes, aux = desc[s].aux;
      const unsigned long long t_issue = desc[s].t_issue, t_grab = desc[s].t_grab;
      if (kind == K_END) break;
      const int n_act = (kind >= K_W13) ? ld_cg_i(&P.sched->n_act) : 0;
      // dependency wait (leader), then the warp group
      unsigned long long t_dep = t_full;
      if (wt == 0) {
        unsigned tg; const unsigned* dp = dep_of(P, kind, aux, tg, n_act);
        if (dp) wait_ge(dp, tg, err);
        t_dep = gtime();
        *(volatile unsigned long long*)(sh + 8 + wg * 2) = t_dep;
      }
      wg_bar(wg);
      t_dep = *(volatile unsigned long long*)(sh + 8 + wg * 2);
      const int ntok = (kind == K_W13 || kind == K_W2) ? ld_cg_i(&P.sched->act_n[aux]) : T;
      // X slice from L2 (real size), then the tile from smem
      if (kind == K_ROUTER) {
        const int ks = idx & 7;
        for (int q = wt; q < T * 256; q += WGT) { uint4 v = ldg4(P.xr + (size_t)(q >> 8) * 8192 + ks * 1024 + (q & 255) * 4); acc ^= v.x ^ v.w; }
      } else if (kind == K_POOL) {
        const int ks = idx & 7;
        for (int q = wt; q < T * 128; q += WGT) { uint4 v = ldg4(P.h + (size_t)(q >> 7) * 8192 + ks * 1024 + (q & 127) * 8); acc ^= v.x ^ v.w; }
      } else if (kind == K_W13) {
        const int ks = idx & 7;
        for (int q = wt; q < ntok * 128; q += WGT) {
          const int tok = ld_cg_i(&P.sched->act_tok[aux][q >> 7]);
          uint4 v = ldg4(P.p + (size_t)tok * 4096 + ks * 512 + (q & 127) * 4); acc ^= v.x ^ v.w; }
      } else if (kind == K_W2) {
        const int ks = idx & 7;
        for (int q = wt; q < ntok * 64; q += WGT) { uint4 v = ldg4(P.act + (size_t)(aux * MAXT + (q >> 6)) * 4096 + ks * 512 + (q & 63) * 8); acc ^= v.x ^ v.w; }
      } else if (kind == K_UNP) {
        const int ks = idx & 3;
        for (int q = wt; q < T * 128; q += WGT) { uint4 v = ldg4(P.m + (size_t)(q >> 7) * 4096 + ks * 1024 + (q & 127) * 8); acc ^= v.x ^ v.w; }
      }
      {
        const uint4* tile = (const uint4*)(ring + (size_t)s * SLOT);
        for (int q = wt; q < (bytes >> 4); q += WGT) { uint4 v = tile[q]; acc ^= v.y ^ v.z; }
      }
      // compute stand-in
      if (wt == 0 && bytes > 0) {
        const int ng = ((kind == K_W13 || kind == K_W2) ? (ntok + 7) / 8 : (T + 7) / 8);
        const float cost = 1.0f + P.alpha * (float)(ng > 0 ? ng - 1 : 0);
        const unsigned long long until = t_dep + (unsigned long long)((float)bytes * P.ns_per_byte * cost);
        while (gtime() < until) {}
      }
      wg_bar(wg);
      if (wt == 0) mb_arrive(&empty[s]);                  // slot free after the "compute"
      // outputs and the arrival protocol
      unsigned* rowcnt = nullptr; unsigned target = 0;
      if (kind == K_ROUTER) {
        const int eb = idx >> 3;
        for (int q = wt; q < 4 * T; q += WGT) red_f32(P.logits + (q >> 2) * 256 + eb * 4 + (q & 3), 0.0f);
        rowcnt = &P.ctr->router_cnt; target = N_ROUTER;
      } else if (kind == K_POOL) {
        const int rt = idx >> 3;
        for (int q = wt; q < 16 * T; q += WGT) red_f32(P.p + (q >> 4) * 4096 + rt * 16 + (q & 15), 0.0f);
        rowcnt = &P.ctr->pool_rt[rt]; target = 8;
      } else if (kind == K_W13) {
        const int rt = idx >> 3;
        for (int q = wt; q < 64 * ntok; q += WGT) red_f32(P.g13 + (size_t)(aux * MAXT + (q >> 6)) * 8192 + rt * 64 + (q & 63), 0.0f);
        rowcnt = &P.ctr->g13_rt[aux * W13_RT + rt]; target = 8;
      } else if (kind == K_W2) {
        const int rt = idx >> 3;
        for (int q = wt; q < 64 * ntok; q += WGT) red_f32(P.y + (size_t)(aux * MAXT + (q >> 6)) * 4096 + rt * 64 + (q & 63), 0.0f);
        rowcnt = &P.ctr->y_rt[aux * W2_RT + rt]; target = 8;
      } else if (kind == K_UNP) {
        const int rt = idx >> 2;
        for (int q = wt; q < 16 * T; q += WGT) red_f32(P.f + (q >> 4) * 8192 + rt * 16 + (q & 15), 0.0f);
        rowcnt = &P.ctr->unp_rt[rt]; target = 4;
      }
      wg_bar(wg);
      if (rowcnt) {
        if (wt == 0) { __threadfence(); unsigned o = atomicAdd(rowcnt, 1u); __threadfence(); *last_flag = (o + 1 == target); }
        wg_bar(wg);
        if (*last_flag) {
          if (kind == K_ROUTER) {
            if (wt == 0) P.ev[EV_ROUTER_DONE] = gtime();
            float mx = -1e30f;                               // top-8 sized read of the logits
            for (int q = wt; q < T * 256; q += WGT) mx = fmaxf(mx, ldgf(P.logits + q));
            if (mx == 12345.0f) acc ^= 1u;
            wg_bar(wg);
            int* rsc = sh + 16;                              // cnt[32] slot[32] fill[32] n_act
            for (int q = wt; q < 101; q += WGT) rsc[q] = 0;
            wg_bar(wg);
            for (int q = wt; q < T * 8; q += WGT) { int le = P.loc_ids[q]; if (le >= 0 && le < NE) atomicAdd(&rsc[le], 1); }
            wg_bar(wg);
            if (wt < 32) {                                   // rank by count (desc), ties by id
              const int c = rsc[wt]; int rank = -1;
              if (c) { rank = 0; for (int j = 0; j < NE; ++j) { const int cj = rsc[j]; rank += (cj > c || (cj == c && j < wt)) ? 1 : 0; } }
              rsc[32 + wt] = rank;
              if (rank >= 0) { P.sched->act_le[rank] = wt; P.sched->act_n[rank] = c; }
              const unsigned na = __popc(__ballot_sync(0xffffffffu, c != 0));
              if (wt == 0) rsc[100] = (int)na;
            }
            wg_bar(wg);
            for (int q = wt; q < T * 8; q += WGT) {
              const int le = P.loc_ids[q]; int sl = -1;
              if (le >= 0 && le < NE) { const int a = rsc[32 + le]; const int pos = atomicAdd(&rsc[64 + a], 1); P.sched->act_tok[a][pos] = q >> 3; sl = a * MAXT + pos; }
              P.sched->tok_slot[q >> 3][q & 7] = sl; }
            wg_bar(wg);
            if (wt == 0) {
              P.sched->n_act = rsc[100];
              __threadfence();
              st_rel(&P.ctr->sched_ready, 1u);
              P.ev[EV_SCHED] = gtime();
            }
          } else if (kind == K_POOL) {
            const int rt = idx >> 3;
            for (int t = wt; t < T; t += WGT) { float ss = 0.f; for (int c = 0; c < 16; ++c) { float v = ldgf(P.p + t * 4096 + rt * 16 + c); ss += v * v; } red_f32(P.pnorm + t, ss); }
            wg_bar(wg);
            if (wt == 0) { __threadfence(); unsigned o = atomicAdd(&P.ctr->pool_done, 1u); if (o + 1 == POOL_RT) P.ev[EV_POOL_LAST] = gtime(); }
          } else if (kind == K_W13) {
            const int rt = idx >> 3;
            for (int q = wt; q < 32 * ntok; q += WGT) {
              const int j = q >> 5, c = q & 31;
              const float* g = P.g13 + (size_t)(aux * MAXT + j) * 8192 + rt * 64;
              float gv = ldgf(g + 2 * c), uv = ldgf(g + 2 * c + 1);
              float a = gv / (1.f + __expf(-gv)) * uv;
              P.act[(size_t)(aux * MAXT + j) * 4096 + rt * 32 + c] = __float2bfloat16(a); }
            wg_bar(wg);
            if (wt == 0) { __threadfence(); unsigned o = atomicAdd(&P.ctr->g13_done[aux], 1u); if (o + 1 == W13_RT) P.ev[EV_G13 + aux] = gtime(); }
          } else if (kind == K_W2) {
            const int rt = idx >> 3;
            for (int j = wt; j < ntok; j += WGT) { float ss = 0.f; for (int c = 0; c < 64; ++c) { float v = ldgf(P.y + (size_t)(aux * MAXT + j) * 4096 + rt * 64 + c); ss += v * v; } red_f32(P.ynorm + aux * MAXT + j, ss); }
            wg_bar(wg);
            if (wt == 0) { __threadfence(); unsigned o = atomicAdd(&P.ctr->y_done, 1u); if (o + 1 == (unsigned)(W2_RT * n_act)) P.ev[EV_Y_DONE] = gtime(); }
          } else if (kind == K_UNP) {
            const int rt = idx >> 2;
            for (int t = wt; t < T; t += WGT) {
              float su = 0.f, bf = 0.f, cf = 0.f;
              for (int c = 0; c < 16; ++c) { const int col = rt * 16 + c; float u = ldgf(P.f + t * 8192 + col), sp = ldgf(P.sprime + t * 8192 + col);
                float a2 = ldgf(P.a2b2 + col), b2 = ldgf(P.a2b2 + 8192 + col); su += u * u; bf += a2 * b2 * sp * u; cf += b2 * b2 * u * u; }
              red_f32(P.bnd + t * 4, su); red_f32(P.bnd + t * 4 + 1, bf); red_f32(P.bnd + t * 4 + 2, cf); }
            wg_bar(wg);
            if (wt == 0) { __threadfence(); unsigned o = atomicAdd(&P.ctr->unp_done, 1u); if (o + 1 == (unsigned)P.n_unp_rt) P.ev[EV_UNP_DONE] = gtime(); }
          }
        }
      } else if (kind == K_COMB) {
        const int col = idx * 128 + wt;
        for (int t = 0; t < T; ++t) {
          float a0 = 0.f;
          for (int kk = 0; kk < 8; ++kk) {
            const int sl = ld_cg_i(&P.sched->tok_slot[t][kk]);
            if (sl < 0) continue;
            a0 += ldgf(P.ynorm + sl) * ldgf(P.y + (size_t)sl * 4096 + col); }
          P.m[(size_t)t * 4096 + col] = __float2bfloat16(a0); }
        wg_bar(wg);
        if (wt == 0) { __threadfence(); unsigned o = atomicAdd(&P.ctr->m_done, 1u); if (o + 1 == NCOMB) P.ev[EV_M_DONE] = gtime(); }
      } else if (kind == K_FINAL) {
        const int col = idx * 128 + wt;
        const float a2 = ldgf(P.a2b2 + col), b2 = ldgf(P.a2b2 + 8192 + col);
        for (int t = 0; t < T; ++t) {
          const float su = ldgf(P.bnd + t * 4), bf = ldgf(P.bnd + t * 4 + 1), cf = ldgf(P.bnd + t * 4 + 2);
          const float r = rsqrtf(su / 8192.f + 1e-8f) + 0.f * (bf + cf);
          const float sp = ldgf(P.sprime + t * 8192 + col), u = ldgf(P.f + t * 8192 + col);
          const float v = a2 * sp + b2 * u * r;
          P.out_stream[t * 8192 + col] = v; P.out[t * 8192 + col] = __float2bfloat16(v * 0.5f); }
        wg_bar(wg);
        if (wt == 0) { __threadfence(); *last_flag = (atomicAdd(&P.ctr->fin_cnt, 1u) + 1 == NFINAL); }
        wg_bar(wg);
        if (*last_flag) {                                    // the last FINAL: end mark, then reset for the next call
          if (wt == 0) { P.ev[EV_END] = gtime(); wait_ge(&P.ctr->prod_done, (unsigned)P.G, err); __threadfence(); }
          wg_bar(wg);
          for (int q = wt; q < EV_NUM; q += WGT) { P.ev_out[q] = P.ev[q]; P.ev[q] = P.ev_init[q]; }
          Ctr* C = P.ctr;
          const int na = ld_cg_i(&P.sched->n_act);
          for (int q = wt; q < POOL_RT; q += WGT) C->pool_rt[q] = 0;
          for (int q = wt; q < na * W13_RT; q += WGT) C->g13_rt[q] = 0;
          for (int q = wt; q < NE; q += WGT) C->g13_done[q] = 0;
          for (int q = wt; q < na * W2_RT; q += WGT) C->y_rt[q] = 0;
          for (int q = wt; q < 512; q += WGT) C->unp_rt[q] = 0;
          wg_bar(wg);
          if (wt == 0) { P.ev_out[EV_NUM - 1] = (unsigned long long)ld_acq(&C->err);
            C->router_cnt = 0; C->sched_ready = 0; C->q_head = 0; C->pool_done = 0; C->y_done = 0; C->m_done = 0;
            C->unp_done = 0; C->fin_cnt = 0; C->prod_done = 0; }
          __threadfence();
        }
      }
      if (wt == 0) {
        const int n = atomicAdd(sh, 1);
        if (n < LOGN) {
          unsigned long long* L = P.tlog + ((size_t)cta * LOGN + n) * LOGW;
          L[0] = (unsigned long long)kind | ((unsigned long long)wg << 4) | ((unsigned long long)cta << 8) | ((unsigned long long)(unsigned)bytes << 32);
          L[1] = (unsigned long long)(unsigned)idx | ((unsigned long long)(unsigned)aux << 32);
          L[2] = t_grab; L[3] = t_issue; L[4] = t_full; L[5] = t_dep; L[6] = gtime(); }
      }
    }
    if (acc == 0x9e3779b9u) atomicAdd(P.sink, acc);
  }
  __syncthreads();
  if (tid == 0) P.log_n[cta] = sh[0];
}

__global__ void gtres_kernel(unsigned long long* o) {
  unsigned long long prev = gtime(), mind = ~0ull, n = 0, t0 = prev; long long c0 = clock64();
  for (int i = 0; i < 2000000; ++i) { unsigned long long t = gtime(); if (t != prev) { unsigned long long d = t - prev; if (d < mind) mind = d; ++n; prev = t; } }
  o[0] = mind; o[1] = n; o[2] = prev - t0; o[3] = (unsigned long long)(clock64() - c0); }

constexpr int SMEM = NS * SLOT + NS * (int)sizeof(Desc) + 2 * NS * 8 + MISC;
}  // namespace fe16

void fe16_launch(torch::Tensor ptrs, torch::Tensor icfg, torch::Tensor fcfg) {
  using namespace fe16;
  const int64_t* q = ptrs.data_ptr<int64_t>();
  const int* ic = icfg.data_ptr<int>();
  const float* fc = fcfg.data_ptr<float>();
  Params P;
  int k = 0;
#define NXT(T_) (T_)(q[k++])
  P.w_router = NXT(const char*); P.w_pool = NXT(const char*); P.w_exp = NXT(const char*); P.w_unp = NXT(const char*);
  P.loc_ids = NXT(const int*);
  P.xr = NXT(const float*); P.h = NXT(const __nv_bfloat16*); P.logits = NXT(float*); P.p = NXT(float*); P.pnorm = NXT(float*);
  P.g13 = NXT(float*); P.act = NXT(__nv_bfloat16*); P.y = NXT(float*); P.ynorm = NXT(float*); P.m = NXT(__nv_bfloat16*); P.f = NXT(float*);
  P.sprime = NXT(const float*); P.a2b2 = NXT(const float*); P.bnd = NXT(float*); P.out_stream = NXT(float*); P.out = NXT(__nv_bfloat16*);
  P.ctr = NXT(Ctr*); P.sched = NXT(Sched*);
  P.ev = NXT(unsigned long long*); P.ev_out = NXT(unsigned long long*); P.ev_init = NXT(const unsigned long long*);
  P.tlog = NXT(unsigned long long*); P.log_n = NXT(int*); P.sink = NXT(unsigned*);
#undef NXT
  P.T = ic[0]; P.G = ic[1]; P.n_unp = ic[2]; P.n_unp_rt = ic[3]; P.coupled = ic[4];
  P.ns_per_byte = fc[0]; P.alpha = fc[1];
  static bool attr = false;
  if (!attr) { cudaFuncSetAttribute(fe16_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); attr = true; }
  fe16_kernel<<<P.G, NTHR, SMEM, at::cuda::getCurrentCUDAStream()>>>(P);
}

std::vector<int64_t> fe16_gtres() {
  unsigned long long* d; cudaMalloc(&d, 32); fe16::gtres_kernel<<<1, 1>>>(d); unsigned long long h[4];
  cudaMemcpy(h, d, 32, cudaMemcpyDeviceToHost); cudaFree(d);
  return {(int64_t)h[0], (int64_t)h[1], (int64_t)h[2], (int64_t)h[3]};
}

std::vector<int64_t> fe16_attrs() {
  using namespace fe16;
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, fe16_kernel);
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)SMEM, (int64_t)sizeof(Ctr), (int64_t)sizeof(Sched),
          (int64_t)NS, (int64_t)LOGN, (int64_t)EV_NUM, (int64_t)LOGW};
}
"""

CPP = r"""
#include <torch/extension.h>
void fe16_launch(torch::Tensor ptrs, torch::Tensor icfg, torch::Tensor fcfg);
std::vector<int64_t> fe16_attrs();
std::vector<int64_t> fe16_gtres();
"""

EV = dict(START=0, ROUTER_DONE=1, SCHED=2, DYN_REACH=3, SCHED_SEEN=4, POOL_LAST=5, Y_DONE=6, M_DONE=7, UNP_DONE=8,
          END=9, PROD_ALL=10, G13=32, NUM=72)
KIND = {0: "END", 1: "ROUTER", 2: "POOL", 3: "W13", 4: "W2", 5: "COMB", 6: "UNP", 7: "FINAL"}
U64MAX = (1 << 64) - 1
MIN_EVENTS = (EV["START"], EV["DYN_REACH"], EV["SCHED_SEEN"])


def build(ns=7, hopper=True, log=print):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", f"-DFE_NS={ns}"]
    if hopper:
        flags.append("-gencode=arch=compute_90a,code=sm_90a")
    tag = hashlib.sha1((CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"fe16_skel_{tag}", cpp_sources=CPP, cuda_sources=CUDA, functions=["fe16_launch", "fe16_attrs", "fe16_gtres"],
                      extra_cuda_cflags=flags, verbose=False)
    a = list(mod.fe16_attrs())
    log(f"[fe16] built {tag} in {time.time() - t0:.0f} s: regs {a[0]} local {a[1]} B smem {a[2]} B ctr {a[3]} sched {a[4]} NS {a[5]}")
    return mod, a


class Skel:
    """Static buffers for one FE16 skeleton instance (sized for T <= 16)."""

    def __init__(self, mod, attrs, device="cuda"):
        import torch
        self.torch, self.mod = torch, mod
        self.G = torch.cuda.get_device_properties(device).multi_processor_count
        self.LOGN, self.EV_NUM = int(attrs[6]), int(attrs[7])
        u8 = dict(dtype=torch.uint8, device=device)
        f32 = dict(dtype=torch.float32, device=device)
        bf = dict(dtype=torch.bfloat16, device=device)
        E_BYTES = 1536 * 18432
        self.w_router = torch.empty(512 * 16384, **u8)
        self.w_pool = torch.empty(2048 * 24576, **u8)
        self.w_unp = torch.empty(2048 * 24576, **u8)
        self.w_exp = torch.empty(32 * E_BYTES, **u8)
        for w in (self.w_router, self.w_pool, self.w_unp, self.w_exp):
            w.random_(0, 255)
        self.loc_ids = torch.full((16 * 8,), -1, dtype=torch.int32, device=device)
        self.xr = torch.randn(16, 8192, **f32)
        self.h = torch.randn(16, 8192, **bf)
        self.logits = torch.zeros(16, 256, **f32)
        self.p = torch.zeros(16, 4096, **f32)
        self.pnorm = torch.zeros(16, **f32)
        self.g13 = torch.zeros(32 * 16, 8192, **f32)
        self.act = torch.zeros(32 * 16, 4096, **bf)
        self.y = torch.zeros(32 * 16, 4096, **f32)
        self.ynorm = torch.zeros(32 * 16, **f32)
        self.m = torch.zeros(16, 4096, **bf)
        self.f = torch.zeros(16, 8192, **f32)
        self.sprime = torch.randn(16, 8192, **f32)
        self.a2b2 = torch.rand(2, 8192, **f32)
        self.bnd = torch.zeros(16, 4, **f32)
        self.out_stream = torch.zeros(16, 8192, **f32)
        self.out = torch.zeros(16, 8192, **bf)
        self.ctr = torch.zeros(int(attrs[3]) // 4, dtype=torch.int32, device=device)
        self.sched = torch.zeros(int(attrs[4]) // 4, dtype=torch.int32, device=device)
        init = [0] * self.EV_NUM
        for e in MIN_EVENTS:
            init[e] = U64MAX
        self.ev_init = torch.tensor([x - (1 << 64) if x >= (1 << 63) else x for x in init], dtype=torch.int64, device=device)
        self.ev = self.ev_init.clone()
        self.ev_out = torch.zeros(self.EV_NUM, dtype=torch.int64, device=device)
        self.LOGW = int(attrs[8])
        self.tlog = torch.zeros(self.G * self.LOGN * self.LOGW, dtype=torch.int64, device=device)
        self.log_n = torch.zeros(self.G, dtype=torch.int32, device=device)
        self.sink = torch.zeros(4, dtype=torch.int32, device=device)
        names = ("w_router", "w_pool", "w_exp", "w_unp", "loc_ids", "xr", "h", "logits", "p", "pnorm", "g13", "act", "y",
                 "ynorm", "m", "f", "sprime", "a2b2", "bnd", "out_stream", "out", "ctr", "sched", "ev", "ev_out", "ev_init",
                 "tlog", "log_n", "sink")
        self.ptrs = torch.tensor([getattr(self, n).data_ptr() for n in names], dtype=torch.int64)

    def set_routing(self, loc):
        """loc: [T, 8] local expert ids (0..31) or -1 for remote."""
        t = self.torch
        flat = t.full((16 * 8,), -1, dtype=t.int32)
        v = t.as_tensor(loc, dtype=t.int32).reshape(-1)
        flat[: v.numel()] = v
        self.loc_ids.copy_(flat.to(self.loc_ids.device))

    def caller(self, T, coupled=False, rc_gbs=30.0, alpha=0.3):
        t = self.torch
        n_unp = 256 if T <= 8 else 2048
        icfg = t.tensor([T, self.G, n_unp, n_unp // 4, 1 if coupled else 0], dtype=t.int32)
        fcfg = t.tensor([2.0 / rc_gbs, alpha], dtype=t.float32)
        mod, ptrs = self.mod, self.ptrs
        return lambda *_: mod.fe16_launch(ptrs, icfg, fcfg)

    def read_marks(self):
        t = self.torch
        ev = [int(x) & U64MAX for x in self.ev_out.cpu().tolist()]
        ln = self.log_n.cpu().tolist()
        L = self.tlog.view(self.G, self.LOGN, self.LOGW).cpu()
        recs = []
        for c in range(self.G):
            for r in L[c, : min(ln[c], self.LOGN)].tolist():
                w0, w1 = r[0] & U64MAX, r[1] & U64MAX
                recs.append(dict(kind=KIND[w0 & 15], wg=(w0 >> 4) & 15, cta=(w0 >> 8) & 0xFFFFFF, bytes=w0 >> 32,
                                 idx=w1 & 0xFFFFFFFF, aux=w1 >> 32, t_grab=r[2] & U64MAX, t_issue=r[3] & U64MAX,
                                 t_full=r[4] & U64MAX, t_dep=r[5] & U64MAX, t_end=r[6] & U64MAX))
        n_act = int(self.sched[0].item())
        err = ev[EV["NUM"] - 1]
        return dict(ev=ev, recs=recs, n_act=n_act, err=err, log_overflow=sum(1 for x in ln if x > self.LOGN))


def analyze(mk, T, coupled=False):
    """Chain table + landing timeline from one marked replay (times in us, relative to EV_START)."""
    ev, recs = mk["ev"], mk["recs"]
    t0 = ev[EV["START"]]
    us = lambda x: (x - t0) / 1e3  # noqa: E731
    E = {k: us(ev[v]) for k, v in EV.items() if k not in ("G13", "NUM") and 0 < ev[v] < U64MAX}
    n_act = mk["n_act"]
    g13 = {e: us(ev[EV["G13"] + e]) for e in range(n_act) if ev[EV["G13"] + e]}
    by = {}
    for r in recs:
        by.setdefault(r["kind"], []).append(r)
    out = {"span": E.get("END", float("nan")), "events": E, "n_act": n_act, "err": mk["err"],
           "tasks": {k: len(v) for k, v in by.items()}, "log_overflow": mk["log_overflow"]}

    def hop(kind, t_done, sel=lambda r: True):
        # decoupled: the first consumer that was already waiting (data landed before the counter completed) passes;
        # coupled: the first producer issue after the counter completed (the producer is the waiter there)
        if coupled:
            w = [r for r in by.get(kind, []) if sel(r) and us(r["t_grab"]) < t_done]
            return (min(us(r["t_issue"]) for r in w) - t_done) if w else None
        w = [r for r in by.get(kind, []) if sel(r) and us(r["t_full"]) < t_done]
        if not w:
            return None
        return min(us(r["t_dep"]) for r in w) - t_done

    ch = {}
    if "ROUTER_DONE" in E and "SCHED" in E:
        ch["route_task"] = E["SCHED"] - E["ROUTER_DONE"]
    if "SCHED" in E and "DYN_REACH" in E:
        ch["sched_slack"] = E["DYN_REACH"] - E["SCHED"]
    if "POOL_LAST" in E:
        ch["hop_pool_w13"] = hop("W13", E["POOL_LAST"])
    hops2 = [hop("W2", g13[e], lambda r, e=e: r["aux"] == e) for e in g13]
    hops2 = [h for h in hops2 if h is not None]
    ch["hop_g13_w2_max"] = max(hops2) if hops2 else None
    ch["hop_g13_w2_med"] = sorted(hops2)[len(hops2) // 2] if hops2 else None
    if "Y_DONE" in E:
        ch["hop_y_comb"] = hop("COMB", E["Y_DONE"])
        cd = [us(r["t_dep"]) for r in by.get("COMB", [])]
        if cd and "M_DONE" in E:
            ch["comb_work"] = E["M_DONE"] - min(cd)
    if "M_DONE" in E:
        ch["hop_m_unp"] = hop("UNP", E["M_DONE"])
    if "UNP_DONE" in E:
        ch["hop_unp_final"] = hop("FINAL", E["UNP_DONE"])
        fd = [us(r["t_dep"]) for r in by.get("FINAL", [])]
        if fd and "END" in E:
            ch["final_work"] = E["END"] - min(fd)
    wt = [r for r in recs if r["bytes"] > 0]
    if wt:
        last_full = max(us(r["t_full"]) for r in wt)
        first_full = min(us(r["t_full"]) for r in wt)
        tot = sum(r["bytes"] for r in wt)
        ch["tail_after_last_byte"] = E.get("END", float("nan")) - last_full
        ch["head_to_first_byte"] = first_full
        out["bytes_MB"] = tot / 1e6
        out["stream_TBs"] = tot / ((last_full - first_full) * 1e-6) / 1e12 if last_full > first_full else None
        # issue and consumption timelines: bytes per 0.5 us bin by t_issue / t_full; idle = bins below half of 4.4 TB/s
        b = 0.5
        nb = int(max(last_full, max(us(r["t_issue"]) for r in wt)) / b) + 1
        land = [0.0] * nb
        iss = [0.0] * nb
        for r in wt:
            land[min(int(us(r["t_full"]) / b), nb - 1)] += r["bytes"]
            iss[min(int(us(r["t_issue"]) / b), nb - 1)] += r["bytes"]
        thr = 2.2e12 * b * 1e-6
        first_iss = min(us(r["t_issue"]) for r in wt)
        last_iss = max(us(r["t_issue"]) for r in wt)
        out["issue_idle_us"] = sum(b for i in range(int(first_iss / b), int(last_iss / b) + 1) if iss[i] < thr)
        idle = [i * b for i in range(int(first_full / b), int(last_full / b) + 1) if land[i] < thr]
        out["idle_us"] = len(idle) * b
        # attribute idle bins to the phase that lands next
        ph = {}
        kinds_by_bin = {}
        for r in wt:
            kinds_by_bin.setdefault(int(us(r["t_full"]) / b), set()).add(r["kind"])
        for x in idle:
            i = int(x / b)
            nxt = next((kinds_by_bin[j] for j in range(i, nb) if j in kinds_by_bin), {"?"})
            key = "/".join(sorted(nxt))
            ph[key] = ph.get(key, 0.0) + b
        out["idle_by_next_phase"] = ph
    out["chain"] = ch
    exp_tiles = len(by.get("W13", [])) + len(by.get("W2", []))
    out["expert_tiles_ok"] = (exp_tiles == n_act * 1536)
    return out

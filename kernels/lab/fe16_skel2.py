"""FE16 step-0, v2 (R-fe-2, notes/memory/gpu_requests.md): the R-fe-1 skeleton with the per-tile protocol taken off
the consumer path. R-fe-1's kernels/lab/fe16_skel.py stays as it ran.

One persistent launch, 1 CTA per SM, 320 threads.
- Warp 0 is the weight producer. It bulk-copies tiles into FE_NS x 24 KB slots and never waits on data (COUPLED arm:
  it waits for every task's dependency before issuing). After sched_ready it copies the schedule into smem.
- Warps 1-4 and 5-8 are two consumer warp groups taking alternate slots. Per tile:
  - wait for the slot; on a bounded-wait timeout, read the error flag and exit;
  - check the dependency once per (dependency, warp group): the result is cached in a bitmask;
  - read the X slice (independent loads; token indices from the smem schedule) and the tile;
  - spin for the modelled compute; release the slot;
  - red.add the dummy partials;
  - the leader pushes a completion record into a smem queue and moves on. It never waits on a gpu-scope fence or an
    atomic return.
- Warp 9 is the helper. It drains the queues. Per batch it issues one fence.acq_rel.gpu, then relaxed red.adds on the
  split-K row counters (fire-and-forget). Epilogues are dependent tasks: the CTA that ran a row tile's last K-split owns
  its epilogue (pooler norm partials, silu-mul, post-expert norm partials, boundary partials). It runs the epilogue once
  the row counter is full, then increments the phase counter, with a return for the marks; this is off the consumer
  path. CTA 0's helper also runs the routing task (dependent on the router counter).
- COMB (m combine) and FINAL (s'' / out) are consumer tasks with one column per thread and independent loads.
No __trap anywhere: bounded waits set an error bit and exit.
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

namespace fe16b {
constexpr int NS = FE_NS;
constexpr int SLOT = 24576;
constexpr int WGT = 128;
constexpr int NTHR = 32 + 2 * WGT + 32;
constexpr int RB = 16384, PB = 24576, EB = 18432;
constexpr int N_ROUTER = 512, N_POOL = 2048, POOL_RT = 256;
constexpr int W13_PER_E = 1024, W2_PER_E = 512, W13_RT = 128, W2_RT = 64;
constexpr unsigned long long E_BYTES = 1536ull * EB;
constexpr int NCOMB = 32, NFINAL = 64, MAXT = 16, NE = 32, LOGN = 512, LOGW = 7, QN = 32, PN = 96;
constexpr unsigned long long TIMEOUT_NS = 50000000ull;
enum { K_END = 0, K_ROUTER, K_POOL, K_W13, K_W2, K_COMB, K_UNP, K_FINAL };
enum { E_POOL = 1, E_W13, E_W2, E_UNP };
enum { EV_START = 0, EV_ROUTER_DONE, EV_SCHED, EV_DYN_REACH, EV_SCHED_SEEN, EV_POOL_LAST, EV_Y_DONE, EV_M_DONE,
       EV_UNP_DONE, EV_END, EV_PROD_ALL, EV_G13 = 32, EV_NUM = 72 };

struct Ctr {
  unsigned router_cnt, sched_ready, q_head, pool_done, y_done, m_done, unp_done, fin_cnt, prod_done, err, help_done;
  unsigned pad[5];
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
struct Smem {
  Desc desc[NS];
  unsigned long long full[NS], empty[NS];
  unsigned long long tdep[2];
  int logn, cons_done, pad0, pad1;
  int cq_head[2], cq_tail[2];
  int4 cq[2][QN];
  int4 pend[PN];
  float bndb[2][MAXT * 4];
  int rsc[104];
  Sched sched;
};
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
__device__ __forceinline__ void fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_cta() { asm volatile("fence.acq_rel.cta;" ::: "memory"); }
__device__ __forceinline__ void red_u32(unsigned* p, unsigned v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_add(unsigned* p, unsigned v) {
  unsigned o; asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }
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
// bounded waits: on timeout set the error bit and return false (the caller exits cleanly); no __trap
__device__ __forceinline__ bool mb_wait(unsigned long long* b, unsigned par, unsigned* err) {
  if (mb_test(b, par)) return true;
  const unsigned long long t0 = gtime();
  while (!mb_test(b, par)) { if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 2u); return false; } }
  return true; }
__device__ __forceinline__ bool wait_ge(const unsigned* p, unsigned target, unsigned* err) {
  if (ld_acq(p) >= target) return true;
  const unsigned long long t0 = gtime();
  while (ld_acq(p) < target) { if (gtime() - t0 > TIMEOUT_NS) { atomicOr(err, 1u); return false; } }
  return true; }
__device__ __forceinline__ void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" :: "r"(1 + wg) : "memory"); }

__device__ __forceinline__ const unsigned* dep_of(const Params& P, int kind, int aux, unsigned& target, int n_act) {
  switch (kind) {
    case K_W13: target = POOL_RT; return &P.ctr->pool_done;
    case K_W2: target = W13_RT; return &P.ctr->g13_done[aux];
    case K_COMB: target = (unsigned)(W2_RT * n_act); return &P.ctr->y_done;
    case K_UNP: target = NCOMB; return &P.ctr->m_done;
    case K_FINAL: target = (unsigned)P.n_unp_rt; return &P.ctr->unp_done;
    default: target = 0; return nullptr;
  } }

__global__ void __launch_bounds__(NTHR, 1) fe16b_kernel(const Params P) {
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* ring = smem;
  Smem* S = (Smem*)(smem + NS * SLOT);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cta = blockIdx.x;
  unsigned* err = &P.ctr->err;
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) {
#if __CUDA_ARCH__ >= 900
      mb_init(&S->full[s], 1);
#else
      mb_init(&S->full[s], 33);
#endif
      mb_init(&S->empty[s], 1);
    }
    S->logn = 0; S->cons_done = 0; S->cq_head[0] = S->cq_head[1] = 0; S->cq_tail[0] = S->cq_tail[1] = 0;
#if __CUDA_ARCH__ >= 900
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
    atomicMin(&P.ev[EV_START], gtime());
  }
  __syncthreads();
  const int T = P.T;
  if (warp == 0) {
    // ============================================================== producer
    int i = 0, phase = 0, k = 0, n_act = 0, N13 = 0, N2 = 0, Ndyn = 0;
    bool sched_seen = false, ok = true;
    for (;;) {
      int kind = K_END, idx = 0, bytes = 0, aux = 0; const char* src = nullptr; unsigned long long t_grab = 0;
      if (phase == 2 && !sched_seen) {                     // whole warp: wait sched_ready, copy the schedule to smem
        if (lane == 0) { atomicMin(&P.ev[EV_DYN_REACH], gtime()); ok = wait_ge(&P.ctr->sched_ready, 1u, err); atomicMin(&P.ev[EV_SCHED_SEEN], gtime()); }
        ok = __shfl_sync(0xffffffffu, ok, 0);
        const int* gs = (const int*)P.sched; int* ss = (int*)&S->sched;
        for (int q = lane; q < (int)(sizeof(Sched) / 4); q += 32) ss[q] = ld_cg_i(gs + q);
        __syncwarp();
        n_act = S->sched.n_act; N13 = n_act * W13_PER_E; N2 = n_act * W2_PER_E; Ndyn = N13 + N2 + NCOMB;
        sched_seen = true;
      }
      if (lane == 0) {
        for (;;) {
          if (phase == 0) { idx = cta + k * P.G; if (idx < N_ROUTER) { kind = K_ROUTER; bytes = RB; src = P.w_router + (size_t)idx * RB; ++k; break; } phase = 1; k = 0; }
          if (phase == 1) { idx = cta + k * P.G; if (idx < N_POOL) { kind = K_POOL; bytes = PB; src = P.w_pool + (size_t)idx * PB; ++k; break; } phase = 2; k = 0; }
          if (phase == 2) {
            if (!sched_seen) { kind = -1; break; }         // leave the selection; the warp copies the schedule first
            const int j = (int)atomicAdd(&P.ctr->q_head, 1u);
            if (j < N13) { const int e = j / W13_PER_E, t = j % W13_PER_E; kind = K_W13; bytes = EB; aux = e; idx = t;
              src = P.w_exp + (size_t)S->sched.act_le[e] * E_BYTES + (size_t)t * EB; break; }
            if (j < N13 + N2) { const int jj = j - N13, e = jj / W2_PER_E, t = jj % W2_PER_E; kind = K_W2; bytes = EB; aux = e; idx = t;
              src = P.w_exp + (size_t)S->sched.act_le[e] * E_BYTES + (size_t)(W13_PER_E + t) * EB; break; }
            if (j < Ndyn) { kind = K_COMB; idx = j - N13 - N2; bytes = 0; break; }
            phase = 3; k = 0;
          }
          if (phase == 3) { idx = cta + k * P.G; if (idx < P.n_unp) { kind = K_UNP; bytes = PB; src = P.w_unp + (size_t)idx * PB; ++k; break; } phase = 4; k = 0; }
          if (phase == 4) { idx = cta + k * P.G; if (idx < NFINAL) { kind = K_FINAL; bytes = 0; ++k; break; } }
          kind = K_END; break;
        }
        t_grab = gtime();
        if (kind > K_END && P.coupled && ok) {             // paused arm: no issue before the dependency is met
          unsigned tg; const unsigned* dp = dep_of(P, kind, aux, tg, n_act);
          if (dp) ok = wait_ge(dp, tg, err);
        }
      }
      kind = __shfl_sync(0xffffffffu, kind, 0);
      phase = __shfl_sync(0xffffffffu, phase, 0);          // the phase lives in lane 0; the schedule copy needs the warp
      if (kind < 0) continue;                              // schedule copy pending (phase 2 entry)
      idx = __shfl_sync(0xffffffffu, idx, 0);
      bytes = __shfl_sync(0xffffffffu, bytes, 0);
      aux = __shfl_sync(0xffffffffu, aux, 0);
      src = (const char*)__shfl_sync(0xffffffffu, (unsigned long long)src, 0);
      t_grab = __shfl_sync(0xffffffffu, t_grab, 0);
      ok = __shfl_sync(0xffffffffu, ok, 0);
      if (!ok) kind = K_END;
      const int nissue = (kind == K_END) ? 2 : 1;
      for (int r = 0; r < nissue; ++r) {
        const int s = i % NS; const unsigned par = ((i / NS) & 1) ^ 1;
        bool w = true;
        if (lane == 0) w = mb_wait(&S->empty[s], par, err);
        w = __shfl_sync(0xffffffffu, w, 0);
        if (!w) { ok = false; break; }
        if (lane == 0) { Desc& d = S->desc[s]; d.kind = kind; d.idx = idx; d.bytes = bytes; d.aux = aux; d.t_grab = t_grab; d.t_issue = gtime(); }
        __syncwarp();
        unsigned char* dst = ring + (size_t)s * SLOT;
#if __CUDA_ARCH__ >= 900
        if (lane == 0) {
          if (bytes > 0) {
            asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared.b64 st, [%0], %1;\n}" :: "r"(su32(&S->full[s])), "r"(bytes) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         :: "r"(su32(dst)), "l"(src), "r"(bytes), "r"(su32(&S->full[s])) : "memory");
          } else {
            mb_arrive(&S->full[s]);
          }
        }
#else
        for (int off = lane * 16; off < bytes; off += 512)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(su32(dst) + off), "l"(src + off) : "memory");
        asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" :: "r"(su32(&S->full[s])) : "memory");
        if (lane == 0) mb_arrive(&S->full[s]);
#endif
        ++i;
      }
      if (kind == K_END || !ok) break;
    }
    if (lane == 0) { const unsigned o = atomicAdd(&P.ctr->prod_done, 1u); if (o + 1 == (unsigned)P.G) P.ev[EV_PROD_ALL] = gtime(); }
  } else if (warp <= 8) {
    // ============================================================== consumers
    const int wg = (warp - 1) >> 2, wt = tid - 32 - wg * WGT;
    unsigned acc = 0, g13_ok = 0;
    bool pool_ok = false, y_ok = false, m_ok = false, unp_ok = false;
    for (int i = wg;; i += 2) {
      const int s = i % NS; const unsigned par = (i / NS) & 1;
      if (!mb_wait(&S->full[s], par, err)) break;          // timeout path only
      const unsigned long long t_full = gtime();
      const int kind = S->desc[s].kind, idx = S->desc[s].idx, bytes = S->desc[s].bytes, aux = S->desc[s].aux;
      const unsigned long long t_issue = S->desc[s].t_issue, t_grab = S->desc[s].t_grab;
      if (kind == K_END) break;
      const int n_act = (kind >= K_W13) ? S->sched.n_act : 0;
      if (wt == 0) {                                       // dependency: acquired once per (dependency, warp group)
        bool need = false;
        if (kind == K_W13) need = !pool_ok; else if (kind == K_W2) need = !((g13_ok >> aux) & 1u);
        else if (kind == K_COMB) need = !y_ok; else if (kind == K_UNP) need = !m_ok; else if (kind == K_FINAL) need = !unp_ok;
        if (need) {
          unsigned tg; const unsigned* dp = dep_of(P, kind, aux, tg, n_act);
          wait_ge(dp, tg, err);
          if (kind == K_W13) pool_ok = true; else if (kind == K_W2) g13_ok |= 1u << aux;
          else if (kind == K_COMB) y_ok = true; else if (kind == K_UNP) m_ok = true; else if (kind == K_FINAL) unp_ok = true;
        }
        S->tdep[wg] = gtime();
      }
      wg_bar(wg);
      const unsigned long long t_dep = S->tdep[wg];
      const int ntok = (kind == K_W13 || kind == K_W2) ? S->sched.act_n[aux] : T;
      if (kind == K_ROUTER) {
        const int ks = idx & 7;
        for (int q = wt; q < T * 256; q += WGT) { uint4 v = ldg4(P.xr + (size_t)(q >> 8) * 8192 + ks * 1024 + (q & 255) * 4); acc ^= v.x ^ v.w; }
      } else if (kind == K_POOL) {
        const int ks = idx & 7;
        for (int q = wt; q < T * 128; q += WGT) { uint4 v = ldg4(P.h + (size_t)(q >> 7) * 8192 + ks * 1024 + (q & 127) * 8); acc ^= v.x ^ v.w; }
      } else if (kind == K_W13) {
        const int ks = idx & 7;
        for (int q = wt; q < ntok * 128; q += WGT) {
          const int tok = S->sched.act_tok[aux][q >> 7];
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
      if (wt == 0 && bytes > 0 && P.ns_per_byte > 0.f) {
        const int ng = ((kind == K_W13 || kind == K_W2) ? (ntok + 7) / 8 : (T + 7) / 8);
        const float cost = 1.0f + P.alpha * (float)(ng > 0 ? ng - 1 : 0);
        const unsigned long long until = t_dep + (unsigned long long)((float)bytes * P.ns_per_byte * cost);
        while (gtime() < until) {}
      }
      wg_bar(wg);
      if (wt == 0) mb_arrive(&S->empty[s]);
      if (kind == K_ROUTER) {
        const int eb = idx >> 3;
        for (int q = wt; q < 4 * T; q += WGT) red_f32(P.logits + (q >> 2) * 256 + eb * 4 + (q & 3), 0.0f);
      } else if (kind == K_POOL) {
        const int rt = idx >> 3;
        for (int q = wt; q < 16 * T; q += WGT) red_f32(P.p + (q >> 4) * 4096 + rt * 16 + (q & 15), 0.0f);
      } else if (kind == K_W13) {
        const int rt = idx >> 3;
        for (int q = wt; q < 64 * ntok; q += WGT) red_f32(P.g13 + (size_t)(aux * MAXT + (q >> 6)) * 8192 + rt * 64 + (q & 63), 0.0f);
      } else if (kind == K_W2) {
        const int rt = idx >> 3;
        for (int q = wt; q < 64 * ntok; q += WGT) red_f32(P.y + (size_t)(aux * MAXT + (q >> 6)) * 4096 + rt * 64 + (q & 63), 0.0f);
      } else if (kind == K_UNP) {
        const int rt = idx >> 2;
        for (int q = wt; q < 16 * T; q += WGT) red_f32(P.f + (q >> 4) * 8192 + rt * 16 + (q & 15), 0.0f);
      } else if (kind == K_COMB) {                         // one column per thread, every (token, slot) load independent
        const int col = idx * 128 + wt;
        for (int t = 0; t < T; ++t) {
          float a = 0.f;
#pragma unroll
          for (int kk = 0; kk < 8; ++kk) {
            const int sl = S->sched.tok_slot[t][kk];
            if (sl >= 0) a += ldgf(P.ynorm + sl) * ldgf(P.y + (size_t)sl * 4096 + col); }
          P.m[(size_t)t * 4096 + col] = __float2bfloat16(a); }
      } else if (kind == K_FINAL) {                        // bnd broadcast through smem, then independent loads
        const int col = idx * 128 + wt;
        if (wt < T * 3) S->bndb[wg][wt] = ldgf(P.bnd + (wt / 3) * 4 + (wt % 3));
        const float a2 = ldgf(P.a2b2 + col), b2 = ldgf(P.a2b2 + 8192 + col);
        wg_bar(wg);
        for (int t = 0; t < T; ++t) {
          const float su = S->bndb[wg][t * 3], bf = S->bndb[wg][t * 3 + 1], cf = S->bndb[wg][t * 3 + 2];
          const float r = rsqrtf(su / 8192.f + 1e-8f) + 0.f * (bf + cf);
          const float sp = ldgf(P.sprime + t * 8192 + col), u = ldgf(P.f + t * 8192 + col);
          const float v = a2 * sp + b2 * u * r;
          P.out_stream[t * 8192 + col] = v; P.out[t * 8192 + col] = __float2bfloat16(v * 0.5f); }
      }
      wg_bar(wg);                                          // the warp group's writes precede the record (CTA scope)
      if (wt == 0) {
        volatile int* tl = &S->cq_tail[wg]; volatile int* hd = &S->cq_head[wg];
        const int t0 = *tl;
        const unsigned long long w0 = gtime();
        while (t0 - *hd >= QN) { if (gtime() - w0 > TIMEOUT_NS) { atomicOr(err, 4u); break; } }
        S->cq[wg][t0 % QN] = make_int4(kind, idx, aux, 0);
        fence_cta();
        *tl = t0 + 1;
        const int n = atomicAdd(&S->logn, 1);
        if (n < LOGN) {
          unsigned long long* L = P.tlog + ((size_t)cta * LOGN + n) * LOGW;
          L[0] = (unsigned long long)kind | ((unsigned long long)wg << 4) | ((unsigned long long)cta << 8) | ((unsigned long long)(unsigned)bytes << 32);
          L[1] = (unsigned long long)(unsigned)idx | ((unsigned long long)(unsigned)aux << 32);
          L[2] = t_grab; L[3] = t_issue; L[4] = t_full; L[5] = t_dep; L[6] = gtime(); }
      }
    }
    if (wt == 0) { fence_cta(); atomicAdd(&S->cons_done, 1); }
    if (acc == 0x9e3779b9u) atomicAdd(P.sink, acc);
  } else {
    // ============================================================== helper (warp 9)
    int qh0 = 0, qh1 = 0, npend = 0;
    bool routed = (cta != 0), final_owner = false;
    unsigned long long idle_t0 = gtime();
    for (;;) {
      bool progress = false;
      // 1. drain the completion queues: one gpu-scope fence per batch, then fire-and-forget row counters
      int nrec = 0;
      if (lane == 0) {
        const int t0 = *(volatile int*)&S->cq_tail[0], t1 = *(volatile int*)&S->cq_tail[1];
        fence_cta();
        nrec = (t0 - qh0) + (t1 - qh1);
        if (nrec) {
          fence_gpu();
          for (int r = 0; r < nrec; ++r) {
            const int4 rc4 = (qh0 != t0) ? S->cq[0][(qh0++) % QN] : S->cq[1][(qh1++) % QN];   // entries stay valid until head moves
            const int kind = rc4.x, idx = rc4.y, aux = rc4.z;
            int ekind = 0, erow = 0;
            if (kind == K_ROUTER) red_u32(&P.ctr->router_cnt, 1u);
            else if (kind == K_POOL) { red_u32(&P.ctr->pool_rt[idx >> 3], 1u); if ((idx & 7) == 7) { ekind = E_POOL; erow = idx >> 3; } }
            else if (kind == K_W13) { red_u32(&P.ctr->g13_rt[aux * W13_RT + (idx >> 3)], 1u); if ((idx & 7) == 7) { ekind = E_W13; erow = idx >> 3; } }
            else if (kind == K_W2) { red_u32(&P.ctr->y_rt[aux * W2_RT + (idx >> 3)], 1u); if ((idx & 7) == 7) { ekind = E_W2; erow = idx >> 3; } }
            else if (kind == K_UNP) { red_u32(&P.ctr->unp_rt[idx >> 2], 1u); if ((idx & 3) == 3) { ekind = E_UNP; erow = idx >> 2; } }
            else if (kind == K_COMB) { if (atom_add(&P.ctr->m_done, 1u) + 1 == NCOMB) P.ev[EV_M_DONE] = gtime(); }
            else if (kind == K_FINAL) { if (atom_add(&P.ctr->fin_cnt, 1u) + 1 == NFINAL) { P.ev[EV_END] = gtime(); final_owner = true; } }
            if (ekind) { if (npend < PN) S->pend[npend++] = make_int4(ekind, aux, erow, 0); else atomicOr(err, 8u); }
          }
          *(volatile int*)&S->cq_head[0] = qh0; *(volatile int*)&S->cq_head[1] = qh1;
        }
      }
      nrec = __shfl_sync(0xffffffffu, nrec, 0);
      if (nrec) progress = true;
      // 2. dependent epilogues: run the first pending one whose row counter is full
      int ek = 0, ee = 0, er = 0, ep = -1;
      if (lane == 0) {
        for (int pidx = 0; pidx < npend; ++pidx) {
          const int4 pe = S->pend[pidx];
          const unsigned* c; unsigned tg;
          if (pe.x == E_POOL) { c = &P.ctr->pool_rt[pe.z]; tg = 8; }
          else if (pe.x == E_W13) { c = &P.ctr->g13_rt[pe.y * W13_RT + pe.z]; tg = 8; }
          else if (pe.x == E_W2) { c = &P.ctr->y_rt[pe.y * W2_RT + pe.z]; tg = 8; }
          else { c = &P.ctr->unp_rt[pe.z]; tg = 4; }
          if (ld_acq(c) >= tg) { ek = pe.x; ee = pe.y; er = pe.z; ep = pidx; break; }
        }
      }
      ek = __shfl_sync(0xffffffffu, ek, 0); ee = __shfl_sync(0xffffffffu, ee, 0); er = __shfl_sync(0xffffffffu, er, 0);
      if (ek) {
        progress = true;
        if (ek == E_POOL) {
          if (lane < T) { float ss = 0.f;
#pragma unroll
            for (int c = 0; c < 16; ++c) { const float v = ldgf(P.p + lane * 4096 + er * 16 + c); ss += v * v; }
            red_f32(P.pnorm + lane, ss); }
        } else if (ek == E_W13) {
          const int ntok = S->sched.act_n[ee];
          for (int q = lane; q < 32 * ntok; q += 32) {
            const int j = q >> 5, c = q & 31;
            const float* g = P.g13 + (size_t)(ee * MAXT + j) * 8192 + er * 64;
            const float gv = ldgf(g + 2 * c), uv = ldgf(g + 2 * c + 1);
            P.act[(size_t)(ee * MAXT + j) * 4096 + er * 32 + c] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv); }
        } else if (ek == E_W2) {
          const int ntok = S->sched.act_n[ee];
          for (int j = 0; j < ntok; ++j) {
            const float* yy = P.y + (size_t)(ee * MAXT + j) * 4096 + er * 64;
            const float v0 = ldgf(yy + lane), v1 = ldgf(yy + 32 + lane);
            float ss = v0 * v0 + v1 * v1;
            for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (lane == 0) red_f32(P.ynorm + ee * MAXT + j, ss); }
        } else {
          if (lane < T) { float su = 0.f, bf = 0.f, cf = 0.f;
#pragma unroll
            for (int c = 0; c < 16; ++c) { const int col = er * 16 + c;
              const float u = ldgf(P.f + lane * 8192 + col), sp = ldgf(P.sprime + lane * 8192 + col);
              const float a2 = ldgf(P.a2b2 + col), b2 = ldgf(P.a2b2 + 8192 + col);
              su += u * u; bf += a2 * b2 * sp * u; cf += b2 * b2 * u * u; }
            red_f32(P.bnd + lane * 4, su); red_f32(P.bnd + lane * 4 + 1, bf); red_f32(P.bnd + lane * 4 + 2, cf); }
        }
        __syncwarp();
        if (lane == 0) {
          fence_gpu();
          if (ek == E_POOL) { if (atom_add(&P.ctr->pool_done, 1u) + 1 == POOL_RT) P.ev[EV_POOL_LAST] = gtime(); }
          else if (ek == E_W13) { if (atom_add(&P.ctr->g13_done[ee], 1u) + 1 == W13_RT) P.ev[EV_G13 + ee] = gtime(); }
          else if (ek == E_W2) { if (atom_add(&P.ctr->y_done, 1u) + 1 == (unsigned)(W2_RT * S->sched.n_act)) P.ev[EV_Y_DONE] = gtime(); }
          else { if (atom_add(&P.ctr->unp_done, 1u) + 1 == (unsigned)P.n_unp_rt) P.ev[EV_UNP_DONE] = gtime(); }
          S->pend[ep] = S->pend[npend - 1]; --npend;
        }
        __syncwarp();
      }
      // 3. routing (CTA 0's helper): a dependent task on the router counter
      if (!routed) {
        unsigned rc = 0;
        if (lane == 0) rc = ld_acq(&P.ctr->router_cnt);
        rc = __shfl_sync(0xffffffffu, rc, 0);
        if (rc >= (unsigned)N_ROUTER) {
          if (lane == 0) P.ev[EV_ROUTER_DONE] = gtime();
          float mx = -1e30f;
          for (int q = lane; q < T * 256; q += 32) mx = fmaxf(mx, ldgf(P.logits + q));
          if (mx == 12345.0f) atomicAdd(P.sink, 1u);
          int* rsc = S->rsc;
          for (int q = lane; q < 101; q += 32) rsc[q] = 0;
          __syncwarp();
          for (int q = lane; q < T * 8; q += 32) { const int le = P.loc_ids[q]; if (le >= 0 && le < NE) atomicAdd(&rsc[le], 1); }
          __syncwarp();
          { const int c = rsc[lane]; int rank = -1;
            if (c) { rank = 0; for (int j = 0; j < NE; ++j) { const int cj = rsc[j]; rank += (cj > c || (cj == c && j < lane)) ? 1 : 0; } }
            rsc[32 + lane] = rank;
            if (rank >= 0) { P.sched->act_le[rank] = lane; P.sched->act_n[rank] = c; }
            const unsigned na = __popc(__ballot_sync(0xffffffffu, c != 0));
            if (lane == 0) rsc[100] = (int)na; }
          __syncwarp();
          for (int q = lane; q < T * 8; q += 32) {
            const int le = P.loc_ids[q]; int sl = -1;
            if (le >= 0 && le < NE) { const int a = rsc[32 + le]; const int pos = atomicAdd(&rsc[64 + a], 1); P.sched->act_tok[a][pos] = q >> 3; sl = a * MAXT + pos; }
            P.sched->tok_slot[q >> 3][q & 7] = sl; }
          __syncwarp();
          if (lane == 0) { P.sched->n_act = rsc[100]; fence_gpu(); st_rel(&P.ctr->sched_ready, 1u); P.ev[EV_SCHED] = gtime(); }
          routed = true; progress = true;
        }
      }
      // exit once both consumer warp groups have ended and nothing is queued or pending
      int done = 0;
      if (lane == 0) done = (*(volatile int*)&S->cons_done == 2 && qh0 == *(volatile int*)&S->cq_tail[0] &&
                             qh1 == *(volatile int*)&S->cq_tail[1] && npend == 0 && routed) ? 1 : 0;
      done = __shfl_sync(0xffffffffu, done, 0);
      if (done) break;
      if (progress) idle_t0 = gtime();
      else { if (gtime() - idle_t0 > TIMEOUT_NS) { if (lane == 0) atomicOr(err, 16u); break; } __nanosleep(64); }
    }
    if (lane == 0) atomicAdd(&P.ctr->help_done, 1u);
    final_owner = __shfl_sync(0xffffffffu, final_owner, 0);
    if (final_owner) {                                     // the helper that completed FINAL resets for the next call
      if (lane == 0) { wait_ge(&P.ctr->help_done, (unsigned)P.G, err); wait_ge(&P.ctr->prod_done, (unsigned)P.G, err); fence_gpu(); }
      __syncwarp();
      for (int q = lane; q < EV_NUM; q += 32) { P.ev_out[q] = P.ev[q]; P.ev[q] = P.ev_init[q]; }
      Ctr* C = P.ctr;
      const int na = S->sched.n_act > 0 ? S->sched.n_act : NE;
      for (int q = lane; q < POOL_RT; q += 32) C->pool_rt[q] = 0;
      for (int q = lane; q < na * W13_RT; q += 32) C->g13_rt[q] = 0;
      for (int q = lane; q < NE; q += 32) C->g13_done[q] = 0;
      for (int q = lane; q < na * W2_RT; q += 32) C->y_rt[q] = 0;
      for (int q = lane; q < 512; q += 32) C->unp_rt[q] = 0;
      __syncwarp();
      if (lane == 0) { P.ev_out[EV_NUM - 1] = (unsigned long long)ld_acq(&C->err);
        C->router_cnt = 0; C->sched_ready = 0; C->q_head = 0; C->pool_done = 0; C->y_done = 0; C->m_done = 0;
        C->unp_done = 0; C->fin_cnt = 0; C->prod_done = 0; C->help_done = 0; }
      fence_gpu();
    }
  }
  __syncthreads();
  if (tid == 0) P.log_n[cta] = S->logn;
}

__global__ void gtres_kernel(unsigned long long* o) {
  unsigned long long prev = gtime(), mind = ~0ull, n = 0, t0 = prev; long long c0 = clock64();
  for (int i = 0; i < 2000000; ++i) { unsigned long long t = gtime(); if (t != prev) { unsigned long long d = t - prev; if (d < mind) mind = d; ++n; prev = t; } }
  o[0] = mind; o[1] = n; o[2] = prev - t0; o[3] = (unsigned long long)(clock64() - c0); }

constexpr int SMEM = NS * SLOT + (int)sizeof(Smem);
}  // namespace fe16b

void fe16_launch(torch::Tensor ptrs, torch::Tensor icfg, torch::Tensor fcfg) {
  using namespace fe16b;
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
  if (!attr) { cudaFuncSetAttribute(fe16b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); attr = true; }
  fe16b_kernel<<<P.G, NTHR, SMEM, at::cuda::getCurrentCUDAStream()>>>(P);
}

std::vector<int64_t> fe16_gtres() {
  unsigned long long* d; cudaMalloc(&d, 32); fe16b::gtres_kernel<<<1, 1>>>(d); unsigned long long h[4];
  cudaMemcpy(h, d, 32, cudaMemcpyDeviceToHost); cudaFree(d);
  return {(int64_t)h[0], (int64_t)h[1], (int64_t)h[2], (int64_t)h[3]};
}

std::vector<int64_t> fe16_attrs() {
  using namespace fe16b;
  cudaFuncAttributes a; cudaFuncGetAttributes(&a, fe16b_kernel);
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

# host-side classes and the analysis are shared with v1 (same buffers, events and log layout)
from fe16_skel import EV, KIND, U64MAX, MIN_EVENTS, Skel  # noqa: E402,F401
from fe16_skel import analyze as _analyze_v1  # noqa: E402


def build(ns=7, hopper=True, log=print):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", "--expt-relaxed-constexpr", "-Xptxas=-warn-spills", f"-DFE_NS={ns}"]
    if hopper:
        flags.append("-gencode=arch=compute_90a,code=sm_90a")
    tag = hashlib.sha1((CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"fe16_skel2_{tag}", cpp_sources=CPP, cuda_sources=CUDA,
                      functions=["fe16_launch", "fe16_attrs", "fe16_gtres"], extra_cuda_cflags=flags, verbose=False)
    a = list(mod.fe16_attrs())
    log(f"[fe16b] built {tag} in {time.time() - t0:.0f} s: regs {a[0]} local {a[1]} B smem {a[2]} B ctr {a[3]} sched {a[4]} NS {a[5]}")
    return mod, a


def analyze(mk, T, coupled=False):
    """v1 analysis plus per-kind service-time medians (saved this time)."""
    out = _analyze_v1(mk, T, coupled=coupled)
    recs, t0 = mk["recs"], mk["ev"][EV["START"]]
    per = {}
    for r in recs:
        per.setdefault(r["kind"], []).append(r)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None

    out["per_kind"] = {k: {"n": len(v),
                           "land_us": med([(r["t_full"] - r["t_issue"]) / 1e3 for r in v]),
                           "dep_us": med([(r["t_dep"] - r["t_full"]) / 1e3 for r in v]),
                           "service_us": med([(r["t_end"] - r["t_dep"]) / 1e3 for r in v]),
                           "first_full_us": min((r["t_full"] - t0) / 1e3 for r in v),
                           "last_end_us": max((r["t_end"] - t0) / 1e3 for r in v)} for k, v in per.items()}
    return out

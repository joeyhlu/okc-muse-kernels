#!/usr/bin/env python3
"""SL1: the combined small-T intercept design for TINY / TINY2 (moe_fused<false>, bs <= 8) as a REAL variant.

Generates kernels/muse_moe_w4a16/mb_layer_sl1.py from kernels/muse_moe_w4a16/mb_layer.py with exact-once string
edits of the _HYBLO_CUDA text (+ a generated build / arm block appended to the module). The same CUDA text lives in
kernels/lab/k3_diet_rq128.py: `--target <file>` checks (and with --out writes) the SAME CUDA edits there. One compiled
binary per engine serves every arm: Params.sl1 is snapshotted from the host global g_sl1 (sl1_set) at launch, the
harness sets it in place AFTER prepare()'s flag-0 validation and captures one graph per arm (bt4b convention).

Kill risks FIRST (no speed claim; MEASURED bounds bs1 / bs8: AB1 1.21 / 1.86, FIN 1.15 / 1.64, AB2+FIN 2.47 / 3.07 us):
 K1 STAG (R-tstag-1/2) built the same time-major plan and LOST +2.5 us at T=1: the helper's per-job __threadfence
    waited for the SM's in-flight bulk copies (R-dsm-0b/0c, 1.9-3.7 us at 2-3 x 36 KB), the 4-slot reduction ring
    stalled the consumers and the early credit never materialised. SL1 moves every mid-stream GPU fence to a new
    PUBLISHER warp that nothing waits on; if the 10th warp or its polling perturbs the consumer SMSP the gain dies.
 K2 Time-major adds ~K job switches per CTA per phase (K = touched experts, ~3 at bs1, ~9 at bs8). Each switch of
    the ship reloads x / inter rows from L2 (exposed when consumer-paced). SL1 prefetches the next job's rows into
    smem; if cp.async traffic or the prefetch bookkeeping costs more than the hop, bs8 loses.
 K3 The last job's credit still needs helper lag + fence + hop (~2.5-5 us) vs slack (K-1)/K x T_B: ~3.7 us at
    bs1 -> the bs1 A->B gain may be partial. PC keeps the combine work (moved to the completer CTA) and one counter
    round trip; only the barrier, zeroing and spin go away (FIN bound).

Flag bits (Params.sl1): 1 TM, 2 PC, 4 NOPF (TM without the prefetch; diagnostic). Valid masks 0, 1, 2, 3, 5, 7.
 TM  CTA c owns units [alo, alo+an) of EVERY job's 512 W13 units and [blo, blo+bn) of every job's 256 W2 units
     (alo = floor(512 c / G), an = floor(512 (c+1) / G) - alo; same for B with 256), A in job order e0..eK-1, then
     B in the same order. The producer (consumer tid0 / helper lane 0 / FAST1 pre-issue) keeps issuing weights
     across the A->B boundary (data-independent), only consumers wait. Per-job doneA credits come from the
     publisher warp (physical threads NTL..NTL+31; launched only when TM: blockDim 320, __launch_bounds__(NTL+32)).
     Each consumer warp prefetches its K-slice of the NEXT job's x (A, skipped at M == 1 where every A job has the
     same row) / inter (B) rows into a private 8 KB smem slot (cp.async.cg) inside the skinny-unused FT tail of the
     dynamic smem; B targets only after their credit was observed (relaxed load + fence.acquire.gpu by lane 0, then
     __syncwarp, = the ship's lane-0 acquire + __syncwarp pattern).
 PC  per-token pending-unit counters; the CTA whose count completes a token's 256 x L_t W2 units performs norm +
     weighted combine + store of that token (the ship's expressions, same slot order s = 0..7); no end-of-kernel
     grid barrier, no zeroing pass, no token loop over all CTAs; generation parity double-buffered state.

PROTOCOL INVARIANTS (G = gridDim.x = #SMs, 1 CTA/SM, all CTAs co-resident as in the ship):
 P1 Ownership. muldiv_g is an exact floor, so per phase and job sum_c share_c = 512 (A) / 256 (B) with disjoint,
    covering ranges. TM maps CTA-local ordinal u -> v = (u div n) R + lo + (u mod n), a bijection [0, njS n) ->
    {units [lo, lo+n) of every job}. The FAST1 pre-issue (warp 7 lane 0, before route), ring_issue (consumer tid0
    and helper lane 0), the consumer loop, the helper loop and the WSBLK tail all apply the SAME map with the SAME
    muldiv_g(R, c, G, __fdividef(1, G)) inputs; S.wsr then holds CTA-local ranges [0, njS an) / [0, njS bn), so
    every ship expression on S.wsr (nA, N, Nh, ws_issue, run_phase v0/v1) is unchanged.
 P2 A credits (TM). The helper epilogues A units in order; after the unit that ends job k's share it __syncwarp()s
    and lane 0 st.release.cta's L.hp = q + 1 (smem; B job ends too when PC). The publisher (all 32 lanes, one
    warp-wide ld.acquire.cta) derives the jobs whose share is complete, runs ONE __threadfence() per batch
    (cumulative GPU release of the helper stores it synchronized with) and lane-parallel atomicAdd(doneA[bank] + k,
    an). Sum over CTAs = 512, the consumers' unchanged poll target. Without TM the ship's end-of-share credit
    (helper, one fence) stays, only bank-indexed under PC.
 P3 B data. A consumer warp reads inter rows of job k only after its lane 0 observed doneA[bank + k] >= 512 with
    acquire semantics (ship ld.acquire spin, or ld.relaxed + fence.acquire.gpu before the prefetch) followed by
    __syncwarp(). No address, loop trip count or synchronization count depends on y / inter / ss values.
 P4 Token counts (PC). Every W2 unit (one job, all its tokens) has exactly one owner (P1); each owner adds, per
    (job share, token of the job), its number of units of that job to tok[bank][token] exactly once: TM+PC = the
    publisher for jobs whose units are all helper-epilogued (signalled through L.hp, P2 chain) and warp 0 after
    the final WS_JOIN for the (<= 2) jobs holding WSBLK tail units; PC alone = warp 0 after the final WS_JOIN for
    the (<= 2) jobs of the CTA's contiguous B range. Every count follows a __threadfence() by the counting thread
    after it synchronized (WS_JOIN, or L.hp acquire.cta) with all y / ss writers of those units. So the counter
    receives exactly 256 x L_t (L_t = local pairs of the token) in totally ordered RMWs: EXACTLY ONE RMW sees
    old + n == need -> no CTA can miss a token's last contribution and no token is combined twice. The completer
    __threadfence()s (acquire) before its smem list is read; its consumer threads load ss / ybuf (__ldcg) only
    after a CTA barrier (bar 1, or bar 4 shared with the publisher). Tokens with L_t == 0 get zeros from CTA
    (t mod G) right after route (the ship writes zeros for them too).
 P5 Parity banks (PC). bank = gen & 1 for doneA[bank*JMAX ..], tok[bank*MMAX ..] (scratch[0] ints 8+176 .. 8+944,
    unused by the ship) and the ship's ss / ss1. The current bank was zeroed by the PREVIOUS generation at its start
    (as its "other" bank: all CTAs, i = c*NT + tid, stride G*NT; ss exactly as the ship) and nobody else writes it;
    that kernel completed before this one began (stream order; under PDL every global access follows
    griddepcontrol.wait). The other bank is zeroed now and first used by the NEXT generation. First generation:
    torch.zeros scratch. A scratch must not alternate flag-0 and PC calls (the harness owns one scratch per arm).
 P6 Generation / exit (PC). tid0 reads gen = ctr[6] / G (volatile) at kernel start; tid0 red.adds ctr[6] += 1 as
    its CTA's last action. A CTA increments only after its own read, so at most G-1 increments of generation g
    precede any read of generation g: every CTA of one launch computes the same g, and after all G exits
    ctr[6] = (g+1) G. Nobody waits on ctr[6], so an exit cannot strand a waiter. The only cross-CTA waits are the
    doneA polls (P3); credits depend on A work only (ring + own helper + own publisher), never on B, on a token
    count or on an exit. The publisher waits only on its own helper (L.hp); the consumers wait for the publisher
    (bar 4) only after the helper finished (final WS_JOIN). No cycle -> hang-free. PC never touches ctr[0..5]
    after the ship's start-of-kernel reset of ctr[4+other] (harmless); ctr[0..2] stay 0.
 P7 TM alone keeps the ship's end (grid barrier on ctr[4+par], ctr[3] generation, post-barrier doneA bank-0
    zeroing, token loop). Every B unit waited for 512 credits, so all publisher atomics were performed before any
    CTA passed the barrier: the zeroing cannot race a credit.
 P8 Prefetch slots. Per warp 8 KB, chunk-major (chunk c of lane L at warp*8192 + c*512 + L*16, conflict-free
    LDS.128): each lane writes (cp.async.cg, its own 16 x 16 B) and reads (after its own cp.async.wait_all) only its
    own bytes. A new prefetch into the slot is issued only after the lane's LDS of the slot completed (data
    dependency on x0 via sl1_dep, or after the unit's MMAs consumed x0). Every issued prefetch is the next job of the
    warp's own sequence, so it is consumed at that switch; the B phase ends with a defensive wait_all. The A->B
    carry (staged job, flag look-ahead) goes through L.pfs[warp] (lane 0 writes, __syncwarp, all lanes read).
 P9 Smem / scratch. Slots (64 KB) + SL1 state (< 2 KB) start at xbuf + NS*UNIT_BYTES, inside okc_smem_bytes()'s FT
    tail that the skinny path never touches (static_assert); the ship's Smem struct and smem size are unchanged.
    static_assert(2*JMAX + 2*MMAX <= 1024): the banks end before ss1.

Hot-path accounting (EST source operations; the gate JSON has the exact SASS counts):
 Weight stages 36,864 B before = after; MMA / dequant / reduction instructions unchanged. Flag 0 adds uniform
 branches on p.sl1 (constant bank) per consumer unit (map, pending-prefetch test), per helper unit (map, signal
 test) and per ring issue (map). TM per consumer unit: map = 2 LDS (L.an/L.alo or bn/blo) + ~6 integer ops;
 per job switch: 16 LDS.128 from the slot (instead of 16 LDG.128 from L2), 16 LOP for the slot dependency,
 32 cp.async.cg x 16 B issued for the next job (only rows g < cnt: 4 of 32 lanes at cnt 1), 1 relaxed LDG flag
 look-ahead (lane 0) + fence.acquire (CCTL only, lane 0). Bytes from L2 per CTA unchanged per switch (same rows),
 K-1 extra switches per phase vs the ship. Helper per job end: 1 st.release.cta (smem). Publisher per batch:
 1 __threadfence + <= 32 lane-parallel atomics. PC per CTA end: 1 fence + <= 16 lane-parallel token RMWs; per
 completed token 4096 x L_t fp32 y + L_t ss reads (as the ship's combine) and 8 KB bf16 output.
"""
import argparse
import ast
import difflib
import hashlib
import json
import pprint
import random
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
DEST = ROOT / 'kernels/muse_moe_w4a16/mb_layer_sl1.py'
LAYER = ROOT / 'kernels/lab/k3_diet_rq128.py'
MASKS = (0, 1, 2, 3, 5, 7)
CUDA_NAME = '_HYBLO_CUDA'


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:120])
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------------------------------------------
# New CUDA (device helpers, publisher warp, PC start / finish). Inserted right after smem_u32().
DEVICE = r'''// ==== SL1 (GENERATED by tools/sl1_patches.py; protocol invariants P1-P9 in its header) ====
constexpr int SL1_TM = 1, SL1_PC = 2, SL1_NOPF = 4, SL1_CMBMAX = 64;
constexpr int SL1_PF_WARP = 2 * NCH * 512;
struct SL1S {
int alo, an, blo, bn;
int hp;
int ncmb;
int pad0, pad1;
int cmb[SL1_CMBMAX];
int pfs[NW][4];
};
constexpr unsigned SL1_XOFF = (unsigned)(((sizeof(Smem) + 127) / 128) * 128);
constexpr unsigned SL1_PFOFF = SL1_XOFF + (unsigned)OKC_NS * (unsigned)UNIT_BYTES;
constexpr unsigned SL1_LOFF = SL1_PFOFF + (unsigned)(NW * SL1_PF_WARP);
static_assert(OKC_NS <= OKC_NSF, "SL1: the skinny ring lies inside the FT region");
static_assert((size_t)OKC_NS * UNIT_BYTES + (size_t)NW * SL1_PF_WARP + sizeof(SL1S) <=
(size_t)OKC_NSF * UNIT_BYTES + 2 * (size_t)FT_T * XP_FT * 2, "SL1: slots + state fit the skinny-unused FT tail");
static_assert(SL1_PFOFF % 128 == 0 && SL1_LOFF % 16 == 0, "SL1: alignment");
static_assert(2 * JMAX + 2 * MMAX <= 1024, "SL1: doneA + token banks end before ss1");
__device__ __forceinline__ SL1S& sl1_L() { return *reinterpret_cast<SL1S*>(smem_raw + SL1_LOFF); }
__device__ __forceinline__ int sl1_div(int u, int n) {
return n == 1 ? u : n == 2 ? (u >> 1) : n == 3 ? (int)(((unsigned)u * 43691u) >> 17) : n == 4 ? (u >> 2) : u / n;
}
__device__ __forceinline__ int sl1_mapu(int u, int lo, int n, int rbs) {
const int k = sl1_div(u, n);
return k * rbs + lo + (u - k * n);
}
template <bool PB>
__device__ __forceinline__ int sl1_map(const Params& p, int u) {
if (!(p.sl1 & SL1_TM)) return u;
const SL1S& L = sl1_L();
return PB ? sl1_mapu(u, L.blo, L.bn, RB_B) : sl1_mapu(u, L.alo, L.an, RB_A);
}
__device__ __forceinline__ int sl1_bank(const Params& p, const Smem& S) { return (p.sl1 & SL1_PC) ? (S.gen & 1) * JMAX : 0; }
__device__ __forceinline__ int* sl1_tok(const Params& p) { return p.doneA + 2 * JMAX; }
__device__ __forceinline__ int sl1_ldr(const int* a) {
int v;
asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(a));
return v;
}
__device__ __forceinline__ void sl1_facq() { asm volatile("fence.acquire.gpu;\n" ::: "memory"); }
__device__ __forceinline__ void sl1_st_rel(int* a, int v) {
asm volatile("st.release.cta.shared.b32 [%0], %1;\n" ::"r"(smem_u32(a)), "r"(v) : "memory");
}
__device__ __forceinline__ int sl1_ld_acq(const int* a) {
int v;
asm volatile("ld.acquire.cta.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(smem_u32(a)) : "memory");
return v;
}
__device__ __forceinline__ int sl1_need(const Smem& S, int t) {
int c = 0;
#pragma unroll
for (int s = 0; s < 8; ++s) c += S.pos[t * 8 + s] >= 0 ? 1 : 0;
return c * RB_B;
}
template <bool PB>
__device__ __forceinline__ int sl1_next(const Params& p, const Smem& S, int j) {
if (!PB) return j + 1 < S.njS ? (p.M == 1 ? -1 : j + 1) : (sl1_L().bn > 0 ? 4096 : -1);
return j + 1 < S.njS ? 4096 + j + 1 : -1;
}
__device__ __forceinline__ int sl1_flagjob(const Smem& S, int t) {
if (t < 0) return -1;
if (t < 4096) return t + 1 < S.njS ? -1 : (sl1_L().bn > 0 ? 4096 : -1);
return (t & 4095) + 1 < S.njS ? t + 1 : -1;
}
__device__ __forceinline__ uint32_t sl1_slot(int warp, int lane) {
return smem_u32(smem_raw + SL1_PFOFF) + (uint32_t)(warp * SL1_PF_WARP + lane * 16);
}
__device__ __forceinline__ uint32_t sl1_dep(const uint4 (&x)[2 * NCH]) {
uint32_t d = 0u;
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) d ^= x[c].x;
asm volatile("and.b32 %0, %0, 0;\n" : "+r"(d));
return d;
}
__device__ __forceinline__ void sl1_issue(const Params& p, const Smem& S, int t, int lane, int warp, uint32_t dep) {
const int jt = t & 4095, cnt = S.jC[jt], st = S.jS[jt], g = lane >> 2, tt = lane & 3;
if (g < cnt) {
const __nv_bfloat16* row = t >= 4096 ? p.inter + (size_t)(st + g) * ID : p.x + (size_t)(S.ent[st + g] >> 3) * HD;
const uint4* src = reinterpret_cast<const uint4*>(row + warp * (64 * NCH) + 16 * tt);
const uint32_t dst = sl1_slot(warp, lane) + dep;
#pragma unroll
for (int c = 0; c < NCH; ++c) {
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst + (uint32_t)(2 * c) * 512u), "l"(src + c * 8) : "memory");
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst + (uint32_t)(2 * c + 1) * 512u), "l"(src + c * 8 + 1) : "memory");
}
}
asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void sl1_take(const Smem& S, int j, int lane, int warp, uint4 (&x)[2 * NCH]) {
asm volatile("cp.async.wait_all;\n" ::: "memory");
if ((lane >> 2) < S.jC[j]) {
const uint4* a = reinterpret_cast<const uint4*>(smem_raw + SL1_PFOFF + warp * SL1_PF_WARP + lane * 16);
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) x[c] = a[c * 32];
} else {
#pragma unroll
for (int c = 0; c < 2 * NCH; ++c) x[c] = make_uint4(0, 0, 0, 0);
}
}
__device__ void sl1_plan(const Params& p, Smem& S, int c, int G, float rG) {
if (!p.sl1) return;
SL1S& L = sl1_L();
L.hp = 0;
L.ncmb = 0;
for (int w = 0; w < NW; ++w) { L.pfs[w][0] = -1; L.pfs[w][1] = -1; L.pfs[w][2] = -1; L.pfs[w][3] = 0; }
if (p.sl1 & SL1_TM) {
const int alo = muldiv_g(RB_A, c, G, rG), blo = muldiv_g(RB_B, c, G, rG);
L.alo = alo; L.an = muldiv_g(RB_A, c + 1, G, rG) - alo;
L.blo = blo; L.bn = muldiv_g(RB_B, c + 1, G, rG) - blo;
S.wsr[0] = 0; S.wsr[1] = S.njS * L.an; S.wsr[2] = 0; S.wsr[3] = S.njS * L.bn;
}
}
__device__ void sl1_pc_start(const Params& p, const Smem& S, int tid) {
const int ob = (S.gen + 1) & 1;
int* tok = sl1_tok(p);
for (int i = blockIdx.x * NT + tid; i < JMAX + MMAX; i += gridDim.x * NT) {
if (i < JMAX) p.doneA[ob * JMAX + i] = 0;
else tok[ob * MMAX + (i - JMAX)] = 0;
}
for (int t = blockIdx.x; t < p.M; t += gridDim.x) {
if (sl1_need(S, t) == 0) {
uint4* o = reinterpret_cast<uint4*>(p.out + (size_t)t * HD);
for (int i = tid; i < HD / 8; i += NT) o[i] = make_uint4(0, 0, 0, 0);
}
}
}
__device__ void sl1_publisher(const Params& p, Smem& S) {
asm volatile("griddepcontrol.wait;\n" ::: "memory");
if (!(p.sl1 & SL1_TM)) return;
asm volatile("bar.sync 3, %0;\n" ::"n"(NTL + 32) : "memory");
const int lane = threadIdx.x & 31;
SL1S& L = sl1_L();
const bool pc = (p.sl1 & SL1_PC) != 0;
const int njS = S.njS, an = L.an, bn = L.bn, nA = njS * an, nB = njS * bn;
const int kbh = (pc && bn > 0) ? (nB - min(OKC_WSBLK, nB)) / bn : 0;
const int aEnd = an > 0 ? njS : 0;
int* const dA = p.doneA + sl1_bank(p, S);
int* const tok = sl1_tok(p) + (S.gen & 1) * MMAX;
int ad = 0, bd = 0;
bool comp = false;
while (ad < aEnd || bd < kbh) {
const int h = sl1_ld_acq(&L.hp);
const int ar = an > 0 ? min(aEnd, h / an) : 0;
const int br = (bn > 0 && h > nA) ? min(kbh, (h - nA) / bn) : 0;
if (ar <= ad && br <= bd) { __nanosleep(256); continue; }
__threadfence();
for (int k = ad + lane; k < ar; k += 32) atomicAdd(dA + k, an);
for (int it = lane; it < (br - bd) * 8; it += 32) {
const int jj = bd + (it >> 3), tk = it & 7;
if (tk < S.jC[jj]) {
const int t = S.ent[S.jS[jj] + tk] >> 3;
if (atomicAdd(tok + t, bn) + bn == sl1_need(S, t)) { L.cmb[atomicAdd(&L.ncmb, 1)] = t; comp = true; }
}
}
if (ar > ad) ad = ar;
if (br > bd) bd = br;
}
if (pc) {
if (__any_sync(0xffffffffu, comp)) __threadfence();
asm volatile("bar.arrive 4, %0;\n" ::"n"(NT + 32) : "memory");
}
}
__device__ void sl1_finish(const Params& p, Smem& S, int tid) {
SL1S& L = sl1_L();
const int lane = tid & 31;
const bool tm = (p.sl1 & SL1_TM) != 0;
if (tid < 32) {
int* const tok = sl1_tok(p) + (S.gen & 1) * MMAX;
const int b0 = S.wsr[2], b1 = S.wsr[3];
int k0 = 0, k1 = 0;
if (tm) {
const int nB = S.njS * L.bn;
if (L.bn > 0) { k0 = (nB - min(OKC_WSBLK, nB)) / L.bn; k1 = S.njS; }
} else if (b1 > b0) {
k0 = b0 / RB_B; k1 = (b1 - 1) / RB_B + 1;
}
__threadfence();
bool comp = false;
for (int it = lane; it < (k1 - k0) * 8; it += 32) {
const int jj = k0 + (it >> 3), tk = it & 7;
if (tk < S.jC[jj]) {
const int n = tm ? L.bn : min(b1, (jj + 1) * RB_B) - max(b0, jj * RB_B);
const int t = S.ent[S.jS[jj] + tk] >> 3;
if (atomicAdd(tok + t, n) + n == sl1_need(S, t)) { L.cmb[atomicAdd(&L.ncmb, 1)] = t; comp = true; }
}
}
if (__any_sync(0xffffffffu, comp)) __threadfence();
}
if (tm) asm volatile("bar.sync 4, %0;\n" ::"n"(NT + 32) : "memory");
else CTA_SYNC();
constexpr int SEG = NT * 4;
const int nc = L.ncmb;
for (int ci = 0; ci < nc; ++ci) {
const int t = L.cmb[ci];
int qv[8];
float wv[8], ssv[8];
#pragma unroll
for (int s = 0; s < 8; ++s) {
qv[s] = S.pos[t * 8 + s];
wv[s] = qv[s] >= 0 ? __ldg(p.topk_w + t * 8 + s) : 0.f;
ssv[s] = qv[s] >= 0 ? __ldcg(ss_cur() + qv[s]) : 0.f;
}
float4 acc[HD / SEG];
#pragma unroll
for (int f = 0; f < HD / SEG; ++f) acc[f] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int s = 0; s < 8; ++s) {
if (qv[s] >= 0) {
const float r = wv[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);
#pragma unroll
for (int f = 0; f < HD / SEG; ++f) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[s] * HD + f * SEG + tid * 4));
acc[f].x += r * v.x; acc[f].y += r * v.y; acc[f].z += r * v.z; acc[f].w += r * v.w;
}
}
}
#pragma unroll
for (int f = 0; f < HD / SEG; ++f) {
__nv_bfloat162 o0 = __floats2bfloat162_rn(acc[f].x, acc[f].y), o1 = __floats2bfloat162_rn(acc[f].z, acc[f].w);
uint2 o;
o.x = *reinterpret_cast<uint32_t*>(&o0);
o.y = *reinterpret_cast<uint32_t*>(&o1);
*reinterpret_cast<uint2*>(p.out + (size_t)t * HD + f * SEG + tid * 4) = o;
}
}
if (tid == 0) atomicAdd(p.ctr + 6, 1);
}
// ==== SL1 end of device helpers ====
'''

# Consumer job switch (skinny_ring_ws): ship text -> SL1 text.
SWITCH_OLD = r'''if (j != curj) {
curj = j;
if (PB) {
if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }
__syncwarp();
}
load_x_regs<PB>(p, S, j, lane, warp, x0);
}'''
SWITCH_NEW = r'''if (j != curj) {
const bool sl1keep = !PB && (p.sl1 & SL1_TM) && p.M == 1 && curj >= 0;
curj = j;
if (sl1pf && sl1pj == (PB ? 4096 : 0) + j) {
sl1_take(S, j, lane, warp, x0);
} else if (!sl1keep) {
if (PB) {
if (lane == 0) { while (ld_acquire(p.doneA + sl1_bank(p, S) + j) < RB_A) __nanosleep(64); }
__syncwarp();
}
load_x_regs<PB>(p, S, j, lane, warp, x0);
}
sl1pj = -1;
if (sl1pf) {
sl1pt = sl1_next<PB>(p, S, j);
if (sl1pt >= 0) {
const bool sl1ok = sl1pt < 4096 || (sl1fn == sl1pt && __shfl_sync(0xffffffffu, sl1fl, 0) >= RB_A);
if (sl1ok) {
if (sl1pt >= 4096) { if (lane == 0) sl1_facq(); __syncwarp(); }
sl1_issue(p, S, sl1pt, lane, warp, sl1_dep(x0));
sl1pj = sl1pt;
sl1pt = -1;
sl1fn = sl1_flagjob(S, sl1pj);
} else {
sl1fn = sl1pt;
}
if (sl1fn >= 0 && lane == 0) sl1fl = sl1_ldr(p.doneA + sl1_bank(p, S) + (sl1fn & 4095));
}
}
}'''
PENDING = r'''if (sl1pt >= 0) {
if (__shfl_sync(0xffffffffu, sl1fl, 0) >= RB_A) {
if (lane == 0) sl1_facq();
__syncwarp();
sl1_issue(p, S, sl1pt, lane, warp, 0u);
sl1pj = sl1pt;
sl1pt = -1;
sl1fn = sl1_flagjob(S, sl1pj);
if (sl1fn >= 0 && lane == 0) sl1fl = sl1_ldr(p.doneA + sl1_bank(p, S) + (sl1fn & 4095));
} else if (lane == 0) {
sl1fl = sl1_ldr(p.doneA + sl1_bank(p, S) + (sl1pt & 4095));
}
}
'''
LOOP_STATE = r'''const bool sl1pf = (p.sl1 & (SL1_TM | SL1_NOPF)) == SL1_TM;
int sl1pj = -1, sl1pt = -1, sl1fn = -1, sl1fl = 0;
if (sl1pf) { const SL1S& L = sl1_L(); sl1pj = L.pfs[warp][0]; sl1fn = L.pfs[warp][2]; sl1fl = L.pfs[warp][3]; }
'''
LOOP_END = r'''if (sl1pf) {
if (PB) {
asm volatile("cp.async.wait_all;\n" ::: "memory");
} else {
if (lane == 0) { SL1S& L = sl1_L(); L.pfs[warp][0] = sl1pj; L.pfs[warp][1] = sl1pt; L.pfs[warp][2] = sl1fn; L.pfs[warp][3] = sl1fl; }
__syncwarp();
}
}
'''
SIGNAL = r'''if ((p.sl1 & SL1_TM) && (!PB || (p.sl1 & SL1_PC))) {
const int nn = PB ? sl1_L().bn : sl1_L().an, u1 = PB ? q - nA + 1 : q + 1;
if (u1 == sl1_div(u1, nn) * nn) { __syncwarp(); if (lane == 0) sl1_st_rel(&sl1_L().hp, q + 1); }
}
'''
ARRIVE = r'''if (p.sl1 & SL1_TM) asm volatile("bar.arrive 3, %0;\n" ::"n"(NTL + 32) : "memory");
'''
HOST_GLOBALS = 'static int g_sl1 = 0;\nvoid sl1_set(int64_t v) { g_sl1 = (int)v; }\n'
HOST_CHECKS = r'''TORCH_CHECK(!(g_sl1 & ~7) && (!(g_sl1 & 4) || (g_sl1 & 1)), "sl1: invalid flag word ", g_sl1);
TORCH_CHECK(!g_sl1 || !g_okc_pdl, "sl1 lab variant requires ordered replay (okc_set_pdl(0))");
TORCH_CHECK(!(g_sl1 & 2) || p.M <= okc::SL1_CMBMAX, "sl1 PC: M <= 64");
const int sl1_ntl = okc::NTL + ((g_sl1 & 1) ? 32 : 0);
'''


def cuda_edits(src):
    """(bit, old, new) applied in order; every old is exact-once in the progressively edited text."""
    edits = []

    def edit(bit, old, new):
        nonlocal src
        src = replace_once(src, old, new)
        edits.append((bit, old, new))

    # ABI: flag word first in Params (constant-bank offset 0), host snapshot, checks, launch width.
    edit('ABI', 'struct Params {\n', 'struct Params {\nint sl1; // SL1 flag word (tools/sl1_patches.py): 1 TM, 2 PC, 4 TM-no-prefetch\n')
    sm_u32 = '__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }\n'
    edit('ABI', sm_u32, sm_u32 + DEVICE)
    edit('ABI', 'static int g_okc_pdl = 0;\n', HOST_GLOBALS + 'static int g_okc_pdl = 0;\n')
    edit('ABI', 'p.zero = 0;\n', 'p.zero = 0;\np.sl1 = g_sl1;\n')
    edit('ABI', 'cudaStream_t st = at::cuda::getCurrentCUDAStream();\n', HOST_CHECKS + 'cudaStream_t st = at::cuda::getCurrentCUDAStream();\n')
    edit('TM', 'cfg.blockDim = dim3(okc::NTL);', 'cfg.blockDim = dim3(sl1_ntl);')
    edit('TM', 'okc::moe_fused<true><<<(int)grid, okc::NTL, sm, st>>>(p);', 'okc::moe_fused<true><<<(int)grid, sl1_ntl, sm, st>>>(p);')
    edit('TM', 'okc::moe_fused<false><<<(int)grid, okc::NTL, sm, st>>>(p);', 'okc::moe_fused<false><<<(int)grid, sl1_ntl, sm, st>>>(p);')
    edit('TM', '__launch_bounds__(NTL, 1) moe_fused(Params p) {', '__launch_bounds__(NTL + 32, 1) moe_fused(Params p) {')
    # TM: publisher warp branch (physical threads NTL..NTL+31 exist only when the host launched TM).
    edit('TM', 'Smem& S = *reinterpret_cast<Smem*>(smem_raw);\n',
         'Smem& S = *reinterpret_cast<Smem*>(smem_raw);\nif (threadIdx.x >= NTL) { sl1_publisher(p, S); return; }\n')
    # TM: one unit map for every producer / consumer of a unit index.
    edit('TM', 'unsigned char* slot, uint64_t* bar) {\nconst int rbs = PB ? RB_B : RB_A;',
         'unsigned char* slot, uint64_t* bar) {\nv = sl1_map<PB>(p, v);\nconst int rbs = PB ? RB_B : RB_A;')
    edit('TM', 'uint32_t lnw0 = 0u, lnw1 = 0u;\nfor (int i = 0; i < nU; ++i) {\nconst int v = v0 + i;\n',
         'uint32_t lnw0 = 0u, lnw1 = 0u;\n' + LOOP_STATE + 'for (int i = 0; i < nU; ++i) {\nconst int v = sl1_map<PB>(p, v0 + i);\n')
    edit('PF', SWITCH_OLD, SWITCH_NEW)
    edit('PF', 'for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\nconst int bb = q % NRB;',
         'for (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n' + PENDING + 'const int bb = q % NRB;')
    edit('TM', 'const int v = v0 + ul, j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;',
         'const int v = sl1_map<true>(p, v0 + ul), j = njF + v / RB_B, rb = v - (v / RB_B) * RB_B;')
    edit('PF', 'seq = seq0 + nU;\n', LOOP_END + 'seq = seq0 + nU;\n')
    edit('TM', 'const int lane = TIDX & 31;\nWS_JOIN();\n', 'const int lane = TIDX & 31;\nWS_JOIN();\n' + ARRIVE)
    edit('TM', 'const int v = PB ? b0 + (q - nA) : a0 + q;',
         'const int v = PB ? sl1_map<true>(p, b0 + (q - nA)) : sl1_map<false>(p, a0 + q);')
    edit('TM', 'if (q == nA - 1) {\n__threadfence();', 'if (!(p.sl1 & SL1_TM) && q == nA - 1) {\n__threadfence();')
    edit('PC', 'atomicAdd(p.doneA + jj, b - a);', 'atomicAdd(p.doneA + sl1_bank(p, S) + jj, b - a);')
    edit('TM', 'atomicAdd(&ss_cur()[st + tk], sq);\n}\n}\n}\n__threadfence();\nWS_JOIN();',
         'atomicAdd(&ss_cur()[st + tk], sq);\n}\n}\n' + SIGNAL + '}\n__threadfence();\nWS_JOIN();')
    edit('TM', 'skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);',
         'skinny_split<BIG, PB>(nSs, nF, G, c, v0, v1);\nif (p.sl1 & SL1_TM) { v0 = S.wsr[PB ? 2 : 0]; v1 = S.wsr[PB ? 3 : 1]; }')
    # TM: FAST1 pre-issue = the CTA's first time-major units (job 0's share).
    edit('TM', 'const int n = min(OKC_FAST1, v1 - v0);',
         'const int sl1a0 = muldiv_g(RB_A, blockIdx.x, gridDim.x, rG), sl1an = muldiv_g(RB_A, blockIdx.x + 1, gridDim.x, rG) - sl1a0;\n'
         'const int n = (p.sl1 & SL1_TM) ? min(OKC_FAST1, __popc(m) * sl1an) : min(OKC_FAST1, v1 - v0);')
    edit('TM', 'for (int i = 0; i < n; ++i) {\nconst int v = v0 + i;\nconst int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);',
         'for (int i = 0; i < n; ++i) {\nconst int v = (p.sl1 & SL1_TM) ? sl1_mapu(i, sl1a0, sl1an, RB_A) : v0 + i;\n'
         'const int e = (int)__fns(m, 0, (v >> RB_A_LOG) + 1);')
    # PC: generation from the monotone exit counter, other-bank reset + empty tokens, end path.
    edit('PC', 'const int gen = *reinterpret_cast<volatile const int*>(p.ctr + 3);',
         'const int gen = (p.sl1 & SL1_PC) ? *reinterpret_cast<volatile const int*>(p.ctr + 6) / (int)gridDim.x\n'
         ': *reinterpret_cast<volatile const int*>(p.ctr + 3);')
    edit('PC', 'if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n}',
         'if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n}\nif (p.sl1 & SL1_PC) sl1_pc_start(p, S, tid);')
    edit('TM', 'S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\n}',
         'S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);\nsl1_plan(p, S, c, G, rG);\n}')
    edit('TM', 'WS_JOIN();\nrun_phase<false, BIG>(p, S, xbuf);', 'WS_JOIN();\n' + ARRIVE + 'run_phase<false, BIG>(p, S, xbuf);')
    edit('PC', 'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\nconstexpr int FSEGP',
         'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");\nif (p.sl1 & SL1_PC) { sl1_finish(p, S, tid); return; }\nconstexpr int FSEGP')
    return edits


def apply_edits(src, edits):
    for bit, old, new in edits:
        src = replace_once(src, old, new)
    return src


def literal(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value), n.lineno
    raise KeyError(name)


EXTENSION = r'''

# SL1_BEGIN -- GENERATED by tools/sl1_patches.py (do not edit; protocol P1-P9 in its header).
_SL1_EDITS = @EDITS@
SL1_MASKS = (0, 1, 2, 3, 5, 7)
_SL1_SHIP = {}  # the UNPATCHED ship tiny / tiny2 modules (control arm "ship": flag-0 cost of the SL1 binary)
_sl1_original_builds = _mbl_builds

def sl1_sources(engine, instrument=True):
 if engine not in ("tiny", "tiny2"):
  raise ValueError(engine)
 src, cpp = _HYBLO_CUDA, _HYB_MOE_CPP
 flags = list(_HYBLO_FLAGS) + list(next(v for k, mx, v in _HYBLO_KEYS if k == engine))
 if instrument:
  for bit, old, new in _SL1_EDITS:
   assert src.count(old) == 1, (bit, old[:80])
   src = src.replace(old, new, 1)
  cpp += "\nvoid sl1_set(int64_t v);\n"
 return src, cpp, flags

def _mbl_builds(deadline):
 # Ship builds first (unchanged), then the two SL1 engines in parallel; they REPLACE tiny / tiny2, so
 # prepare()'s flag-0 validation runs on the SL1 binaries. A failed SL1 build fails prepare (no fallback).
 mods = _sl1_original_builds(deadline)
 built, errs = {}, {}
 def one(engine):
  try:
   src, cpp, flags = sl1_sources(engine)
   tag = hashlib.sha1((cpp + src + " ".join(flags)).encode()).hexdigest()[:12]
   built[engine] = _mbl_load_inline(name="sl1_" + engine + "_" + tag, cpp_sources=cpp, cuda_sources=src,
                                    functions=["okc_run", "okc_set_pdl", "sl1_set"], extra_cuda_cflags=flags, verbose=False)
  except Exception as ex:
   errs[engine] = repr(ex)[:600]
 jobs = [threading.Thread(target=one, args=(e,), name="sl1-" + e, daemon=True) for e in ("tiny", "tiny2")]
 for job in jobs:
  job.start()
 for job in jobs:
  job.join(max(0.0, deadline - time.monotonic()))
  _mbl_deadline(deadline, "SL1 build")
 if errs or sorted(built) != ["tiny", "tiny2"]:
  raise RuntimeError("SL1 build failed: " + repr(errs))
 _SL1_SHIP.update(tiny=mods["tiny"], tiny2=mods["tiny2"])
 for engine in ("tiny", "tiny2"):
  built[engine].sl1_set(0)
  built[engine].okc_set_pdl(int(HYBLO_PDL))
  mods[engine] = built[engine]
 return mods

def _sl1_init(state, dev):
 state["sl1_flag"] = torch.zeros(1, dtype=torch.int32, device=dev)
 state["sl1_arms"] = {}

def sl1_arm(state, mask, m):
 # One mutable device word, read back outside capture; each launch snapshots the verified value as
 # Params.sl1 (graphs keep the value of their capture). Every (mask, M) arm owns a fresh zeroed scratch.
 if mask not in SL1_MASKS or m not in range(1, 9):
  raise ValueError((mask, m))
 state["sl1_flag"].fill_(mask)
 actual = int(state["sl1_flag"].item())
 assert actual == mask
 print(f"[sl1] flag readback={actual} M={m}", flush=True)
 for name in ("tiny", "tiny2"):
  state["mods"][name].okc_set_pdl(0)
  state["mods"][name].sl1_set(actual)
 key = (mask, m)
 if key not in state["sl1_arms"]:
  arm = dict(state)
  arm["h_sc"], arm["h_full"] = _hyb_scratch(state["out"][m].device)
  state["sl1_arms"][key] = arm
 return state["sl1_arms"][key]

def sl1_ship_arm(state, m):
 # Control: the unpatched ship binaries (no Params.sl1 at all), own scratch, PDL off. Flag-0 semantics.
 if m not in range(1, 9) or any(_SL1_SHIP.get(k) is None for k in ("tiny", "tiny2")):
  raise ValueError(("ship arm", m))
 for name in ("tiny", "tiny2"):
  _SL1_SHIP[name].okc_set_pdl(0)
 key = ("ship", m)
 if key not in state["sl1_arms"]:
  arm = dict(state)
  arm["mods"] = dict(state["mods"], tiny=_SL1_SHIP["tiny"], tiny2=_SL1_SHIP["tiny2"])
  arm["h_sc"], arm["h_full"] = _hyb_scratch(state["out"][m].device)
  state["sl1_arms"][key] = arm
 return state["sl1_arms"][key]
# SL1_END
'''


def generate(base):
    cu, _ = literal(base, CUDA_NAME)
    edits = cuda_edits(cu)
    text = replace_once(base, ' _mbl_validate(state,weights)\n', ' _sl1_init(state,dev)\n _mbl_validate(state,weights)\n')
    text = text.rstrip() + '\n' + EXTENSION.replace('@EDITS@', pprint.pformat(edits, width=110, sort_dicts=False))
    ast.parse(text)
    return text


def extension_env(source):
    """Execute ONLY the generated block (with literal globals), no torch."""
    env = {}
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign):
            try:
                v = ast.literal_eval(n.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for t in n.targets:
                if isinstance(t, ast.Name):
                    env[t.id] = v
    env['_mbl_builds'] = lambda deadline: {}
    exec(compile(source[source.index('# SL1_BEGIN'):], '<SL1 block>', 'exec'), env)
    return env


# ---------------------------------------------------------------------------------------------------------------
# CPU models of every new device-side plan / protocol path (mirrors of the CUDA expressions; string-tied below).
def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def muldiv_g(a, c, G, rG):
    """CUDA muldiv_g: q = rz(float(a*c) * rG) (fp32), then +-1 fix-ups -> exact floor(a*c/G)."""
    p = a * c
    q = int(f32(f32(float(p)) * rG))
    if q * G > p:
        q -= 1
    elif (q + 1) * G <= p:
        q += 1
    return q


def rcp(G):
    return f32(1.0 / f32(float(G)))  # __fdividef(1, G): exact enough for the +-1 fix-ups (checked vs floor)


def sl1_div(u, n):
    return u if n == 1 else (u >> 1) if n == 2 else ((u * 43691) & 0xffffffff) >> 17 if n == 3 else (u >> 2) if n == 4 else u // n


def mapu(u, lo, n, rbs):
    k = sl1_div(u, n)
    return k * rbs + lo + (u - k * n)


def shares(c, G):
    rG = rcp(G)
    alo = muldiv_g(512, c, G, rG)
    blo = muldiv_g(256, c, G, rG)
    return alo, muldiv_g(512, c + 1, G, rG) - alo, blo, muldiv_g(256, c + 1, G, rG) - blo


def plan_model():
    out = {'sl1_div_exact_u_lt_65536': True}
    for n in range(1, 9):
        for u in range(1 << 16):
            assert sl1_div(u, n) == u // n, (u, n)
    for G in (1, 2, 3, 7, 64, 128, 131, 132, 133, 255, 256, 257, 300, 511, 512, 600):
        rG = rcp(G)
        for a in (256, 512):
            assert [muldiv_g(a, c, G, rG) for c in range(G + 1)] == [a * c // G for c in range(G + 1)], (a, G)
        for njS in (0, 1, 2, 3, 5, 8, 9, 17, 32):
            A, B = {}, {}
            for c in range(G):
                alo, an, blo, bn = shares(c, G)
                nA, nB = njS * an, njS * bn
                us_a = [mapu(u, alo, an, 512) for u in range(nA)]
                us_b = [mapu(u, blo, bn, 256) for u in range(nB)]
                # time-major order: job index non-decreasing, each job's share contiguous, within [lo, lo+n)
                assert [v // 512 for v in us_a] == sorted(v // 512 for v in us_a)
                assert all(alo <= v % 512 < alo + an for v in us_a) and all(blo <= v % 256 < blo + bn for v in us_b)
                for v in us_a:
                    assert v not in A
                    A[v] = c
                for v in us_b:
                    assert v not in B
                    B[v] = c
                # FAST1 pre-issue (warp 7 lane 0) == consumer / ring q = 0, 1 (FAST1 = 2): same (expert, block)
                n1 = min(2, njS * an)
                pre = [mapu(i, muldiv_g(512, c, G, rG), muldiv_g(512, c + 1, G, rG) - muldiv_g(512, c, G, rG), 512)
                       for i in range(n1)]
                assert pre == us_a[:n1]
                for wsblk in (1, 2):
                    nh = nB - min(wsblk, nB)
                    helper = list(range(nh))
                    tail = [nB - 1 - w for w in range(wsblk) if w < nB]
                    assert sorted(helper + tail) == list(range(nB))
                    kbh = (nB - min(wsblk, nB)) // bn if bn > 0 else 0
                    pub = set(range(kbh))
                    end = set(range(kbh, njS)) if bn > 0 else set()
                    assert pub.isdisjoint(end) and (pub | end) == (set(range(njS)) if bn > 0 else set())
                    # jobs counted by the publisher are fully helper-owned
                    assert all((k + 1) * bn <= nh for k in pub)
            assert sorted(A) == list(range(njS * 512)) and sorted(B) == list(range(njS * 256)), (G, njS)
    out['ownership'] = 'PASS: bijection A/B, time-major order, FAST1 == first units, helper/WSBLK/publisher/end partition'
    return out


def credit_model():
    """Publisher batches (any h sequence) credit each job exactly once with an; sums == 512 per job."""
    rnd = random.Random(7)
    for G in (132, 64, 7):
        for njS in (1, 2, 3, 9, 32):
            tot = [0] * njS
            for c in range(G):
                alo, an, blo, bn = shares(c, G)
                nA = njS * an
                if an == 0:
                    continue
                hs = sorted(rnd.sample(range(an, nA + 1, an), k=min(njS, rnd.randint(1, njS)))) + [nA]
                ad = 0
                for h in hs:
                    ar = min(njS, h // an)
                    for k in range(ad, ar):
                        tot[k] += an
                    ad = max(ad, ar)
                assert ad == njS
            assert tot == [512] * njS, (G, njS, tot)
    return 'PASS: per-job credits sum to 512 for any batch boundaries'


def token_model():
    """Exactly-one completer per token, every routing / ownership / order; empty tokens owned by t mod G."""
    rnd = random.Random(11)
    cases = 0
    for trial in range(400):
        G = rnd.choice((132, 132, 132, 64, 7))
        M = rnd.randint(1, 8)
        ids = [[rnd.randrange(64) for _ in range(8)] for _ in range(M)]  # experts >= 32 are non-local
        # distinct experts per token (top-8 has no duplicates)
        for row in ids:
            seen = set()
            for s in range(8):
                while row[s] in seen:
                    row[s] = rnd.randrange(64)
                seen.add(row[s])
        experts = sorted({e for row in ids for e in row if e < 32})
        jobs = []  # (expert, [tokens]) in ascending expert order (route's job order)
        for e in experts:
            jobs.append((e, [t for t in range(M) for s in range(8) if ids[t][s] == e]))
        L = [sum(1 for s in range(8) if ids[t][s] < 32) for t in range(M)]
        need = [256 * x for x in L]
        njS = len(jobs)
        for tm in (False, True):
            events = []  # (cta, token, n)
            for c in range(G):
                alo, an, blo, bn = shares(c, G)
                if tm:
                    for k in range(njS) if bn > 0 else ():
                        events += [(c, t, bn) for t in jobs[k][1]]
                else:
                    rG = rcp(G)
                    b0, b1 = muldiv_g(njS * 256, c, G, rG), muldiv_g(njS * 256, c + 1, G, rG)
                    if b1 > b0:
                        for jj in range(b0 // 256, (b1 - 1) // 256 + 1):
                            n = min(b1, (jj + 1) * 256) - max(b0, jj * 256)
                            events += [(c, t, n) for t in jobs[jj][1]]
            for order in range(3):
                rnd.shuffle(events)
                cnt = [0] * M
                completers = [[] for _ in range(M)]
                for c, t, n in events:
                    old = cnt[t]
                    cnt[t] += n
                    if old + n == need[t]:
                        completers[t].append(c)
                assert cnt == need, (cnt, need)
                for t in range(M):
                    assert len(completers[t]) == (1 if L[t] else 0), (t, completers[t])
                cases += 1
        writers = [[c for c in range(G) if c == t % G and L[t] == 0] for t in range(M)]
        assert all(len(w) == (1 if L[t] == 0 else 0) for t, w in enumerate(writers))
    return f'PASS: {cases} (routing, ownership, order) cases, exactly one completer per routed token'


def prefetch_model():
    """Warp-level target logic of SWITCH_NEW / PENDING: one outstanding slot, every issue consumed next."""
    rnd = random.Random(5)
    n = 0
    for trial in range(3000):
        njS = rnd.randint(1, 12)
        an, bn = rnd.choice((3, 4)), rnd.choice((1, 2))
        M = rnd.choice((1, 2, 8))
        seq = [(0, k) for k in range(njS) for _ in range(an)] + [(1, k) for k in range(njS) for _ in range(bn)]

        def nxt(pb, j):
            if not pb:
                return (-1 if M == 1 else j + 1) if j + 1 < njS else (4096 if bn > 0 else -1)
            return 4096 + j + 1 if j + 1 < njS else -1

        def flagjob(t):
            if t < 0:
                return -1
            if t < 4096:
                return -1 if t + 1 < njS else (4096 if bn > 0 else -1)
            return t + 1 if (t & 4095) + 1 < njS else -1
        ready = {k: rnd.random() < 0.8 for k in range(njS)}  # flag observed ready at a given check
        pj, pt, fn, fl, curj, staged_hits = -1, -1, -1, False, None, 0
        outstanding = None
        cur_phase = 0
        for pb, j in seq:
            if pb != cur_phase:  # A call -> B call carry: pj, fn, fl kept; pt dropped (re-derived)
                cur_phase, curj, pt = pb, None, -1
            if j != curj:
                curj = j
                enc = (4096 if pb else 0) + j
                if outstanding is not None:   # the one staged prefetch MUST be this switch's job
                    assert outstanding == enc == pj, (outstanding, enc, pj)
                    outstanding = None
                    staged_hits += 1
                else:
                    assert pj == -1, pj
                pj = -1
                pt = nxt(pb, j)
                if pt >= 0:
                    ok = pt < 4096 or (fn == pt and fl)
                    if ok:
                        assert outstanding is None
                        outstanding, pj, pt = pt, pt, -1
                        fn = flagjob(pj)
                    else:
                        fn = pt
                    fl = ready[fn & 4095] if fn >= 0 else False
            # pending check after the MMAs
            if pt >= 0:
                if fl:
                    assert outstanding is None and fn == pt
                    outstanding, pj, pt = pt, pt, -1
                    fn = flagjob(pj)
                    fl = ready[fn & 4095] if fn >= 0 else False
                else:
                    fl = True if rnd.random() < 0.7 else ready[pt & 4095]
        assert outstanding is None and pj == -1, (outstanding, pj)
        n += 1
    return f'PASS: {n} warp sequences, <= 1 outstanding prefetch, every issue consumed by the next switch'


def generation_model():
    """Monotone exit counter: one generation per launch for any CTA start / exit interleaving."""
    rnd = random.Random(3)
    for G in (1, 4, 132):
        ctr6, banks = 0, {0: 'zero', 1: 'zero'}
        for launch in range(40):
            starts, exits, gens = list(range(G)), list(range(G)), {}
            events = [('s', c) for c in range(G)] + [('e', c) for c in range(G)]
            # random interleaving with start(c) before exit(c)
            rnd.shuffle(events)
            done, order = set(), []
            pending = events[:]
            while pending:
                for i, (k, c) in enumerate(pending):
                    if k == 's' or c in done:
                        order.append((k, c))
                        if k == 's':
                            done.add(c)
                        pending.pop(i)
                        break
            for k, c in order:
                if k == 's':
                    gens[c] = ctr6 // G
                else:
                    ctr6 += 1
            g = launch
            assert set(gens.values()) == {g}, (G, launch, set(gens.values()))
            assert banks[g & 1] == 'zero'
            banks[(g + 1) & 1] = 'zero'   # zeroed at start by this launch
            banks[g & 1] = 'used'
            assert ctr6 == (launch + 1) * G
    return 'PASS: gen = ctr[6] / G identical within a launch, +1 per launch, used bank zeroed by the previous launch'


def cuda_ties(variant_cuda):
    """Tie the CPU models to the CUDA text (the expressions the models mirror must be literally present)."""
    need = ['return n == 1 ? u : n == 2 ? (u >> 1) : n == 3 ? (int)(((unsigned)u * 43691u) >> 17) : n == 4 ? (u >> 2) : u / n;',
            'return k * rbs + lo + (u - k * n);',
            'const int alo = muldiv_g(RB_A, c, G, rG), blo = muldiv_g(RB_B, c, G, rG);',
            'L.alo = alo; L.an = muldiv_g(RB_A, c + 1, G, rG) - alo;',
            'L.blo = blo; L.bn = muldiv_g(RB_B, c + 1, G, rG) - blo;',
            'S.wsr[0] = 0; S.wsr[1] = S.njS * L.an; S.wsr[2] = 0; S.wsr[3] = S.njS * L.bn;',
            'const int sl1a0 = muldiv_g(RB_A, blockIdx.x, gridDim.x, rG), sl1an = muldiv_g(RB_A, blockIdx.x + 1, gridDim.x, rG) - sl1a0;',
            'const int kbh = (pc && bn > 0) ? (nB - min(OKC_WSBLK, nB)) / bn : 0;',
            'if (L.bn > 0) { k0 = (nB - min(OKC_WSBLK, nB)) / L.bn; k1 = S.njS; }',
            'k0 = b0 / RB_B; k1 = (b1 - 1) / RB_B + 1;',
            'const int n = tm ? L.bn : min(b1, (jj + 1) * RB_B) - max(b0, jj * RB_B);',
            'if (atomicAdd(tok + t, bn) + bn == sl1_need(S, t))', 'if (atomicAdd(tok + t, n) + n == sl1_need(S, t))',
            'const int ar = an > 0 ? min(aEnd, h / an) : 0;',
            'const int br = (bn > 0 && h > nA) ? min(kbh, (h - nA) / bn) : 0;',
            'for (int k = ad + lane; k < ar; k += 32) atomicAdd(dA + k, an);',
            'if (!PB) return j + 1 < S.njS ? (p.M == 1 ? -1 : j + 1) : (sl1_L().bn > 0 ? 4096 : -1);',
            'return j + 1 < S.njS ? 4096 + j + 1 : -1;',
            'if (t < 4096) return t + 1 < S.njS ? -1 : (sl1_L().bn > 0 ? 4096 : -1);',
            'return (t & 4095) + 1 < S.njS ? t + 1 : -1;',
            '*reinterpret_cast<volatile const int*>(p.ctr + 6) / (int)gridDim.x', 'if (tid == 0) atomicAdd(p.ctr + 6, 1);',
            'const float r = wv[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);',
            'if (sl1_need(S, t) == 0) {', 'for (int t = blockIdx.x; t < p.M; t += gridDim.x) {']
    for s in need:
        assert variant_cuda.count(s) >= 1, s
    # The ship's combine expressions that PC must reproduce (same slot order, same per-pair factor).
    for s in ('const float r = wvp[s] * rsqrtf(ssv[s] * (1.f / HD) + 1e-8f);', 'acc.x += r * yv[s].x;'):
        assert variant_cuda.count(s) == 1, s
    return 'PASS: model expressions present verbatim'


def sync_counts(src):
    pats = {k: len(re.findall(p, src)) for k, p in (
        ('mbar_wait', r'\bmbar_wait\('), ('mbar_arrive1', r'\bmbar_arrive1\('), ('CTA_SYNC', r'\bCTA_SYNC\(\)'),
        ('WS_JOIN', r'\bWS_JOIN\(\)'), ('__syncwarp', r'__syncwarp\(\)'), ('bar.sync3', r'bar\.sync 3'),
        ('bar.sync4', r'bar\.sync 4'), ('bar.arrive3', r'bar\.arrive 3'), ('bar.arrive4', r'bar\.arrive 4'),
        ('__threadfence', r'__threadfence\(\)'), ('ld_acquire', r'ld_acquire\('), ('griddepcontrol.wait', r'griddepcontrol\.wait'))}
    return pats


def cpu_gates(base, variant):
    assert variant == generate(base), 'generated drift: rerun tools/sl1_patches.py'
    b_env = extension_env(variant)
    strings = ('VMS_CUDA_SRC', 'VMS_CPP_SRC', '_HYBLO_CUDA', '_HYB_MOE_CUDA', '_HYB_MOE_CPP', '_vml_CPP',
               '_HYBLO_FLAGS', '_HYBLO_KEYS', 'HYBLO_T', 'HYB_T', 'VMS_MAXT')
    for name in strings:
        assert literal(base, name)[0] == literal(variant, name)[0], name
    # Untouched routes / call ABI / validation: identical function ASTs; prepare differs by ONE inserted call.
    def funcs(s):
        return {n.name: n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)}
    fa, fb = funcs(base), funcs(variant)
    for name in ('_mbl_call', '_mbl_candidates', '_mbl_select', '_mbl_validate', 'kernel_fn', '_mbl_banks', '_hyblo_run',
                 '_hyblo_route', '_hyb_scratch', '_mbl_compare', '_vms_build', '_hyblo_build'):
        assert ast.dump(fa[name]) == ast.dump(fb[name]), name
    pa, pb = fa['prepare'], fb['prepare']
    body = [ast.dump(x) for x in pb.body]
    ins = [i for i, x in enumerate(pb.body) if '_sl1_init' in ast.dump(x)]
    assert len(ins) == 1 and body[:ins[0]] + body[ins[0] + 1:] == [ast.dump(x) for x in pa.body], 'prepare drift'
    cu, line = literal(base, CUDA_NAME)
    edits = b_env['_SL1_EDITS']
    patched = b_env['sl1_sources']('tiny')[0]
    assert patched == b_env['sl1_sources']('tiny2')[0] == apply_edits(cu, edits)
    assert b_env['sl1_sources']('tiny', False)[0] == cu
    back = patched
    for bit, old, new in reversed(edits):
        back = replace_once(back, new, old)
    assert back == cu, 'edits not reversible'
    sa, sb = sync_counts(cu), sync_counts(patched)
    expect = dict(sa)
    expect.update({'CTA_SYNC': sa['CTA_SYNC'] + 1, 'bar.sync3': 1, 'bar.sync4': 1, 'bar.arrive3': 2, 'bar.arrive4': 1,
                   '__syncwarp': sa['__syncwarp'] + 4, '__threadfence': sa['__threadfence'] + 4,
                   'griddepcontrol.wait': sa['griddepcontrol.wait'] + 1})
    assert sb == expect, {'base': sa, 'variant': sb, 'expect': expect}
    report = {'status': 'PASS', 'original_strings_equal': list(strings), 'untouched_functions_equal': True,
              'prepare_insertion_only': True, 'edits': len(edits), 'reversible': True, 'sync_counts': {'base': sa, 'variant': sb},
              'models': {'plan': plan_model(), 'credits': credit_model(), 'tokens': token_model(),
                         'prefetch': prefetch_model(), 'generation': generation_model(), 'cuda_ties': cuda_ties(patched)},
              'smem': smem_budget(cu), 'scratch_ints_used_before_ss1': 2 * 88 + 2 * 384,
              'sha256': hashlib.sha256(variant.encode()).hexdigest()}
    assert report['scratch_ints_used_before_ss1'] <= 1024
    return report


def smem_budget(cu):
    d = {k: int(v) for k, v in re.findall(r'-DOKC_(NS|NSF|NW)=(\d+)', ' '.join(literal(BASE.read_text(encoding='utf-8'), '_HYBLO_FLAGS')[0]))}
    unit, ft_t, xp_ft = 36864, 64, 264
    free = d['NSF'] * unit + 2 * ft_t * xp_ft * 2 - d['NS'] * unit
    slots = d['NW'] * 2 * (64 // d['NW']) * 512
    state = 4 * (8 + 64 + 4 * d['NW'])
    assert 'constexpr int UNIT_CODES = 32768, UNIT_SC = 4096' in cu and 'constexpr int XP_FT = KS + 8;' in cu and 'constexpr int KS = 256;' in cu
    assert slots + state <= free, (slots, state, free)
    return {'free_ft_tail_B': free, 'prefetch_slots_B': slots, 'sl1_state_B': state, 'spare_B': free - slots - state}


def bit_lines(cu, edits, offset):
    out = {}
    src = cu
    for bit, old, new in edits:
        at = src.index(old)
        line = offset + src[:at].count('\n') + 1
        out.setdefault(bit, []).append({'cuda_line_in_edited_text': src[:at].count('\n') + 1,
                                        'base_file_line_approx': line, 'old_lines': old.count('\n') + 1,
                                        'new_lines': new.count('\n') + 1})
        src = src[:at] + new + src[at + len(old):]
    return out


def change_report(base, variant):
    cu, line = literal(base, CUDA_NAME)
    env = extension_env(variant)
    edits = env['_SL1_EDITS']
    patched = apply_edits(cu, edits)
    a, b = cu.splitlines(), patched.splitlines()
    diff = [{'kind': tag, 'base_file_lines': [line + i, line + j - 1] if i < j else [],
             'base_insert_before_file_line': line + i if i == j else None,
             'variant_cuda_lines': [k + 1, l] if k < l else [], 'before': a[i:j], 'after': b[k:l]}
            for tag, i, j, k, l in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if tag != 'equal']
    # line of the CUDA string start is `_HYBLO_CUDA=r"""#include ...` (CUDA line 1 = file line `line`)
    return {'cuda_string_file_line': line, 'per_edit': [{'bit': bit, 'before': old, 'after': new} for bit, old, new in edits],
            'per_bit_positions': bit_lines(cu, edits, line - 1), 'final_diff': diff}


def target_report(path, out=None):
    """Apply the SAME CUDA edits to another file's _HYBLO_CUDA (e.g. kernels/lab/k3_diet_rq128.py)."""
    text = Path(path).read_text(encoding='utf-8')
    cu, line = literal(text, CUDA_NAME)
    base_cu, _ = literal(BASE.read_text(encoding='utf-8'), CUDA_NAME)
    edits = cuda_edits(cu)
    same = edits == cuda_edits(base_cu)
    rep = {'target': str(path), 'cuda_line': line, 'cuda_equal_to_board': cu == base_cu, 'edits_exact_once': len(edits),
           'edits_identical_to_board': same, 'cpp_needs': 'void sl1_set(int64_t v); in the extension cpp + "sl1_set" in functions',
           'lab_guard': 'HOST_CHECKS requires okc_set_pdl(0) (lab-only; see P5 for the PDL argument before dropping it)'}
    if out:
        new_cu = apply_edits(cu, edits)
        # Replace the string literal body in place (raw triple-quoted literal in both files).
        start = text.index('_HYBLO_CUDA=r"""') + len('_HYBLO_CUDA=r"""')
        end = text.index('"""', start)
        assert text[start:end] == cu
        Path(out).write_bytes((text[:start] + new_cu + text[end:]).encode('utf-8'))
        rep['written'] = str(out)
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--check', action='store_true', help='verify the generated file is current (read-only)')
    p.add_argument('--changes', action='store_true', help='include the exact change report')
    p.add_argument('--target', help='apply the SAME CUDA edits to this file\'s _HYBLO_CUDA (report only unless --out)')
    p.add_argument('--out', help='with --target: write the edited copy here (new file)')
    a = p.parse_args()
    if a.target:
        print(json.dumps(target_report(a.target, a.out), indent=2))
        return
    base = BASE.read_text(encoding='utf-8')
    variant = generate(base)
    if a.check:
        assert DEST.read_bytes() == variant.encode('utf-8'), 'regenerate kernels/muse_moe_w4a16/mb_layer_sl1.py'
    else:
        DEST.write_bytes(variant.encode('utf-8'))
    result = cpu_gates(base, variant)
    if a.changes:
        result['changes'] = change_report(base, variant)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

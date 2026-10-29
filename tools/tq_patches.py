"""tq_patches.py: TICKET QUEUE (VM_TQ) + null consumers (VM_NULL) for the VML MoE at T >= 32 (moe-tq builder, 2026-10-29).

  python tools/tq_patches.py skel [--base=kernels/muse_decode_layer/k3_gc_21.py]   -> kernels/lab/k3_tq_s.py (step-1 skeleton arms)
  python tools/tq_patches.py rider --base=<k3 file> --out=<file>                    (step 2: VM_TQ routed at T >= 32, real consumers)

VM_TQ (compile time; bf2 / w72 configurations, VM_RT route):
  * unit = (job, 256-row block, full K, 16 stages); no K-splits, no fix-up slots, no cost tables in the schedule.
  * route(): warp NCW-1 writes T.tord = jobs in heavy-first order (tier descending, job index ascending) next to the job table.
  * tickets: ONE global counter (A.gbar[0], reset by vm_fin). CTA c's first ticket is c (static); the w-producer lane claims
    the next ticket with one atomicAdd when it issues stage A.tqat of its current unit (claim 1 unit ahead, result consumed at
    the next unit's first issue), publishes each unit in T.tq[k & 15] / T.tqpub (DP's q / qpub machinery at unit quantum);
    the x producer and the 16 consumer warps follow T.tq in order.
  * ticket -> unit (tq_unit): A.tqlag < 0: A-then-B (all A units heavy-first, then all B units heavy-first);
    A.tqlag = L >= 0: merged, B block k (16 units of heavy-first job k) starts at merged index min(32 (k + 1) + L, nA) + 16 k.
    Every B ticket of a job follows all of its A tickets (no deadlock: lowest unfinished ticket always runs; deferred doneA
    releases are flushed before any B unit).
  * B readiness: the x producer waits doneA[j][s] per stage (as shipped); A units wait xready[j] (gather) once.
VM_NULL (lab only): consumers wait each weight slot, release it, wait the x slot, optionally spin
  A.ncyc[phase x tier] cycles (0: pure null; the 'truth' pace, independent of the planner's cost table), release the x slot; epilogues / fix-ups / releases run as shipped
  on zero accumulators. Marks (VM_MKW 64, TQ; columns free in TQ builds): 51 claims, 55 idle-on-claim ns sum,
  62 = max idle ns << 32 | first-claim RTT ns (sync), 63 = claims idle < 300 ns << 32 | < 100 ns; RTT mode (A.tqat | 256):
  every claim synchronous, 55 = RTT sum, 62 hi = max RTT, 63 = 8 x 8-bit RTT histogram (< 0.5 / 1 / 1.5 / 2 / 3 / 4.5 / 6 us / more);
  49 / 50 units A / B;
  6 first B unit start (phase A end); 40..44 x producer (B doneA wait ns, B stages that waited, A xready wait, first / last B).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_21.py"


def _sub(s, old, new, n=1):
    c = s.count(old)
    assert c == n, (c, n, old[:120])
    return s.replace(old, new)


DEFAULTS = r"""#ifndef VM_TQ
#define VM_TQ 0
#endif
#ifndef VM_NULL
#define VM_NULL 0
#endif
"""

TAB = r"""#if VM_TQ
long long tqmk[4];                 // TQ marks (smem, not registers: the w72 producer runs at PREG 32)
int tqpub, tqpad0;
int tq[16];
uint8_t tord[JMAX];                // TQ: heavy-first job order (route, warp NCW-1)
#endif
"""

ARGS = r"""#if VM_TQ || VM_NULL
int tqlag, tqat, tqpad0, tqpad;
#endif
#if VM_NULL
uint32_t ncyc[2 * VM_NTM];         // NULL: emulated consumer cycles per stage by phase x tier (0: pure null)
#endif
"""

ROUTE = r"""#if VM_TQ
{   // TQ: heavy-first job order (tier descending, job index ascending); every CTA computes the same table
int pos = 0;
for (int nt = VM_NTM - 1; nt >= 0; --nt) {
int n = 0;
for (int k = 0; k < nj; ++k) n += ((((min(base, tot - k * base) + 7) >> 3) - 1) == nt) ? 1 : 0;
const int ex = excl_scan(n, lane);
int p = pos + ex;
for (int k = 0; k < nj; ++k)
if ((((min(base, tot - k * base) + 7) >> 3) - 1) == nt) T.tord[p++] = (uint8_t)(jst + k);
pos += __shfl_sync(0xffffffffu, ex + n, 31);
}
if (lane == 0) T.tqpub = 0;
}
#endif
"""

UNIT = r"""#if VM_TQ
static_assert(VM_RT && !VM_DP && !VM_SC && !VM_AD && !VM_DEFER2 && !VM_PS && VM_LD != 1 && JMAX <= 255, "VM_TQ: bf2 / w72 configurations");
#define TQ_HAS_WORK(T) ((int)blockIdx.x < 48 * (T).nj)
__device__ __forceinline__ int tq_unit(const Tab& T, int t, int L) {   // ticket -> (pb << 16) | unit (A: j * 32 + ub, B: j * 16 + ub)
const int nj = T.nj, nA = nj << 5;
if (t < 0 || t >= 48 * nj) return -1;
if (L < 0) {   // A-then-B
if (t < nA) return (int)T.tord[t >> 5] * 32 + (t & 31);
const int b = t - nA;
return (1 << 16) | ((int)T.tord[b >> 4] * 16 + (b & 15));
}
int lo = 0, hi = nj;   // k = number of B blocks starting at or before t; block i starts at min(32 (i + 1) + L, nA) + 16 i
while (lo < hi) { const int m = (lo + hi) >> 1; if (min(32 * (m + 1) + L, nA) + 16 * m <= t) lo = m + 1; else hi = m; }
const int k = lo;
if (k > 0) {
const int p = min(32 * k + L, nA) + 16 * (k - 1);
if (t < p + 16) return (1 << 16) | ((int)T.tord[k - 1] * 16 + (t - p));
}
const int a = t - 16 * k;
return (int)T.tord[a >> 5] * 32 + (a & 31);
}
__device__ __forceinline__ int tq_nfill(const Args& A, const Tab& T) {
const int v = tq_unit(T, blockIdx.x, A.tqlag);
return ((A.flags & 64) && v >= 0 && (v >> 16) == 0) ? NX - 1 : 0;
}
__device__ void tq_self_fill(const Args& A, const Tab& T, uint8_t* smem, uint32_t bars, const uint16_t* ent, int nf) {
const int tid = threadIdx.x, v = tq_unit(T, blockIdx.x, A.tqlag);
const int j = (v >= 0) ? ((v & 0xFFFF) >> 5) : 0, cn = T.jC[j];
for (int f = 0; f < nf; ++f) {
const int st = f;
uint32_t* xs = reinterpret_cast<uint32_t*>(smem + NS * US + f * XS);
for (int w = tid; w < cn * 32; w += 32 * NCW) {
const int r = w >> 5, ch = w & 31, i = ch >> 3, t = (ch >> 1) & 3, qb = (ch & 1) * 4;
const uint4 x4 = __ldg(reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM + 256 * st) + ch);
uint32_t* d = xs + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
d[0] = x4.x; d[32] = x4.y; d[64] = x4.z; d[96] = x4.w;
}
}
asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
bar_sync(5, 32 * NCW);
if (tid == 0)
for (int f = 0; f < nf; ++f) mbar_arrive(bars + 8 * (2 * NS + f));
}
#endif
"""

PRODW = r"""#if VM_TQ
{
Tab& TT = const_cast<Tab&>(T);
const int G = gridDim.x, L = A.tqlag, at = min(max(A.tqat & 255, 0), 15);
long long* mk = (VM_MKW == 64 && A.marks) ? A.marks + (size_t)blockIdx.x * MKW : nullptr;   // TQ marks: MKW 64 builds only
const bool rtm = mk && (A.tqat & 256);   // marks, RTT mode: every claim synchronous (round-trip histogram), no idle measure
int it = 0, k = 0, t = blockIdx.x;
if (mk) { TT.tqmk[0] = 0; TT.tqmk[1] = 0; TT.tqmk[2] = 0; TT.tqmk[3] = 0; }
for (;;) {
const int v = tq_unit(T, t, L);
TT.tq[k & 15] = v;
__threadfence_block();
*reinterpret_cast<volatile int*>(&TT.tqpub) = k + 1;
++k;
if (v < 0) break;
const int pb = v >> 16, u = v & 0xFFFF;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int e = T.jE[j];
const uint8_t* wb = pb ? A.ws2 + (size_t)(e * 32 + 2 * ub) * 16 * SW : A.ws13 + (size_t)(e * 64 + 2 * ub) * 16 * SW;
int tn = -1;
for (int s = 0; s < 16; ++s, ++it) {
if (s == at) {
if (mk && (k == 1 || rtm)) {   // marks only: synchronous claim round trip (the producer usually waits for a slot here anyway)
const long long t0 = gtime();
tn = atomicAdd(A.gbar, 1) + G;
int tv;
asm volatile("mov.b32 %0, %1;" : "=r"(tv) : "r"(tn) : "memory");
const long long w = gtime() - t0;
tn = tv;
if (k == 1) TT.tqmk[3] = w;
if (rtm) {
TT.tqmk[0] += w; if (w > TT.tqmk[1]) TT.tqmk[1] = w;
const int bin = (w < 500) ? 0 : (w < 1000) ? 1 : (w < 1500) ? 2 : (w < 2000) ? 3 : (w < 3000) ? 4 : (w < 4500) ? 5 : (w < 6000) ? 6 : 7;
TT.tqmk[2] += 1ll << (8 * bin);
}
} else {
tn = atomicAdd(A.gbar, 1) + G;
}
}
const uint8_t* src = wb + (size_t)s * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
if (mk && !rtm) {   // idle-on-claim: stall of the producer lane on the claimed ticket at the next unit's first issue
const long long t0 = gtime();
int tv;
asm volatile("mov.b32 %0, %1;" : "=r"(tv) : "r"(tn) : "memory");
const long long w = gtime() - t0;
TT.tqmk[0] += w; if (w > TT.tqmk[1]) TT.tqmk[1] = w;
TT.tqmk[2] += 1ll + ((long long)(w < 300) << 20) + ((long long)(w < 100) << 40);
t = tv;
} else {
t = tn;
}
}
if (mk) {
const long long c2 = TT.tqmk[2];
mk[51] = rtm ? 0 : (c2 & 0xFFFFF); mk[55] = TT.tqmk[0]; mk[62] = (TT.tqmk[1] << 32) | (TT.tqmk[3] & 0xFFFFFFFFll);
mk[63] = rtm ? c2 : ((((c2 >> 20) & 0xFFFFF) << 32) | ((c2 >> 40) & 0xFFFFF));
}
return;
}
#endif
"""

PRODX = r"""#if VM_TQ
{
const int c = blockIdx.x, nf = tq_nfill(A, T);
int it = 0, kq = 0, nB = 0;
long long wA = 0, wB = 0, tB0 = 0, tB1 = 0;
for (;;) {
if (*reinterpret_cast<const volatile int*>(&T.tqpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.tqpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 4); break; }
}
}
__threadfence_block();
const int v = T.tq[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
const int pb = v >> 16, u = v & 0xFFFF, j = pb ? (u >> 4) : (u >> 5);
const int npad = (T.jC[j] + 7) & ~7;
if (!pb) {
if (A.marks && ld_acquire(A.xready + j) < npad) {
const long long t = gtime();
wait_flag(A.xready + j, npad, A.err);
wA += gtime() - t;
} else {
wait_flag(A.xready + j, npad, A.err);
}
fence_proxy_async();
}
for (int s = 0; s < 16; ++s, ++it) {
if (kq == 1 && s < nf) continue;
if (pb) {
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
}
if (A.marks) {
long long* m = A.marks + (size_t)c * MKW;
m[MKX] = wB; m[MKX + 1] = nB; m[MKX + 2] = wA; m[MKX + 3] = tB0; m[MKX + 4] = tB1;
}
return;
}
#endif
"""

NULLST = r"""#if VM_NULL
__device__ __forceinline__ void null_stage(const Args& A, uint32_t bars, int& it, int q, int wp, int lane, bool first, Pend& pd, int ci) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
if (A.ncyc[ci] != 0) {   // emulated consumer time per stage (cycles, by phase x tier)
const long long t1 = clock64() + (long long)A.ncyc[ci];
while (clock64() < t1) { }
}
if constexpr (VM_DEFER) { if (first) pend_flush(A, pd, q, wp, lane); }
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
++it;
}
#endif
"""

SEG_NULL = r"""#if VM_NULL
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = 0.f;
#pragma unroll 1
for (int s = s0; s < s1; ++s) null_stage(A, bars, it, q, wp, lane, s == s0, pd, (PB ? VM_NTM : 0) + N / 8 - 1);
#else
bool sync_path = false;
"""

TQPHASE = r"""#if VM_TQ
template <bool PB>
__device__ __forceinline__ void tq_seg(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int u, int q, int wp, int lane, Pend& pd) {
const int j = PB ? (u >> 4) : (u >> 5);
if (A.marks && q == 0 && wp == 0 && lane == 0) {
long long* mk = A.marks + (size_t)blockIdx.x * MKW;
mk[8 + (VM_MKW == 64 ? 16 : 8) * PB + ((T.jC[j] + 7) >> 3) - 1] += 16;
#if VM_MKW == 64
mk[49 + PB] += 1;
#endif
}
switch ((T.jC[j] + 7) >> 3) {
case 1: seg<PB, 8>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 2: seg<PB, 16>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 3: seg<PB, 24>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 4: seg<PB, 32>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 5: seg<PB, 40>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 6: seg<PB, 48>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
case 7: seg<PB, 56>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
#if VM_NTM > 8
case 8: seg<PB, 64>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
#endif
default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, 0, 16, q, wp, lane, pd); break;
}
}
__device__ __forceinline__ void tq_phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int q, int wp, int lane) {
Pend pd{-1};
int it = 0, kq = 0;
bool inB = false;
for (;;) {
if (*reinterpret_cast<const volatile int*>(&T.tqpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.tqpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 5); break; }
}
}
__threadfence_block();
const int v = T.tq[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
if (v >> 16) {
if (!inB) {
inB = true;
if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();
}
pend_flush(A, pd, q, wp, lane);   // this WG's deferred doneA release may gate the B unit's x
tq_seg<true>(A, T, smem, bars, it, v & 0xFFFF, q, wp, lane, pd);
} else {
tq_seg<false>(A, T, smem, bars, it, v & 0xFFFF, q, wp, lane, pd);
}
}
pend_flush(A, pd, q, wp, lane);
if (!inB && wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();
}
#endif
"""

MAIN_OLD = """const int q = warp >> 2, wp = warp & 3;
int it = 0;
#if VM_DP
int kq = 0;
phase<false>(A, T, smem, bars, it, q, wp, lane, kq);
"""
MAIN_NEW = """const int q = warp >> 2, wp = warp & 3;
#if VM_TQ
tq_phase(A, T, smem, bars, q, wp, lane);
#else
int it = 0;
#if VM_DP
int kq = 0;
phase<false>(A, T, smem, bars, it, q, wp, lane, kq);
"""
MAIN_END_OLD = """#else
phase<true>(A, T, smem, bars, it, q, wp, lane);
#endif
if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();
"""
MAIN_END_NEW = """#else
phase<true>(A, T, smem, bars, it, q, wp, lane);
#endif
#endif
if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();
"""

HOST = r"""#if VM_TQ || VM_NULL
static_assert(!VM_DP && !VM_SC && !VM_PS, "vml_run: TQ / NULL args follow the cost table");
TORCH_CHECK((int64_t)V.size() >= 9 + 2 * VM_NTM + (VM_NULL ? 2 * VM_NTM : 0), "vml_run: TQ / NULL args missing");
a.tqlag = (int)V[6 + 2 * VM_NTM]; a.tqat = (int)V[7 + 2 * VM_NTM]; a.tqpad0 = 0; a.tqpad = 0;
#if VM_NULL
for (int k = 0; k < 2 * VM_NTM; ++k) a.ncyc[k] = (uint32_t)V[9 + 2 * VM_NTM + k];
#endif
#endif
"""

PYEXT = '''

def _vml_tq_ext(mod, tq):
    c = _vml_cfg(mod)
    if len(c) <= 12 or not (c[11] or c[12]):
        return []
    out = [int(tq[0]), int(tq[1]), 0]
    if c[12]:
        cyc = list(tq[2]) if len(tq) > 2 and tq[2] else [0] * (2 * c[1])
        assert len(cyc) == 2 * c[1], (len(cyc), c[1])
        out += [int(v) for v in cyc]
    return out
'''


def cuda(s):
    """VM_TQ + VM_NULL in the VML CUDA source and its Python call path (defaults 0: the shipped kernel)."""
    a = s.index('_vml_CUDA_T = r"""')
    b = s.index('"""', a + 20)
    c = s[a:b]
    c = _sub(c, "#ifndef VM_DPI\n#define VM_DPI 4\n#endif\n", "#ifndef VM_DPI\n#define VM_DPI 4\n#endif\n" + DEFAULTS)
    c = _sub(c, "unsigned long long bperm[JMAX];   // bit 9", TAB + "unsigned long long bperm[JMAX];   // bit 9")
    c = _sub(c, "#if VM_MKW == 64\nint G0;\n#endif\n};\n", "#if VM_MKW == 64\nint G0;\n#endif\n" + ARGS + "};\n")
    c = _sub(c, "T.jS[njT] = est + tot; T.jrow[njT] = rst + prow; T.cpA[njT] = totA; T.cpB[njT] = totB;\n}\n} else if (warp < 10) {",
             "T.jS[njT] = est + tot; T.jrow[njT] = rst + prow; T.cpA[njT] = totA; T.cpB[njT] = totB;\n}\n" + ROUTE
             + "} else if (warp < 10) {")
    c = _sub(c, "__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n",
             UNIT + "__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n" + PRODW)
    c = _sub(c, "__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {\n",
             "__device__ void prod_x(const Args& A, const Tab& T, uint32_t xring, uint32_t bars) {\n" + PRODX)
    c = _sub(c, "template <bool PB, int N>\n__device__ __forceinline__ void seg(", NULLST + "template <bool PB, int N>\n__device__ __forceinline__ void seg(")
    c = _sub(c, "bool sync_path = false;\n", SEG_NULL)
    c = _sub(c, "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n}\nif constexpr (VM_DEFER2) pend_flush(A, pd, q, wp, lane);",
             "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n}\n#endif\nif constexpr (VM_DEFER2) pend_flush(A, pd, q, wp, lane);")
    c = _sub(c, "if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);   // before phase B: its x waits on these releases\n}\n",
             "if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);   // before phase B: its x waits on these releases\n}\n" + TQPHASE)
    c = _sub(c, "if (A.flags & 64) self_fill(A, T, smem, bars, ent, nfill_of(A, T));\n",
             "#if VM_TQ\nif (A.flags & 64) tq_self_fill(A, T, smem, bars, ent, tq_nfill(A, T));\n#else\n"
             "if (A.flags & 64) self_fill(A, T, smem, bars, ent, nfill_of(A, T));\n#endif\n")
    c = _sub(c, "if (tid == 0 && A.marks && (T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) + (T.bB[blockIdx.x + 1] - T.bB[blockIdx.x]) > 0) {",
             "#if VM_TQ\nif (tid == 0 && A.marks && TQ_HAS_WORK(T)) {\n#else\n"
             "if (tid == 0 && A.marks && (T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) + (T.bB[blockIdx.x + 1] - T.bB[blockIdx.x]) > 0) {\n#endif")
    c = _sub(c, MAIN_OLD, MAIN_NEW)
    c = _sub(c, MAIN_END_OLD, MAIN_END_NEW)
    c = _sub(c, "#if VM_DP\nif (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }\n#endif\n",
             "#if VM_DP || VM_TQ\nif (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }\n#endif\n")
    c = _sub(c, "a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];\n", HOST + "a.magG = (unsigned long long)V[3]; a.flags = (int)V[5];\n")
    c = _sub(c, "VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS}; }", "VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS, VM_TQ, VM_NULL}; }")
    s = s[:a] + c + s[b:]
    s = _sub(s, "\ndef _vml_cfg(mod):\n", PYEXT + "\n\ndef _vml_cfg(mod):\n")
    s = _sub(s, "def _vml_call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None, pool=(0, 0), seg=(0, 0), pdel=0):",
             "def _vml_call(mod, k, sc, x, ids, wts, ls=0, marks=None, mask=3, out=None, flags=None, cost=None, pool=(0, 0), seg=(0, 0), pdel=0, tq=(-1, 12, None)):")
    s = _sub(s, "+ _vml_seg_ext(mod, seg) + _vml_pdel_ext(mod, pdel))\n",
             "+ _vml_seg_ext(mod, seg) + _vml_pdel_ext(mod, pdel) + _vml_tq_ext(mod, tq))\n")
    return s


def defs(s, new):
    """add variants to the _VML_DEFS dict literal (kept a literal: tools/pdl_sass.py reads it from the AST)."""
    a = s.index("\n_VML_DEFS = {") + 1
    b = s.index("\n_VML = {", a)
    blk = s[a:b]
    assert blk.endswith(")}"), blk[-20:]
    add = ""
    for v, (src, extra) in new.items():
        i = blk.index(f'"{src}": (') + len(f'"{src}": (')
        k = blk.index(")", i)
        add += f',\n             "{v}": (' + blk[i:k] + "".join(f', "{d}"' for d in extra) + ")"
    return s[:a] + blk[:-1] + add + "}" + s[b:]


SKEL_DEFS = {"bf2n": ("bf2", ("-DVM_NULL=1", "-DVM_MKW=64")), "bf2q": ("bf2", ("-DVM_NULL=1", "-DVM_TQ=1", "-DVM_MKW=64")),
             "w72n": ("w72", ("-DVM_NULL=1", "-DVM_MKW=64")), "w72q": ("w72", ("-DVM_NULL=1", "-DVM_TQ=1", "-DVM_MKW=64"))}
FULL_DEFS = {"bf2t": ("bf2", ("-DVM_TQ=1",)), "w72t": ("w72", ("-DVM_TQ=1",)),
             "bf2tm": ("bf2", ("-DVM_TQ=1", "-DVM_MKW=64")), "w72tm": ("w72", ("-DVM_TQ=1", "-DVM_MKW=64"))}


def nonce(s, tag):
    lines = s.split("\n", 2)
    assert lines[1].startswith("# nonce "), lines[1][:60]
    lines[1] = f"# nonce {tag}"
    return "\n".join(lines)


def build(name, s):
    p = ROOT / "kernels/lab" / name
    p.write_text(s, encoding="utf-8", newline="\n")
    print(f"wrote {p.relative_to(ROOT)} ({len(s)} B)")


def main():
    mode = sys.argv[1]
    base_p = next((ROOT / a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--base=")), BASE)
    out_p = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--out=")), None)
    s0 = base_p.read_text(encoding="utf-8")
    if mode == "skel":
        s = defs(cuda(s0), {**SKEL_DEFS, **FULL_DEFS})
        build("k3_tq_s.py", nonce(s, f"k3-tq-s ({base_p.name} + VM_TQ / VM_NULL arms bf2n bf2q w72n w72q bf2t w72t; tools/tq_patches.py; lab)"))
    elif mode == "rider":
        assert out_p, "--out=kernels/...py required"
        raise SystemExit("rider: written after step 1 / step 2 (routing + tq args at the call site)")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

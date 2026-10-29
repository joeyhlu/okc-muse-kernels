"""rt_patches.py: VM_RT = route() of the VML MoE kernel (vm_main) in two barriers instead of six (R-rt-1, 2026-10-28).

Cause (R-m32e-18 marks + source): after griddepcontrol.wait the consumers run six CTA barriers in route(), with two
warp-0-only serial sections between them (16 dependent hist loads + 3 scans + the job loop; the 32-job cost-scan loop),
two __match_any_sync passes per 32-entry chunk (MATCH is a slow multi-pass op), and a 7-step dependent smem binary
search per thread for bA / bB. The 1-wave CTAs all wait on the slowest of these, so the latency adds up serially.

VM_RT 1 (compile-time; default 0 = the shipped route, unchanged):
  1. ids loads unpredicated (all 5 in flight), the cost / mag tables read with them and staged in smem (no indexed-param
     LDC chains later); per 32-entry chunk six ballots (valid + 5 id bits) give each entry's rank in its chunk and each
     expert lane's count (no match_any, no shared RMW); each warp stores its 32-expert row: hist needs no zeroing.
  2. ONE CTA barrier. Every consumer warp reads the 16 rows (tot, its own warp prefix), est = scan(tot), and scatters its
     own entries: r = shfl(est + prefix + chunk base, x) + chunk rank (ent, rmap).
  3. The plan once, in the last warp: job split (the same closed form), padded rows, per-expert costs, their scans; it
     hands 6 x 32 per-expert values to warps 0-9 on named barrier 10, then writes the job table (jE .. cpB) off the
     critical path. Warps 0-9 (0-4 phase A, 5-9 phase B) find bA / bB by a 5-step shuffle binary search over the 32
     experts' cost starts + a walk over that expert's (<= 5) jobs: the same (job, stage) as stage_at's search over cp[].
  4. ONE closing barrier. Tab, ent and rmap are bit-identical to VM_RT 0 (tools/rt_local_test.py checks them).
Static plan only (not with VM_DP / VM_SC).
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTE_HEAD = "__device__ void route(const Args& A, Tab& T, int* hist, uint16_t* ent, uint32_t* fin) {\n"
GATHER_HEAD = "__device__ void gather(const Args& A, const Tab& T, const uint16_t* ent) {\n"
_MB = "if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));"
_LOOP = "if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }"

RT_SRC = r"""#ifndef VM_RT
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
if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));
#else
const int nj = (tot + VM_JN - 1) / VM_JN;
const int base = nj ? (tot + nj - 1) / nj : VM_JN;
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
"""
RT_END = "#endif   // VM_RT\n"


def sub1(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"{what}: anchor found {n} times: {old[:90]!r}"
    return text.replace(old, new)


def apply(cuda):
    """VML CUDA text (stripped, m32e-patched with W5T) -> + the VM_RT switch around route()."""
    assert "VM_RSYNC()" in cuda and "#define VM_RT" not in cuda, "needs the W5T VM_RSYNC macro, not yet RT-patched"
    cuda = sub1(cuda, ROUTE_HEAD, RT_SRC + ROUTE_HEAD, "route head")
    cuda = sub1(cuda, "\n" + GATHER_HEAD, "\n" + RT_END + GATHER_HEAD, "gather head")
    return cuda


def make_lab():
    """kernels/lab/rt_moe.py = kernels/lab/m32e_moe.py with the ship's closed-form job split (gen_k3_mb.py) + VM_RT."""
    src = (ROOT / "kernels/lab/m32e_moe.py").read_text(encoding="utf-8")
    a = src.index('CUDA_T = r"""') + len('CUDA_T = r"""')
    b = src.index('"""', a)
    cuda = sub1(src[a:b], _LOOP, _MB, "closed-form split")
    out = src[:a] + apply(cuda) + src[b:]
    out = '"""rt_moe.py: GENERATED by tools/rt_patches.py (m32e_moe.py + MB split + VM_RT) - do not edit.\n' + out[3:]
    dst = ROOT / "kernels/lab/rt_moe.py"
    dst.write_text(out, encoding="utf-8", newline="\n")
    return dst


if __name__ == "__main__":
    print(f"[rt] wrote {make_lab()}")

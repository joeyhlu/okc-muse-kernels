"""vff_patches.py: VM_FF = fused per-token finalize inside vm_main, no vm_fin launch (moe redesign, R-vff; bs128 / bs256 tail).
Ship: vm_fin (one CTA per token, PDL) can only become resident as vm_main CTAs retire, waits for the whole grid, then loads rmap,
then y (two dependent round trips) before its combine. VM_FF 1:
  - route() writes rmap from every CTA (identical values), so each CTA later reads entries its own threads wrote;
  - after phase B each CTA (all 16 consumer warps, bar 11) fences once and adds its B stage-range overlap with every job it
    touched to doneB[job] (A.cnt + 2048: the static plan cuts the job-major B stage list [256 j, 256 j + 256) into
    [bB[c], bB[c + 1]), so each job sums to exactly 256, split shares included);
  - then the CTA's two 256-thread halves claim tokens from one counter; warp 0 of a half loads the token's route rows, finds each
    row's job (jS binary search), waits doneB == 256 (acquire), compacts the routes; the half runs vm_fin's VM_FTR math
    (bf16r(y) * (1 + nw), per-route sum of squares, rsqrt(ss / 4096 + 1e-8) * weight, routes in rmap order; VM_FF_CH routes per
    chunk, default 2: same accumulation order as vm_fin's 4) and zeroes the y rows
    it read (VM_BRED), as vm_fin does;
  - the last CTA to finish (exit counter) resets xready / doneA / the split slots / doneB / claim / exit for the next launch.
Most tokens finish inside the phase-B end spread (b256 marks: end p50 210.9, max 218.2 us). Needs VM_BRED, VM_FTR, !VM_DP, NCW 16.
At VM_FF 0 the source preprocesses to the base's.
    python -I tools/vff_patches.py <file.py>
"""
import pathlib
import sys

DEF_A = "#ifndef VM_BRED\n#define VM_BRED 0\n#endif\n"
DEF_N = DEF_A + "#ifndef VM_FF\n#define VM_FF 0   // R-vff: 1 = fused per-token finalize in vm_main (no vm_fin launch)\n#endif\n"
OLD_RMAP = "if (blockIdx.x == 0 && e < e1) A.rmap[e] = r;"
NEW_RMAP = "if ((VM_FF || blockIdx.x == 0) && e < e1) A.rmap[e] = r;   // R-vff: every CTA (same values) under VM_FF"
OLD_MAIN = "__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n"
NEW_MAIN = r"""#if VM_FF
static_assert(VM_BRED && VM_FTR && !VM_DP && NCW == 16, "VM_FF: BRED / FTR static-plan builds with 16 consumer warps");
#ifndef VM_FF_LW
#define VM_FF_LW 0   // 1 = norm weights reloaded per chunk (fewer live registers)
#endif
#ifndef VM_FF_NI
#define VM_FF_NI 0   // 1 = ff_tail not inlined
#endif
#ifndef VM_FF_CH
#define VM_FF_CH 2   // routes per chunk (vm_fin: 4); 2 keeps the tail inside the 96-register cap
#endif
constexpr int FF_CH = VM_FF_CH;
constexpr int FF_OFF = 2048;   // doneB[JMAX], claim, exit: A.cnt beyond the split slots
static_assert(2 * SLOTSD * 4 <= FF_OFF && FF_OFF + JMAX + 2 <= 2 * (SLOTS + 384) * 4, "VM_FF counters fit in A.cnt");
struct FfS { int tok, n, last, pad; int r[8]; float w[8]; float ss[2][8][4]; };
static_assert(sizeof(FfS) <= 512 && 2 * 512 <= NS * US, "VM_FF scratch in the drained weight ring");
#if VM_FF_NI
__device__ __noinline__   // R-vff: own register allocation (call ABI saves live registers)
#else
__device__ __forceinline__
#endif
void ff_tail(const Args& A, const Tab& T, uint8_t* smem, int tid) {
int* doneB = A.cnt + FF_OFF;
int* claim = doneB + JMAX;
int* exitc = claim + 1;
const int c = blockIdx.x, g0 = T.bB[c], g1 = T.bB[c + 1];
bar_sync(11, 32 * NCW);   // every consumer WG of this CTA has issued its phase-B y writes
if (tid == 0 && g1 > g0) {
__threadfence();
for (int j = g0 >> 8; j <= (g1 - 1) >> 8; ++j) red_release_add(doneB + j, min(g1, 256 * j + 256) - max(g0, 256 * j));
}
const int half = tid >> 8, ht = tid & 255, warp = ht >> 5, lane = tid & 31;
FfS& F = *reinterpret_cast<FfS*>(smem + half * 512);
#if !VM_FF_LW
float4 w4[4];
#pragma unroll
for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * ht));
#endif
const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
for (;;) {
if (ht == 0) F.tok = atomicAdd(claim, 1);
bar_sync(12 + half, 256);
const int t = F.tok;
if (t >= A.M) break;
if (warp == 0) {
const int r = (lane < 8) ? __ldcg(A.rmap + t * 8 + lane) : -1;   // written by this CTA's route() (VM_FF)
const float wv = (lane < 8) ? __ldg(A.wts + t * 8 + lane) : 0.f;
if (r >= 0) {
int lo = 0, hi = T.nj - 1;
while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (T.jS[mid] <= r) lo = mid; else hi = mid - 1; }
wait_flag(doneB + lo, 256, A.err);   // acquire: every B stage of the row's job is in y
}
const unsigned bm = __ballot_sync(0xffffffffu, r >= 0);
if (r >= 0) { const int pos = __popc(bm & ((1u << lane) - 1u)); F.r[pos] = r; F.w[pos] = wv; }
if (lane == 0) F.n = __popc(bm);
__syncwarp();
}
bar_sync(12 + half, 256);
const int nloc = F.n;
float a[4][4];
#pragma unroll
for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
int par = 0;
for (int c0 = 0; c0 < nloc; c0 += FF_CH, par ^= 1) {
float4 v[FF_CH][4];
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj) {
const int r = (c0 + jj < nloc) ? F.r[c0 + jj] : -1;
#pragma unroll
for (int m = 0; m < 4; ++m)
v[jj][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * ht)) : z4;
}
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj)
#pragma unroll
for (int m = 0; m < 4; ++m) {
#if VM_FF_LW
const float4 wm = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * ht));   // L1 hit after the first chunk
#else
const float4 wm = w4[m];
#endif
v[jj][m].x = bf16r(v[jj][m].x) * wm.x; v[jj][m].y = bf16r(v[jj][m].y) * wm.y;
v[jj][m].z = bf16r(v[jj][m].z) * wm.z; v[jj][m].w = bf16r(v[jj][m].w) * wm.w;
}
float ss[FF_CH];
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
f = v[jj][m].x; ss[jj] += f * f; f = v[jj][m].y; ss[jj] += f * f;
f = v[jj][m].z; ss[jj] += f * f; f = v[jj][m].w; ss[jj] += f * f;
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) {
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj) F.ss[par][warp][jj] = ss[jj];
}
bar_sync(12 + half, 256);
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj) {
if (c0 + jj < nloc) {
float sm = 0.f;
for (int w = 0; w < 8; ++w) sm += F.ss[par][w][jj];
const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * F.w[c0 + jj];
#pragma unroll
for (int m = 0; m < 4; ++m) {
a[m][0] += v[jj][m].x * fr; a[m][1] += v[jj][m].y * fr;
a[m][2] += v[jj][m].z * fr; a[m][3] += v[jj][m].w * fr;
}
}
}
#pragma unroll
for (int jj = 0; jj < FF_CH; ++jj) {
const int r = (c0 + jj < nloc) ? F.r[c0 + jj] : -1;
if (r >= 0) {
#pragma unroll
for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * ht), z4);
}
}
}
#pragma unroll
for (int m = 0; m < 4; ++m) {
__nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
uint2 o;
o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * ht) = o;
}
bar_sync(12 + half, 256);   // F is rewritten by the next claim
}
bar_sync(11, 32 * NCW);
FfS& F0 = *reinterpret_cast<FfS*>(smem);
if (tid == 0) {
__threadfence();
const int o = atomicAdd(exitc, 1);
F0.last = (o == (int)gridDim.x - 1);
if (F0.last) __threadfence();
}
bar_sync(11, 32 * NCW);
if (F0.last) {   // every CTA is past all its counter reads: reset for the next launch (vm_fin's job in the ship)
for (int i = tid; i < JMAX; i += 32 * NCW) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 32 * NCW) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTSD * 4; i += 32 * NCW) A.cnt[i] = 0;
for (int i = tid; i < JMAX + 2; i += 32 * NCW) doneB[i] = 0;
}
}
#endif
__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {
"""
OLD_END = "if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();\n}\n"
NEW_END = ("if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();\n"
           "#if VM_FF\nff_tail(A, T, smem, tid);   // R-vff: fused finalize (consumer threads only reach here)\n#endif\n}\n")
OLD_HOST = "if ((mask & 2) && M > 0) {\n"
NEW_HOST = "if ((mask & 2) && M > 0 && !VM_FF) {   // R-vff: vm_main finalizes itself under VM_FF\n"


def apply(s):
    for old, tag, n in ((DEF_A, "defines", 1), (OLD_RMAP, "rmap", 2), (OLD_MAIN, "vm_main", 1), (OLD_END, "vm_main end", 1),
                        (OLD_HOST, "host fin launch", 1)):
        assert s.count(old) == n, f"vff anchor '{tag}' found {s.count(old)} times (want {n})"
    s = s.replace(DEF_A, DEF_N).replace(OLD_RMAP, NEW_RMAP).replace(OLD_MAIN, NEW_MAIN)
    return s.replace(OLD_END, NEW_END).replace(OLD_HOST, NEW_HOST)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vff] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

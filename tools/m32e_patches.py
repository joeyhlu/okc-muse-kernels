"""m32e_patches.py: hot-wide VML ("m32e w"): one weight pass per expert up to VM_JN tokens (notes/study/moe_t32plus.md).

Why: tools/m32e_breakdown.py (step 1) puts ~20 us of VML's T = 256 time in re-reading hot experts that are split into
<= 64-token jobs (every job streams and dequantizes the expert's 28.3 MB again; the public WGS kernel does the same). With
jobs up to VM_JN tokens, one weight pass and one dequant serve all of an expert's tokens; the RS-wgmma consumer runs
m64nNk16 with N up to VM_JN (acc N / 2 registers per thread).

Compile-time switches (the defaults reproduce VML lab v13 exactly; every change is behind them):
  VM_JN  max tokens per job (64)         VM_NTM n-tiles per job = VM_JN / 8 (8)
  VM_NX  x ring slots (3)                VM_XSN x slot capacity in tokens (64): smem = 3 x 36 KB weights + VM_NX x VM_XSN x 512 B
  VM_MKW marks columns per CTA (32; 64 = m32e timeline: A / B stages by n-tile 1..16 in cols 8-23 / 24-39, x-producer
         waits 40-44, %smid + probes 45-47, first weight slot landed 48, A / B segments 49 / 50, split shares 51; vm_fin
         first entry / last exit (gtime) at row G cols 0 / 1)
  Wide build: -DVM_JN=112 -DVM_NX=2 -DVM_XSN=112 (3 x 36,864 + 2 x 57,344 B + tables = 229,296 B of smem).

Applied as anchored text patches (each anchor asserted unique) to
  (1) kernels/lab/vml_moe.py -> kernels/lab/m32e_moe.py (CUDA dedented like the layer copy; lab names);
  (2) a generated layer's `_vml_` block (tools/gen_k3_m32e.py), names prefixed `_vml_`.

    python -I tools/m32e_patches.py lab        # writes kernels/lab/m32e_moe.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB_SRC = ROOT / "kernels/lab/vml_moe.py"
LAB_DST = ROOT / "kernels/lab/m32e_moe.py"

WIDE_DEFS = ("-DVM_JN=112", "-DVM_NX=2", "-DVM_XSN=112")

_CASES = "".join(
    f"#if VM_NTM > {k}\ncase {k}: seg<PB, {8 * k}>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n#endif\n"
    for k in range(8, 16))

# VM_WG5 x gather: warps 17-19 (after the producer warpgroup's setmaxnreg.dec) each take a third of the CTA's rows, 8 loads
# in flight per lane (two passes per 8 KB row), and release their own rows' xready counts
GATHER_W5 = r"""#if VM_WG5
#ifndef VM_GV
#define VM_GV 8
#endif
__device__ void gather_w5(const Args& A, const Tab& T, const uint16_t* ent, int lane, int part) {
const int G = gridDim.x, c = blockIdx.x;
const int q0c = (int)mdiv((uint32_t)(T.Rpad * c), A.magG);
const int q1c = (c == G - 1) ? T.Rpad : (int)mdiv((uint32_t)(T.Rpad * (c + 1)), A.magG);
const int n = q1c - q0c, q0 = q0c + (n * part) / 3, q1 = q0c + (n * (part + 1)) / 3;
const int kk = lane * 8, i = kk >> 6, t = (kk >> 4) & 3, qb = (kk >> 1) & 7;
uint32_t* xa = reinterpret_cast<uint32_t*>(A.xA);
int j = 0;
for (int q = q0; q < q1; ++q) {
while (T.jrow[j + 1] <= q) ++j;
const int r = q - T.jrow[j], cn = T.jC[j];
if (r >= cn) continue;
const int npad = (cn + 7) & ~7;
const uint4* src = reinterpret_cast<const uint4*>(A.x + (size_t)(ent[T.jS[j] + r] >> 3) * KDIM) + lane;
uint32_t* d = xa + T.jrow[j] * 2048 + (((((r >> 3) * 4 + i) * 8 + qb) * 8 + (r & 7)) * 4 + t);
#pragma unroll
for (int h = 0; h < 16 / VM_GV; ++h) {
uint4 v[VM_GV];
#pragma unroll
for (int m = 0; m < VM_GV; ++m) v[m] = __ldg(src + 32 * (VM_GV * h + m));
#pragma unroll
for (int m = 0; m < VM_GV; ++m) {
uint32_t* dm = d + (VM_GV * h + m) * npad * 128;
dm[0] = v[m].x; dm[32] = v[m].y; dm[64] = v[m].z; dm[96] = v[m].w;
}
}
}
fence_proxy_async();
__syncwarp();
if (lane == 0) {
int jj = 0;
for (int q = q0; q < q1;) {
while (T.jrow[jj + 1] <= q) ++jj;
const int e = min(q1, T.jrow[jj + 1]);
red_release_add(A.xready + jj, e - q);
q = e;
}
}
}
#endif
"""

# (old, new) pairs on the dedented CUDA text (no comment-only lines: the layer copy has them stripped)
CUDA_PATCHES = [
    ("constexpr int NS = 3, NX = 3;\nconstexpr int XS = 64 * 512;\n",
     "#ifndef VM_JN\n#define VM_JN 64\n#endif\n#ifndef VM_NTM\n#define VM_NTM (VM_JN / 8)\n#endif\n#ifndef VM_NX\n#define VM_NX 3\n#endif\n"
     "#ifndef VM_XSN\n#define VM_XSN 64\n#endif\n#ifndef VM_MKW\n#define VM_MKW 32\n#endif\n"
     "static_assert(VM_NTM * 8 == VM_JN && VM_JN <= VM_XSN && VM_NTM <= 16 && (VM_MKW == 32 || VM_MKW == 64), \"m32e config\");\n"
     "constexpr int NS = 3, NX = VM_NX;\nconstexpr int XS = VM_XSN * 512;\n"
     "constexpr int MKW = VM_MKW, MKB = (VM_MKW == 64) ? 24 : 16, MKX = (VM_MKW == 64) ? 40 : 24, MKP = (VM_MKW == 64) ? 45 : 29;\n"),
    ("FIXW = @FIXW@;\n", "FIXW = (@FIXW@ > VM_NTM * 512) ? @FIXW@ : VM_NTM * 512;\n"),
    ("uint32_t cost[16]; unsigned long long mag[16];\n",
     "uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];\n#if VM_MKW == 64\nint G0;\n#endif\n"),
    ("uint32_t st = mdiv(x - cp[lo], A.mag[8 * pb + nt - 1]);\n", "uint32_t st = mdiv(x - cp[lo], A.mag[VM_NTM * pb + nt - 1]);\n"),
    ("const int nj = (tot + 63) >> 6;\nint base = 64;\nif (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }\n",
     "#if VM_JN == 64 && !VM_DP\nconst int nj = (tot + 63) >> 6;\nint base = 64;\nif (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }\n"
     "#else\nconst int nj = (tot + VM_JN - 1) / VM_JN;\nconst int base = nj ? (tot + nj - 1) / nj : VM_JN;\n#endif\n"),
    ("(int)A.cost[8 + nt] : 0;\n", "(int)A.cost[VM_NTM + nt] : 0;\n"),
    ("long long* m = A.marks + (size_t)c * 32;\nm[24] = wB; m[25] = nB; m[26] = wA; m[27] = tB0; m[28] = tB1;\n",
     "long long* m = A.marks + (size_t)c * MKW;\nm[MKX] = wB; m[MKX + 1] = nB; m[MKX + 2] = wA; m[MKX + 3] = tB0; m[MKX + 4] = tB1;\n"),
    ("if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;\n",
     "#if VM_MKW == 64\nif (A.marks && q == 0 && wp == 0 && lane == 0) {\nlong long* mk = A.marks + (size_t)c * MKW;\n"
     "mk[8 + 16 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;\nmk[49 + PB] += 1;\nif (!(s0 == 0 && s1 == 16)) mk[51] += 1;\n}\n#else\n"
     "if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * 32 + 8 + 8 * PB + ((T.jC[j] + 7) >> 3) - 1] += s1 - s0;\n#endif\n"),
    ("case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n"
     "default: seg<PB, 64>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n",
     "case 7: seg<PB, 56>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n" + _CASES +
     "default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n"),
    ("long long* m = A.marks + (size_t)blockIdx.x * 32;\nunsigned smid;\n", "long long* m = A.marks + (size_t)blockIdx.x * MKW;\nunsigned smid;\n"),
    ("m[29] = smid;\nm[30] = lat_probe(A.ws13, A.flags);\nm[31] = lat_probe(A.ws13 + 4096, A.flags);\n",
     "m[MKP] = smid;\nm[MKP + 1] = lat_probe(A.ws13, A.flags);\nm[MKP + 2] = lat_probe(A.ws13 + 4096, A.flags);\n"),
    ("if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 7] = gtime();\n",
     "if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 7] = gtime();\n"),
    ("if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 5] = gtime();\n",
     "if (tid == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 5] = gtime();\n"
     "#if VM_MKW == 64\nif (tid == 0 && A.marks && (T.bA[blockIdx.x + 1] - T.bA[blockIdx.x]) + (T.bB[blockIdx.x + 1] - T.bB[blockIdx.x]) > 0) {\n"
     "mbar_wait(bars, 0, A.err);\nA.marks[(size_t)blockIdx.x * MKW + 48] = gtime();\n}\n#endif\n"),
    ("if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * 32 + 6] = gtime();\n",
     "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n"),
    ("if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 1 + q] = gtime();\n",
     "if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * MKW + 1 + q] = gtime();\n"),
    (r'''for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));
asm volatile("griddepcontrol.wait;\n" ::: "memory");
''',
     r'''for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * tid));
asm volatile("griddepcontrol.wait;\n" ::: "memory");
#if VM_MKW == 64
if (tid == 0 && A.marks) atomicMin(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW), (unsigned long long)gtime());
#endif
'''),
    ("*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * tid) = o;\n}\n}\n",
     "*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * tid) = o;\n}\n#if VM_MKW == 64\n__syncthreads();\n"
     "if (tid == 0 && A.marks) atomicMax(reinterpret_cast<unsigned long long*>(A.marks + (size_t)A.G0 * MKW + 1), (unsigned long long)gtime());\n"
     "#endif\n}\n"),
    ("a.marks = T[18].numel() >= G * 32 ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;\n",
     "a.marks = T[18].numel() >= (G + (vm::MKW == 64 ? 1 : 0)) * vm::MKW ? reinterpret_cast<long long*>(T[18].data_ptr()) : nullptr;\n"
     "#if VM_MKW == 64\na.G0 = (int)G;\n#endif\nTORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, \"vml_run: cost table too short\");\n"),
    ("for (int k = 0; k < 16; ++k) { a.cost[k] =", "for (int k = 0; k < 2 * VM_NTM; ++k) { a.cost[k] ="),
    ("C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n}\n}\n",
     "C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, vm::vm_fin, a));\n}\n}\n"
     "std::vector<int64_t> vml_cfg() { return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM}; }\n"),
    # VM_WG5: a fifth (producer) warpgroup so setmaxnreg can move registers from the producers (VM_PREG) to the consumer
    # warpgroups (VM_CREG): at 576 threads ptxas caps every thread at 96 (5 warps on an SMSP), and N > 64 serializes wgmma
    ("constexpr int NTH = 32 * NCW + 64;\n",
     "#ifndef VM_WG5\n#define VM_WG5 0\n#endif\n#ifndef VM_PREG\n#define VM_PREG 56\n#endif\n#ifndef VM_CREG\n#define VM_CREG 112\n#endif\n"
     "#define VM_S2(x) #x\n#define VM_S(x) VM_S2(x)\nconstexpr int NTH = 32 * NCW + (VM_WG5 ? 128 : 64);\n"),
    ("if (warp >= NCW) {\nif (warp == NCW) {\n",
     "if (warp >= NCW) {\n#if VM_WG5\nasm volatile(\"setmaxnreg.dec.sync.aligned.u32 \" VM_S(VM_PREG) \";\\n\" ::: \"memory\");\n"
     "if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); return; }\n#endif\nif (warp == NCW) {\n"),
    ("} else {\nif (A.flags & 64) gather_warp(A, T, ent, lane);\n",
     "} else {\n#if VM_WG5\nif (A.flags & 64) gather_w5(A, T, ent, lane, 0);\n#else\nif (A.flags & 64) gather_warp(A, T, ent, lane);\n#endif\n"),
    ("__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {\n",
     GATHER_W5 + "__device__ __forceinline__ int nfill_of(const Args& A, const Tab& T) {\n"),
    ("bar_arrive(6, 32 * NCW + 32);\nif (A.flags & 4) bar_arrive(7, 32 * NCW + 32);\n",
     "bar_arrive(6, 32 * NCW + 32);\nif (A.flags & 4) bar_arrive(7, 32 * NCW + 32);\n"
     "#if VM_WG5\nasm volatile(\"setmaxnreg.inc.sync.aligned.u32 \" VM_S(VM_CREG) \";\\n\" ::: \"memory\");\n#endif\n"),
]

# ---- VM_DP: dynamic tail pools (notes/study/moe_t32plus.md s3). Jobs sorted hot-first (flags bit 13), the static stream-K
# plan covers stages [0, sA) / [0, sB); the last A.poolA / A.poolB 256ths of each phase's units (cold units, whole units at
# the enumeration end) are VM_DPI-stage items claimed by the weight producer after its static range (atomic counter
# A.gbar[pb]) and published to the x producer and the consumers through a 16-entry smem queue (T.q / T.qpub, -1 ends a
# phase). Pool units reduce through their own fixup slots (G + unit offset). vm_fin zeroes the claim counters.
_DP_SWITCH = ("switch ((T.jC[j] + 7) >> 3) {\n"
              + "".join(f"case {k}: seg<PB, {8 * k}>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n" for k in range(1, 8))
              + _CASES + "default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n}\n")
_DP_QWAIT = r"""if (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
const long long t0 = clock64();
while (*reinterpret_cast<const volatile int*>(&T.qpub) <= kq) {
__nanosleep(32);
if (*reinterpret_cast<volatile int*>(A.err) != 0) break;
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, @E@); break; }
}
}
__threadfence_block();
const int v = T.q[kq & 15];
++kq;
if (v < 0 || *reinterpret_cast<volatile int*>(A.err) != 0) break;
"""
DP_PATCHES = [
    ("FIXW = (@FIXW@ > VM_NTM * 512) ? @FIXW@ : VM_NTM * 512;\n",
     "FIXW = (@FIXW@ > VM_NTM * 512) ? @FIXW@ : VM_NTM * 512;\n#ifndef VM_DP\n#define VM_DP 0\n#endif\n#ifndef VM_DPI\n#define VM_DPI 4\n#endif\n"
     "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n"),
    ("unsigned long long bperm[JMAX];",
     "#if VM_DP\nint sA, sB, qpub, qpad;\nuint32_t cA, cB;\nint q[16];\n#endif\nunsigned long long bperm[JMAX];"),
    ("uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];\n",
     "uint32_t cost[2 * VM_NTM]; unsigned long long mag[2 * VM_NTM];\n#if VM_DP\nint poolA, poolB;\n#endif\n"),
    ("if (c >= G) return nj << (9 - pb);\nconst uint32_t* cp = pb ? T.cpB : T.cpA;\nconst uint32_t x = mdiv(cp[nj] * (uint32_t)c, magG);\n",
     "#if VM_DP\nif (c >= G) return pb ? T.sB : T.sA;\nconst uint32_t* cp = pb ? T.cpB : T.cpA;\nconst uint32_t x = mdiv((pb ? T.cB : T.cA) * (uint32_t)c, magG);\n"
     "#else\nif (c >= G) return nj << (9 - pb);\nconst uint32_t* cp = pb ? T.cpB : T.cpA;\nconst uint32_t x = mdiv(cp[nj] * (uint32_t)c, magG);\n#endif\n"),
    ("const int jst = excl_scan(nj, lane), rst = excl_scan(prow, lane);\n",
     r"""#if VM_DP
int rank = lane;
if (A.flags & 8192) { rank = 0; for (int k = 0; k < 32; ++k) { const int t2 = __shfl_sync(0xffffffffu, tot, k); rank += (t2 > tot) || (t2 == tot && k < lane); } }
fin[rank] = (uint32_t)nj; fin[32 + rank] = (uint32_t)prow;
__syncwarp();
const int jsr = excl_scan((int)fin[lane], lane), rsr = excl_scan((int)fin[32 + lane], lane);
fin[64 + lane] = (uint32_t)jsr; fin[96 + lane] = (uint32_t)rsr;
__syncwarp();
const int jst = (int)fin[64 + rank], rst = (int)fin[96 + rank];
#else
const int jst = excl_scan(nj, lane), rst = excl_scan(prow, lane);
#endif
"""),
    ("if (lane == 31) {\nT.nj = jst + nj; T.R = est + tot; T.Rpad = rst + prow;\nT.jS[jst + nj] = est + tot; T.jrow[jst + nj] = rst + prow;\n}\n",
     r"""#if VM_DP
{
const int njT = __reduce_add_sync(0xffffffffu, nj), RT = __reduce_add_sync(0xffffffffu, tot), RpT = __reduce_add_sync(0xffffffffu, prow);
if (lane == 0) { T.nj = njT; T.R = RT; T.Rpad = RpT; T.jS[njT] = RT; T.jrow[njT] = RpT; }
}
#else
if (lane == 31) {
T.nj = jst + nj; T.R = est + tot; T.Rpad = rst + prow;
T.jS[jst + nj] = est + tot; T.jrow[jst + nj] = rst + prow;
}
#endif
"""),
    ("if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }\n",
     r"""if (lane == 0) { T.cpA[T.nj] = accA; T.cpB[T.nj] = accB; }
#if VM_DP
if (lane == 0) {
const int SA = T.nj << 9, SB = T.nj << 8;
const int pa = min(DPMAX, (int)(((long long)(SA >> 4) * A.poolA) >> 8)), pq = min(DPMAX, (int)(((long long)(SB >> 4) * A.poolB) >> 8));
T.sA = SA - (pa << 4); T.sB = SB - (pq << 4);
const int ja = T.sA >> 9, jb = T.sB >> 8;
T.cA = (ja < T.nj) ? T.cpA[ja] + (uint32_t)(T.sA - (ja << 9)) * A.cost[((T.jC[ja] + 7) >> 3) - 1] : T.cpA[T.nj];
T.cB = (jb < T.nj) ? T.cpB[jb] + (uint32_t)(T.sB - (jb << 8)) * A.cost[VM_NTM + ((T.jC[jb] + 7) >> 3) - 1] : T.cpB[T.nj];
T.qpub = 0;
}
#endif
"""),
    ("bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);\n}\n}\n}\n",
     r"""bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
#if VM_DP
{
Tab& TT = const_cast<Tab&>(T);
const int S0 = pb ? T.sB : T.sA, nit = ((pb ? (T.nj << 8) : (T.nj << 9)) - S0) / VM_DPI;
for (;;) {
const int i0 = (nit > 0) ? atomicAdd(A.gbar + pb, 1) : 0;
const int v = (i0 < nit) ? i0 : -1;
const int k = TT.qpub;
TT.q[k & 15] = v;
__threadfence_block();
*reinterpret_cast<volatile int*>(&TT.qpub) = k + 1;
if (v < 0) break;
for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5), ub = pb ? (u & 15) : (u & 31);
const int s = gs & 15;
const int e = T.jE[j];
const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW
: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;
const int sl = it % NS;
if (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * sl, dst = ring + sl * US;
mbar_expect_tx(fb, US);
bulk_g2s_ef(dst, src, SW, fb, pol);
bulk_g2s_ef(dst + SW, src + (size_t)16 * SW, SW, fb, pol);
}
}
}
#endif
}
}
"""),
    ("int nB = 0;\n", "int nB = 0;\n#if VM_DP\nint kq = 0;\n#endif\n"),
    ("mbar_expect_tx(fb, bytes);\nbulk_g2s(xring + sl * XS, src, bytes, fb);\n}\n}\n",
     r"""mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
#if VM_DP
{
const int S0 = pb ? T.sB : T.sA;
for (;;) {
""" + _DP_QWAIT.replace("@E@", "4") + r"""for (int s2 = 0; s2 < VM_DPI; ++s2, ++it) {
const int gs = S0 + v * VM_DPI + s2;
const int u = gs >> 4;
const int j = pb ? (u >> 4) : (u >> 5);
const int s = gs & 15;
const int npad = (T.jC[j] + 7) & ~7;
if (!pb) wait_flag(A.xready + j, npad, A.err);
else wait_flag(A.doneA + j * 16 + s, 8, A.err);
fence_proxy_async();
const uint32_t bytes = (uint32_t)npad * 512u;
const __nv_bfloat16* src = (pb ? A.xB : A.xA) + (size_t)T.jrow[j] * KDIM + (size_t)s * npad * 256;
const int sl = it % NX;
if (it >= NX) mbar_wait(bars + 8 * (2 * NS + NX + sl), ((it / NX) & 1) ^ 1, A.err);
const uint32_t fb = bars + 8 * (2 * NS + sl);
mbar_expect_tx(fb, bytes);
bulk_g2s(xring + sl * XS, src, bytes, fb);
}
}
}
#endif
}
"""),
    ("const int slot = cta_of(T, PB, u << 4, gridDim.x);\nfloat* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;\n"
     "int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;\n",
     r"""#if VM_DP
const int s0u = (PB ? T.sB : T.sA) >> 4;
const int slot = (u >= s0u) ? (int)gridDim.x + (u - s0u) : cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTSD + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTSD + slot) * 4 + q;
#else
const int slot = cta_of(T, PB, u << 4, gridDim.x);
float* fb = A.fix + ((size_t)((PB * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + (PB * SLOTS + slot) * 4 + q;
#endif
"""),
    ("__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {\n",
     "#if VM_DP\n__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane, int& kq) {\n"
     "#else\n__device__ __forceinline__ void phase(const Args& A, Tab& T, uint8_t* smem, uint32_t bars, int& it, int q, int wp, int lane) {\n#endif\n"),
    # one loop body for static segments and pool items (a second switch would duplicate every seg<> instantiation)
    ("#pragma unroll 1\nfor (int gs = sp, ge = g1; ;) {\nif (gs >= ge) {\nif (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;\n}\n"
     "const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));\n",
     r"""#if VM_DP
bool pool = false;
const int S0 = PB ? T.sB : T.sA;
#endif
#pragma unroll 1
for (int gs = sp, ge = g1; ;) {
#if VM_DP
int u, s0, s1;
if (!pool && gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; }
else {
pool = true;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 54 + PB] = gtime();
#endif
}
}
if (pool) {
""" + _DP_QWAIT.replace("@E@", "5") + r"""const int g2 = S0 + v * VM_DPI;
u = g2 >> 4; s0 = g2 & 15; s1 = s0 + VM_DPI;
#if VM_MKW == 64
if (A.marks && q == 0 && wp == 0 && lane == 0) A.marks[(size_t)c * MKW + 52 + PB] += 1;
#endif
} else {
u = gs >> 4; s0 = gs & 15; s1 = min(16, s0 + (ge - gs));
}
#else
if (gs >= ge) {
if (ge == g1 && sp > g0) { gs = g0; ge = sp; } else break;
}
const int u = gs >> 4, s0 = gs & 15, s1 = min(16, s0 + (ge - gs));
#endif
"""),
    ("int it = 0;\nphase<false>(A, T, smem, bars, it, q, wp, lane);\n",
     "int it = 0;\n#if VM_DP\nint kq = 0;\nphase<false>(A, T, smem, bars, it, q, wp, lane, kq);\n#else\nphase<false>(A, T, smem, bars, it, q, wp, lane);\n#endif\n"),
    ("phase<true>(A, T, smem, bars, it, q, wp, lane);\n",
     "#if VM_DP\nphase<true>(A, T, smem, bars, it, q, wp, lane, kq);\n#else\nphase<true>(A, T, smem, bars, it, q, wp, lane);\n#endif\n"),
    ("for (int i = tid; i < 2 * SLOTS * 4; i += 256) A.cnt[i] = 0;\n",
     "for (int i = tid; i < 2 * SLOTSD * 4; i += 256) A.cnt[i] = 0;\n#if VM_DP\nif (tid == 0) { A.gbar[0] = 0; A.gbar[1] = 0; }\n#endif\n"),
    ("TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, \"vml_run: cost table too short\");\n",
     "TORCH_CHECK((int64_t)V.size() >= 6 + 2 * VM_NTM, \"vml_run: cost table too short\");\n#if VM_DP\n"
     "TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, \"vml_run: pool fractions missing\");\n"
     "a.poolA = (int)V[6 + 2 * VM_NTM]; a.poolB = (int)V[7 + 2 * VM_NTM];\n#endif\n"),
    ("return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM}; }",
     "return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP}; }"),
]
# ---- VM_SC: segment-aware static plan (R-m32e-1 regression: ~1 us per segment, i.e. per unit boundary inside a CTA's
# range). The cost prefix charges A.segA / A.segB (cost units) per unit start, stage_at inverts per unit; flags bit 14 snaps a
# CTA cut to the nearest unit edge when it moves by <= segX's worth of stages (fewer split shares).
SC_PATCHES = [
    ("constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n",
     "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n#ifndef VM_SC\n#define VM_SC 0\n#endif\n"
     "static_assert(!(VM_SC && VM_DP), \"m32e: VM_SC and VM_DP are separate plans\");\n"),
    ("#if VM_DP\nint poolA, poolB;\n#endif\n", "#if VM_DP\nint poolA, poolB;\n#endif\n#if VM_SC\nint segA, segB;\n#endif\n"),
    ("uint32_t st = mdiv(x - cp[lo], A.mag[VM_NTM * pb + nt - 1]);\n",
     r"""#if VM_SC
const uint32_t xr = x - cp[lo], cs = A.cost[VM_NTM * pb + nt - 1], sg = (uint32_t)(pb ? A.segB : A.segA);
const uint32_t uc = 16u * cs + sg, uu = xr / uc, rr = xr - uu * uc;
uint32_t st = 16u * uu + (rr > sg ? (rr - sg) / cs : 0u);
#else
uint32_t st = mdiv(x - cp[lo], A.mag[VM_NTM * pb + nt - 1]);
#endif
"""),
    ("if (st > ns - 1) st = ns - 1;\n",
     r"""if (st > ns - 1) st = ns - 1;
#if VM_SC
if (A.flags & 16384) {
const uint32_t ks = (pb ? (uint32_t)A.segB : (uint32_t)A.segA) / A.cost[VM_NTM * pb + nt - 1], w = st & 15u;
if (w <= ks) st -= w;
else if (16u - w <= ks) st += 16u - w;
}
#endif
"""),
    ("const int sc = excl_scan(ca, lane), sd = excl_scan(cb, lane);\n"
     "if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }\n"
     "accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);\n"
     "accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);\n",
     r"""#if VM_SC
const int ja = (j < T.nj) ? 512 * ca + 32 * A.segA : 0, jb = (j < T.nj) ? 256 * cb + 16 * A.segB : 0;
const int sc = excl_scan(ja, lane), sd = excl_scan(jb, lane);
if (j < T.nj) { T.cpA[j] = accA + (uint32_t)sc; T.cpB[j] = accB + (uint32_t)sd; }
accA += (uint32_t)__shfl_sync(0xffffffffu, sc + ja, 31);
accB += (uint32_t)__shfl_sync(0xffffffffu, sd + jb, 31);
#else
const int sc = excl_scan(ca, lane), sd = excl_scan(cb, lane);
if (j < T.nj) { T.cpA[j] = accA + 512u * (uint32_t)sc; T.cpB[j] = accB + 256u * (uint32_t)sd; }
accA += 512u * (uint32_t)__shfl_sync(0xffffffffu, sc + ca, 31);
accB += 256u * (uint32_t)__shfl_sync(0xffffffffu, sd + cb, 31);
#endif
"""),
    ("a.poolA = (int)V[6 + 2 * VM_NTM]; a.poolB = (int)V[7 + 2 * VM_NTM];\n#endif\n",
     "a.poolA = (int)V[6 + 2 * VM_NTM]; a.poolB = (int)V[7 + 2 * VM_NTM];\n#endif\n#if VM_SC\n"
     "TORCH_CHECK((int64_t)V.size() >= 8 + 2 * VM_NTM, \"vml_run: segment costs missing\");\n"
     "a.segA = (int)V[6 + 2 * VM_NTM]; a.segB = (int)V[7 + 2 * VM_NTM];\n#endif\n"),
    ("(int64_t)vm::SMEM, VM_DP}; }", "(int64_t)vm::SMEM, VM_DP, VM_SC}; }"),
]
# ---- cheaper segments (R-m32e-4). VM_BRED: a phase-B split share red.adds its partial (x gs2) straight into the y rows
# (no fixup slot, no count atomic, no bar_sync wait, no read-back); vm_fin zeroes every y row it read, so y is all-zero
# between calls. VM_DEFER2: a whole phase-A unit's proxy fence + WG barrier + doneA release are deferred to the WG's NEXT
# segment end (after its wgmma drain, before its own stores, so the fence finds nothing outstanding) and to the phase end.
_BRED_FN = r"""#if VM_BRED
template <int N>
__device__ __forceinline__ void bred(const Args& A, const Tab& T, int j, int ub, int q, int wp, int lane, const float* d) {
const int g = lane >> 2, t = lane & 3;
const int e = T.jE[j], cn = T.jC[j];
const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;
const float gs = __ldg(A.gsc2 + e);
const int h0 = 16 * tg + g;
float* yb = A.y + (size_t)T.jS[j] * KDIM;
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt)
#pragma unroll
for (int hh = 0; hh < 2; ++hh) {
const int n = 8 * nt + 2 * t + hh;
if (n < cn) {
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0), "f"(d[4 * nt + hh] * gs) : "memory");
asm volatile("red.global.add.f32 [%0], %1;\n" :: "l"(yb + (size_t)n * KDIM + h0 + 8), "f"(d[4 * nt + 2 + hh] * gs) : "memory");
}
}
}
#endif
"""
CS_PATCHES = [
    ("constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n",
     "constexpr int DPMAX = 384, SLOTSD = SLOTS + (VM_DP ? DPMAX : 0);\n#ifndef VM_BRED\n#define VM_BRED 0\n#endif\n"
     "#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n"
     "static_assert(!(VM_BRED && VM_DP), \"m32e: VM_BRED and VM_DP use different split paths\");\n"),
    ("if constexpr (VM_DEFER) {\npd.i = j * 16 + (ub >> 1);\n", "if constexpr (VM_DEFER || VM_DEFER2) {\npd.i = j * 16 + (ub >> 1);\n"),
    ("if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n}\nconstexpr int UBL = PB ? 4 : 5;\n"
     "const int j = u >> UBL, ub = u & ((1 << UBL) - 1);\nif (!(s0 == 0 && s1 == 16)) {\n",
     "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));\n}\n"
     "if constexpr (VM_DEFER2) pend_flush(A, pd, q, wp, lane);\n"
     "constexpr int UBL = PB ? 4 : 5;\nconst int j = u >> UBL, ub = u & ((1 << UBL) - 1);\nif (!(s0 == 0 && s1 == 16)) {\n"
     "#if VM_BRED\nif constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); return; }\n#endif\n"),
    ("if constexpr (VM_DEFER && !PB) pend_flush(A, pd, q, wp, lane);",
     "if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);"),
    ("template <bool PB, int N>\n__device__ __forceinline__ void seg(",
     _BRED_FN + "template <bool PB, int N>\n__device__ __forceinline__ void seg("),
    ("a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;\n}\n}\n}\n",
     r"""a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;
}
}
}
#if VM_BRED
#pragma unroll
for (int jj = 0; jj < 4; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
if (r >= 0) {
#pragma unroll
for (int m = 0; m < 4; ++m) __stcg(reinterpret_cast<float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * tid), z4);
}
}
#endif
"""),
    ("(int64_t)vm::SMEM, VM_DP, VM_SC}; }", "(int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2}; }"),
]
# ---- R-m32e-6: VM_PS (staggered first weight copies: the producer issues ring slot it < NS at tps + A.pdel x it ns, so
# every CTA's first stage lands before the whole 14 MB start burst; marks showed the first slot landing 3.3-3.6 us after the
# consumers are ready at T = 32 / 128) and, in VM_MKW=64 builds, per-segment timing marks: cols 56..61 = sum over WG 0's
# segments of first-stage time / rest of the k-loop / wgmma drain / fixup + epilogue, segment count, stages after the first.
_SEGMK = r"""#if VM_MKW == 64
struct SegMk { long long t0, t1, t2, t3; bool on; };
__device__ __forceinline__ void segmk_done(const Args& A, SegMk& m, int s0, int s1) {
if (m.on) {
const long long t4 = gtime();
long long* mk = A.marks + (size_t)blockIdx.x * MKW;
mk[56] += m.t1 - m.t0; mk[57] += m.t2 - m.t1; mk[58] += m.t3 - m.t2; mk[59] += t4 - m.t3; mk[60] += 1; mk[61] += s1 - s0 - 1;
}
}
#endif
"""
PS_PATCHES = [
    ("#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n",
     "#ifndef VM_DEFER2\n#define VM_DEFER2 0\n#endif\n#ifndef VM_PS\n#define VM_PS 0\n#endif\n"),
    ("#if VM_SC\nint segA, segB;\n#endif\n", "#if VM_SC\nint segA, segB;\n#endif\n#if VM_PS\nint pdel;\n#endif\n"),
    ("__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n",
     "__device__ void prod_w(const Args& A, const Tab& T, uint32_t ring, uint32_t bars) {\nconst uint64_t pol = policy_evict_first();\n"
     "#if VM_PS\nconst long long tps = gtime();\n#endif\n"),
    ("const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);\nconst int e = T.jE[j];\n"
     "const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW\n: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;\n"
     "const int sl = it % NS;\nif (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);\n",
     "const int s = pb ? bstage(A, T, j, gs & 15) : (gs & 15);\nconst int e = T.jE[j];\n"
     "const uint8_t* src = pb ? A.ws2 + ((size_t)(e * 32 + 2 * ub) * 16 + s) * SW\n: A.ws13 + ((size_t)(e * 64 + 2 * ub) * 16 + s) * SW;\n"
     "const int sl = it % NS;\nif (it >= NS) mbar_wait(bars + 8 * (NS + sl), ((it / NS) & 1) ^ 1, A.err);\n"
     "#if VM_PS\nelse if (it > 0 && A.pdel > 0) { const long long tw = tps + (long long)A.pdel * it; while (gtime() < tw) __nanosleep(32); }\n#endif\n"),
    ("a.segA = (int)V[6 + 2 * VM_NTM]; a.segB = (int)V[7 + 2 * VM_NTM];\n#endif\n",
     "a.segA = (int)V[6 + 2 * VM_NTM]; a.segB = (int)V[7 + 2 * VM_NTM];\n#endif\n#if VM_PS\n"
     "TORCH_CHECK((int64_t)V.size() >= 7 + 2 * VM_NTM, \"vml_run: producer delay missing\");\na.pdel = (int)V[6 + 2 * VM_NTM];\n#endif\n"),
    ("(int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2}; }", "(int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS}; }"),
    ("template <bool PB, int N>\n__device__ __forceinline__ void seg(", _SEGMK + "template <bool PB, int N>\n__device__ __forceinline__ void seg("),
    ("int q, int wp, int lane, Pend& pd) {\nfloat d[N / 2];\n",
     "int q, int wp, int lane, Pend& pd) {\nfloat d[N / 2];\n"
     "SEGMK(SegMk smk{0, 0, 0, 0, A.marks != nullptr && q == 0 && wp == 0 && lane == 0}; if (smk.on) smk.t0 = gtime();)\n"),
    ("stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\nif constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);\n",
     "stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\nSEGMK(if (smk.on) smk.t1 = gtime();)\nif constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);\n"),
    ("for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\n",
     "for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane);\nSEGMK(if (smk.on) smk.t2 = smk.t3 = gtime();)\n"),
    ("stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);\n",
     "stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, true, pd);\nSEGMK(if (smk.on) smk.t1 = gtime();)\n"),
    ("for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);\nwg_wait<0>();\n",
     "for (int s = s0 + 1; s < s1; ++s) stage<N>(A, smem, bars, it, prevx, d, hb, q, wp, lane, false, pd);\n"
     "SEGMK(if (smk.on) smk.t2 = gtime();)\nwg_wait<0>();\nSEGMK(if (smk.on) smk.t3 = gtime();)\n"),
    ("if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); return; }",
     "if constexpr (PB) { bred<N>(A, T, j, ub, q, wp, lane, d); SEGMK(segmk_done(A, smk, s0, s1);) return; }"),
    ("if (!T.wlast[q]) return;\n", "if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }\n"),
    ("epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\n}\ntemplate <bool PB>\n",
     "epilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\nSEGMK(segmk_done(A, smk, s0, s1);)\n}\ntemplate <bool PB>\n"),
]
# ---- R-m32e-7: VM_AD (runtime flags bit 15): a phase-A split share red.adds its partial and returns; its count atomic,
# last-arriver read-back and epilogue run one segment later (m32e_resolve, after the next segment, or at the next split /
# the phase end), when its red.adds have long completed (R-m32e-6 marks: ~2 us per split share, wgmma drain ~0). MKW=64
# builds also time the split-share atomic step (col 62) and the A epilogue's proxy fence + barrier + release (col 63).
_AD_FN = r"""#if VM_AD
template <int N>
__device__ __forceinline__ void m32e_finish(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
float* fb = A.fix + ((size_t)(pd.sslot * 4 + q)) * FIXW + wt * 4;
int* fc = A.cnt + pd.sslot * 4 + q;
float d[N / 2];
fence_acq_rel();
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
if (wt == 0) *fc = 0;
epilogue<false, N>(A, T, pd.su >> 5, pd.su & 31, q, wp, lane, d, pd);
}
__device__ __noinline__ void m32e_resolve(const Args& A, Tab& T, Pend& pd, int q, int wp, int lane) {
const int wt = (wp << 5) | lane;
int* fc = A.cnt + pd.sslot * 4 + q;
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, pd.scnt) + pd.scnt == 16);
bar_sync(1 + q, 128);
pd.son = 0;
if (!T.wlast[q]) return;
switch (pd.snt) {
case 1: m32e_finish<8>(A, T, pd, q, wp, lane); break;
case 2: m32e_finish<16>(A, T, pd, q, wp, lane); break;
case 3: m32e_finish<24>(A, T, pd, q, wp, lane); break;
case 4: m32e_finish<32>(A, T, pd, q, wp, lane); break;
case 5: m32e_finish<40>(A, T, pd, q, wp, lane); break;
case 6: m32e_finish<48>(A, T, pd, q, wp, lane); break;
case 7: m32e_finish<56>(A, T, pd, q, wp, lane); break;
default: m32e_finish<64>(A, T, pd, q, wp, lane); break;
}
}
#endif
"""
AD_PATCHES = [
    ("#ifndef VM_PS\n#define VM_PS 0\n#endif\n", "#ifndef VM_PS\n#define VM_PS 0\n#endif\n#ifndef VM_AD\n#define VM_AD 0\n#endif\n"
     "#if VM_MKW == 64\n#define SEGMK(...) __VA_ARGS__\n#else\n#define SEGMK(...)\n#endif\n"),
    ("struct Pend { int i; };", "#if VM_AD\nstruct Pend { int i; int su, sslot, snt, scnt, son, stag; };\n#else\nstruct Pend { int i; };\n#endif\n//"),
    ("template <bool PB, int N>\n__device__ __forceinline__ void seg(", _AD_FN + "template <bool PB, int N>\n__device__ __forceinline__ void seg("),
    ("for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);\n"
     "bar_sync(1 + q, 128);\nif (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);\nbar_sync(1 + q, 128);\n",
     r"""for (int nt = 0; nt < N / 8; ++nt) red_add_v4(fb + nt * 512, d[4 * nt], d[4 * nt + 1], d[4 * nt + 2], d[4 * nt + 3]);
#if VM_AD
if constexpr (!PB) {
if (A.flags & 32768) {
if (pd.son) m32e_resolve(A, T, pd, q, wp, lane);
pd.son = 1; pd.su = u; pd.sslot = slot; pd.snt = N / 8; pd.scnt = s1 - s0; ++pd.stag;
SEGMK(segmk_done(A, smk, s0, s1);)
return;
}
}
#endif
SEGMK(long long t_at = smk.on ? gtime() : 0;)
bar_sync(1 + q, 128);
if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);
bar_sync(1 + q, 128);
SEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 62] += gtime() - t_at;)
"""),
    ("} else {\nfence_proxy_async();\nbar_sync(1 + q, 128);\nif (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);\n}\n",
     "} else {\nSEGMK(const long long t_f = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)\nfence_proxy_async();\nbar_sync(1 + q, 128);\n"
     "if (wt == 0) red_release_add(A.doneA + j * 16 + (ub >> 1), 1);\n"
     "SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 63] += gtime() - t_f;)\n}\n"),
    ("switch ((T.jC[j] + 7) >> 3) {\ncase 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n",
     "#if VM_AD\nconst int tag0 = pd.stag;\n#endif\n"
     "switch ((T.jC[j] + 7) >> 3) {\ncase 1: seg<PB, 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n"),
    ("default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n}\ngs += s1 - s0;\n",
     "default: seg<PB, VM_NTM * 8>(A, T, smem, bars, it, u, s0, s1, q, wp, lane, pd); break;\n}\n"
     "#if VM_AD\nif constexpr (!PB) { if (pd.son && pd.stag == tag0) m32e_resolve(A, T, pd, q, wp, lane); }\n#endif\ngs += s1 - s0;\n"),
    ("if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);",
     "#if VM_AD\nif constexpr (!PB) { if (pd.son) m32e_resolve(A, T, pd, q, wp, lane); }\n#endif\n"
     "if constexpr ((VM_DEFER || VM_DEFER2) && !PB) pend_flush(A, pd, q, wp, lane);"),
]
# ---- R-m32e-8: VM_GSM (per-expert global scales copied to smem in route(); the epilogues read smem instead of a dependent
# __ldg) and VM_FSW (SwiGLU with __fdividef; FP32 intermediates, same bf16 output rounding). MKW=64 builds add finer
# epilogue marks: col 52 A epilogues, 53 A gs + SwiGLU + stores, 54 B epilogue (stores or BRED red.adds), 55 A last-arriver
# read-back (fence_acq_rel + ldcg + zeroing). Compile-time switches only (no runtime branches in the consumer).
GF_PATCHES = [
    ("#ifndef VM_AD\n#define VM_AD 0\n#endif\n",
     "#ifndef VM_AD\n#define VM_AD 0\n#endif\n#ifndef VM_GSM\n#define VM_GSM 0\n#endif\n#ifndef VM_FSW\n#define VM_FSW 0\n#endif\n"),
    ("int est[E_LOC];\n", "int est[E_LOC];\n#if VM_GSM\nfloat gs13[E_LOC], gs2[E_LOC];\n#endif\n"),
    ("T.est[lane] = est;\n", "T.est[lane] = est;\n#if VM_GSM\nT.gs13[lane] = __ldg(A.gsc13 + lane); T.gs2[lane] = __ldg(A.gsc2 + lane);\n#endif\n"),
    ("__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }",
     "#if VM_FSW\n__device__ __forceinline__ float swiglu(float g, float u) { return __fdividef(g, 1.f + __expf(-g)) * u; }\n#else\n"
     "__device__ __forceinline__ float swiglu(float g, float u) { return g / (1.f + __expf(-g)) * u; }\n#endif"),
    ("if constexpr (!PB) {\nconst float gs = __ldg(A.gsc13 + e);\nconst int npad = (cn + 7) & ~7;\n",
     "if constexpr (!PB) {\nSEGMK(const long long t_a0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)\n"
     "#if VM_GSM\nconst float gs = T.gs13[e];\n#else\nconst float gs = __ldg(A.gsc13 + e);\n#endif\nconst int npad = (cn + 7) & ~7;\n"),
    ("__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));\n}\n",
     "__float2bfloat16_rn(swiglu(d[4 * nt + hh] * gs, d[4 * nt + 2 + hh] * gs));\n}\n"
     "SEGMK(if (A.marks && wt == 0 && q == 0) { A.marks[(size_t)blockIdx.x * MKW + 52] += 1; A.marks[(size_t)blockIdx.x * MKW + 53] += gtime() - t_a0; })\n"),
    ("} else {\nconst float gs = __ldg(A.gsc2 + e);\nconst int h0 = 16 * tg + g;\nfloat* yb = A.y + (size_t)T.jS[j] * KDIM;\n",
     "} else {\nSEGMK(const long long t_b0 = (A.marks && wt == 0 && q == 0) ? gtime() : 0;)\n"
     "#if VM_GSM\nconst float gs = T.gs2[e];\n#else\nconst float gs = __ldg(A.gsc2 + e);\n#endif\n"
     "const int h0 = 16 * tg + g;\nfloat* yb = A.y + (size_t)T.jS[j] * KDIM;\n"),
    ("yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;\n}\n}\n}\n}\n",
     "yb[(size_t)n * KDIM + h0 + 8] = d[4 * nt + 2 + hh] * gs;\n}\n}\n"
     "SEGMK(if (A.marks && wt == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_b0;)\n}\n}\n"),
    ("const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;\nconst float gs = __ldg(A.gsc2 + e);\nconst int h0 = 16 * tg + g;\n",
     "const int tg = (2 * ub + (q >> 1)) * 8 + (q & 1) * 4 + wp;\n"
     "SEGMK(const long long t_r0 = (A.marks && wp == 0 && lane == 0 && q == 0) ? gtime() : 0;)\n"
     "#if VM_GSM\nconst float gs = T.gs2[e];\n#else\nconst float gs = __ldg(A.gsc2 + e);\n#endif\nconst int h0 = 16 * tg + g;\n"),
    ("asm volatile(\"red.global.add.f32 [%0], %1;\\n\" :: \"l\"(yb + (size_t)n * KDIM + h0 + 8), \"f\"(d[4 * nt + 2 + hh] * gs) : \"memory\");\n}\n}\n}\n",
     "asm volatile(\"red.global.add.f32 [%0], %1;\\n\" :: \"l\"(yb + (size_t)n * KDIM + h0 + 8), \"f\"(d[4 * nt + 2 + hh] * gs) : \"memory\");\n}\n}\n"
     "SEGMK(if (A.marks && wp == 0 && lane == 0 && q == 0) A.marks[(size_t)blockIdx.x * MKW + 54] += gtime() - t_r0;)\n}\n"),
    ("if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }\nfence_acq_rel();\n",
     "if (!T.wlast[q]) { SEGMK(segmk_done(A, smk, s0, s1);) return; }\nSEGMK(const long long t_rb = smk.on ? gtime() : 0;)\nfence_acq_rel();\n"),
    ("if (wt == 0) *fc = 0;\n}\nepilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\n",
     "if (wt == 0) *fc = 0;\nSEGMK(if (smk.on) A.marks[(size_t)blockIdx.x * MKW + 55] += gtime() - t_rb;)\n}\nepilogue<PB, N>(A, T, j, ub, q, wp, lane, d, pd);\n"),
]
# ---- R-m32e-10 (parent queue after FSW): VM_UD = the wgmma B descriptor as one 64-bit "l" operand, built once per stage
# (base + slot offset) and advanced by constants, instead of mov.b64 from two 32-bit registers inside every wgmma asm (the
# R-unp-1 SASS: R2UR / IMAD.MOV / VIADD at every wgmma); VM_SYNC7 = flags bit 7 (mma.sync for nt <= 2) at compile time;
# VM_NOB9 = flags bit 9 code (B stage order, a 64-bit divide in afin) compiled out; VM_LD = griddepcontrol.launch_dependents
# at the A -> B boundary, so vm_fin (PDL secondary) launches and loads its norm weights under vm_main's end.
UD_PATCHES = [
    ("#ifndef VM_GSM\n#define VM_GSM 0\n#endif\n",
     "#ifndef VM_GSM\n#define VM_GSM 0\n#endif\n#ifndef VM_UD\n#define VM_UD 0\n#endif\n#ifndef VM_SYNC7\n#define VM_SYNC7 0\n#endif\n"
     "#ifndef VM_NOB9\n#define VM_NOB9 0\n#endif\n#ifndef VM_LD\n#define VM_LD 0\n#endif\n"
     "#if VM_NOB9\n#define VM_F9(a) (0)\n#else\n#define VM_F9(a) ((a).flags & 512)\n#endif\n"),
    ("template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);\n",
     "template <int N> __device__ __forceinline__ void wgmma_zero(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t lo, uint32_t hi);\n"
     "template <int N> __device__ __forceinline__ void wgmma_acc64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);\n"
     "template <int N> __device__ __forceinline__ void wgmma_zero64(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc);\n"),
    ("wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);\n}\n",
     "wgmma_acc<N>(d, hh[2], hh[6], hh[3], hh[7], dlo + 16, dhi);\n}\n#if VM_UD\ntemplate <int N, bool Z>\n"
     "__device__ __forceinline__ void mma2u(float* d, const uint32_t* hh, uint64_t desc) {\n"
     "if constexpr (Z) wgmma_zero64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);\nelse wgmma_acc64<N>(d, hh[0], hh[4], hh[1], hh[5], desc);\n"
     "wgmma_acc64<N>(d, hh[2], hh[6], hh[3], hh[7], desc + 16);\n}\n#endif\n"),
    ("const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n",
     "const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
     "#if VM_UD\nconst uint64_t dsc = d0 + (uint64_t)((uint32_t)xs * (uint32_t)(XS >> 4));\n#endif\n"),
    ("const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);\nconst uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);\n",
     "const uint64_t d0 = wg_desc_sa(sa(smem + NS * US + xs * XS), 4096u);\nconst uint32_t dlo = (uint32_t)d0, dhi = (uint32_t)(d0 >> 32);\n"
     "#if VM_UD\nconst uint64_t dsc = d0;\n#endif\n"),
    ("if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);\nelse mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);\n",
     "#if VM_UD\nif (first && i == 0 && hf == 0) mma2u<N, true>(d, hh, dsc + (uint64_t)(64 * i + 32 * hf));\n"
     "else mma2u<N, false>(d, hh, dsc + (uint64_t)(64 * i + 32 * hf));\n#else\n"
     "if (first && i == 0 && hf == 0) mma2<N, true>(d, hh, dlo + 64 * i + 32 * hf, dhi);\nelse mma2<N, false>(d, hh, dlo + 64 * i + 32 * hf, dhi);\n#endif\n"),
    ("if constexpr (N <= 16) sync_path = (A.flags & 128) != 0;\n",
     "#if VM_SYNC7\nif constexpr (N <= 16) sync_path = true;\n#else\nif constexpr (N <= 16) sync_path = (A.flags & 128) != 0;\n#endif\n"),
    ("if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n",
     "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();\n"
     "#if VM_LD\nif (tid == 0) asm volatile(\"griddepcontrol.launch_dependents;\\n\" ::: \"memory\");\n#endif\n"),
]
# ---- R-m32e-12 (parent GO 2026-10-28): VM_UD2 = the wgmma B descriptor from a PROVABLY warp-uniform value. bf2 SASS: every
# HGMMA reads gdesc[UR4] whose B half (UR6/UR7) is refilled per wgmma by VIADD (lo) + IMAD.MOV (hi = 0x100) + 2 x R2UR,
# because `it` (stage counter, carried through table-driven loops) is not uniform to ptxas. A REDUX (result in a UR) or
# SHFL re-uniformizes it: 1 = REDUX of xs, 2 = SHFL of xs, 3 = REDUX of the stage's dlo, 4 = 3 at stage entry.
UD2_PATCHES = [
    ("#ifndef VM_LD\n#define VM_LD 0\n#endif\n", "#ifndef VM_LD\n#define VM_LD 0\n#endif\n#ifndef VM_UD2\n#define VM_UD2 0\n#endif\n"),
    ("uint32_t (&hb)[2][8], int q, int wp, int lane, bool first, Pend& pd) {\nconst int ws = it % NS, xs = it % NX;\n",
     "uint32_t (&hb)[2][8], int q, int wp, int lane, bool first, Pend& pd) {\nconst int ws = it % NS, xs = it % NX;\n"
     "#if VM_UD2 == 4\nconst uint32_t dlo_e = __reduce_max_sync(0xffffffffu, (uint32_t)wg_desc_sa(sa(smem + NS * US), 4096u)"
     " + (uint32_t)xs * (uint32_t)(XS >> 4));\n#endif\n"),
    ("const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
     "#if VM_UD\nconst uint64_t dsc = d0 + (uint64_t)((uint32_t)xs * (uint32_t)(XS >> 4));\n#endif\n",
     "#if VM_UD2 == 1 || VM_UD2 == 2\n"
     "const uint32_t xsd = VM_UD2 == 1 ? __reduce_max_sync(0xffffffffu, (uint32_t)xs) : (uint32_t)__shfl_sync(0xffffffffu, xs, 0);\n"
     "const uint32_t dlo = (uint32_t)d0 + xsd * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
     "#elif VM_UD2 == 3\n"
     "const uint32_t dlo = __reduce_max_sync(0xffffffffu, (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4)), dhi = (uint32_t)(d0 >> 32);\n"
     "#elif VM_UD2 == 4\n"
     "const uint32_t dlo = dlo_e, dhi = (uint32_t)(d0 >> 32);\n"
     "#else\n"
     "const uint32_t dlo = (uint32_t)d0 + (uint32_t)xs * (uint32_t)(XS >> 4), dhi = (uint32_t)(d0 >> 32);\n"
     "#endif\n"
     "#if VM_UD\nconst uint64_t dsc = VM_UD2 ? (((uint64_t)dhi << 32) | dlo) : d0 + (uint64_t)((uint32_t)xs * (uint32_t)(XS >> 4));\n#endif\n"),
]
CUDA_PATCHES = (CUDA_PATCHES + DP_PATCHES + SC_PATCHES + CS_PATCHES + PS_PATCHES + AD_PATCHES + GF_PATCHES + UD_PATCHES
                + UD2_PATCHES)
CUDA_REPLACE_ALL = [("(A.flags & 512)", "VM_F9(A)")]   # every bit-9 test (compile-time 0 under VM_NOB9)

# wgmma specializations with the B descriptor as one 64-bit operand (VM_UD); same formatting as the 32-bit-pair specs
WG64_FN = r'''def @WG@64(n):
    na = n // 2
    outs = ",".join(f"%{i}" for i in range(na))
    a = ",".join(f"%{na + i}" for i in range(4))
    plus = ", ".join(f'"+f"(d[{i}])' for i in range(na))
    eq = ", ".join(f'"=f"(d[{i}])' for i in range(na))
    body = "wgmma.mma_async.sync.aligned.m64n%dk16.f32.bf16.bf16 {%s}, {%s}, %%%d, %d, 1, 1, 0;\\n"
    s = ""
    for nm, sd, cons in (("wgmma_acc64", 1, plus), ("wgmma_zero64", 0, eq)):
        s += ("template <> __device__ __forceinline__ void %s<%d>(float* d, uint32_t a0, uint32_t a1, uint32_t a2, "
              "uint32_t a3, uint64_t desc) {\n  asm volatile(\"" % (nm, n)
              + body % (n, outs, a, na + 4, sd)
              + "\"\n    : %s\n    : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc));\n}\n" % cons)
    return s


'''

# Python-level patches; @P@ = "" (lab) or "_vml_" (layer). Lab and layer spell some lines differently (indent, names).
PY_PATCHES = [
    ('"".join(@WG@(n) for n in (8, 16, 24, 32, 40, 48, 56, 64))', '"".join(@WG@(n) + @WG@64(n) for n in range(8, 129, 8))'),
    ("def @WG@(n):\n", WG64_FN + "def @WG@(n):\n"),
    ('void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V);\n',
     'void vml_run(std::vector<torch::Tensor> T, std::vector<int64_t> V);\nstd::vector<int64_t> vml_cfg();\n'),
    ('functions=["vml_run"]', 'functions=["vml_run", "vml_cfg"]'),
    ('"fix": z((2 * @P@SLOTS * 4 * @P@FIXW,), torch.float32)', '"fix": z((2 * (@P@SLOTS + 384) * 4 * @P@FIXW,), torch.float32)'),
    ('"cnt": z((2 * @P@SLOTS * 4,))', '"cnt": z((2 * (@P@SLOTS + 384) * 4,))'),
    ('mask=3, out=None, flags=None, cost=None):', 'mask=3, out=None, flags=None, cost=None, pool=(0, 0), seg=(0, 0), pdel=0):'),
    ('list(@P@COST + @P@COST if cost is None else cost))',
     '@P@cost_ext(mod, @P@COST + @P@COST if cost is None else cost) + @P@pool_ext(mod, pool) + @P@seg_ext(mod, seg)'
     ' + @P@pdel_ext(mod, pdel))'),
]

COST_EXT = '''

def @P@cost_ext(mod, cost):
    """cost table for a build: 16 entries (nt 1..8 per phase) are extended to 2 x NTM by the nt7 -> nt8 step (m32e)."""
    ntm = @P@cfg(mod)[1]
    assert ntm == 8, "m32e scratch (fix regions of 4096 floats) supports NTM 8 builds only"
    c = list(cost)
    if len(c) == 2 * ntm:
        return c
    assert len(c) == 16 and ntm >= 8, (len(c), ntm)
    out = []
    for pb in (0, 1):
        h = c[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out += h + [h[7] + step * (k - 7) for k in range(8, ntm + 1)]
    return out


def @P@pool_ext(mod, pool):
    """the two pool fractions (256ths of each phase's units) a VM_DP build reads after the cost table; [] otherwise."""
    return [int(pool[0]), int(pool[1])] if @P@cfg(mod)[6] else []


def @P@seg_ext(mod, seg):
    """the per-unit-start costs (A, B; cost units) a VM_SC build reads after the cost table; [] otherwise."""
    return [int(seg[0]), int(seg[1])] if @P@cfg(mod)[7] else []


def @P@pdel_ext(mod, pdel):
    """the staggered-start delay (ns per ring slot) a VM_PS build reads after the cost table; [] otherwise."""
    return [int(pdel)] if @P@cfg(mod)[10] else []


def @P@cfg(mod):
    """(JN, NTM, NX, XSN, MKW, SMEM, DP, SC) of a build (m32e); cached on the module object."""
    c = getattr(mod, "_m32e_cfg", None)
    if c is None:
        c = tuple(int(v) for v in mod.vml_cfg())
        try:
            mod._m32e_cfg = c
        except Exception:
            pass
    return c
'''


def sub1(text, old, new, what):
    n = text.count(old)
    assert n == 1, f"{what}: anchor found {n} times: {old[:90]!r}"
    return text.replace(old, new)


# k3_mb_1 and later bases carry MB's closed-form job split in route() (gen_k3_mb.py, -1..-1.6 us per call) instead of the
# countdown loop; the JN / VM_DP split patch then wraps that line instead of the loop (same jobs, same code at defaults).
_NL = chr(10)
_MB_SPLIT = _NL.join(["const int nj = (tot + 63) >> 6;", "int base = 64;",
                      "if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));"])
_LOOP_SPLIT = _NL.join(["const int nj = (tot + 63) >> 6;", "int base = 64;",
                        "if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }", ""])


def patch_cuda(cuda):
    for i, (old, new) in enumerate(CUDA_PATCHES):
        if old == _LOOP_SPLIT and cuda.count(old) == 0 and cuda.count(_MB_SPLIT) == 1:
            ln = cuda[cuda.index(_MB_SPLIT):].split(_NL, 3)
            mb = _NL.join(ln[:3]) + _NL   # the closed-form line keeps its trailing comment, if any
            new = new.replace(_LOOP_SPLIT, mb)
            old = mb
        cuda = sub1(cuda, old, new, f"cuda patch {i}")
    for old, new in CUDA_REPLACE_ALL:
        assert cuda.count(old) >= 1, old
        cuda = cuda.replace(old, new)
    return cuda


def patch_py(text, prefix):
    wg = "_vml_wg_spec" if prefix else "_wg_spec"
    for i, (old, new) in enumerate(PY_PATCHES):
        old, new = (v.replace("@P@", prefix).replace("@WG@", wg) for v in (old, new))
        text = sub1(text, old, new, f"py patch {i}")
    return text


def cuda_span(text, name):
    a = text.index(f'{name} = r"""') + len(f'{name} = r"""')
    b = text.index('"""', a)
    return a, b


def make_lab():
    src = LAB_SRC.read_text(encoding="utf-8")
    a, b = cuda_span(src, "CUDA_T")
    cuda = "\n".join(ln.lstrip(" ") for ln in src[a:b].split("\n"))
    cuda = "\n".join(ln for ln in cuda.split("\n") if not ln.startswith("//"))   # as in the layer copy
    out = src[:a] + patch_cuda(cuda) + src[b:]
    out = patch_py(out, "")
    anchor = "\n\nFLAGS_DEFAULT = 1"
    out = sub1(out, anchor, COST_EXT.replace("@P@", "") + f"\n\nWIDE_DEFS = {WIDE_DEFS!r}" + anchor, "lab cost_ext")
    hdr = ('"""m32e_moe.py: GENERATED by tools/m32e_patches.py from kernels/lab/vml_moe.py (lab v13) - do not edit by hand.\n'
           'Hot-wide VML switches VM_JN / VM_NTM / VM_NX / VM_XSN / VM_MKW (see tools/m32e_patches.py); defaults = VML v13.\n\n')
    assert out.startswith('"""')
    out = hdr + out[3:]
    LAB_DST.write_text(out, encoding="utf-8", newline="\n")
    return LAB_DST


if __name__ == "__main__":
    if sys.argv[1:] == ["lab"]:
        print(f"[m32e] wrote {make_lab()}")
    else:
        print(__doc__)

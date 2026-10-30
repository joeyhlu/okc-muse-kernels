"""vrh_patches.py: VM_RH = release helper for VML's phase-A doneA hand-off (moe redesign, R-vrh).

Shipped (VM_DEFER): at each A segment end the consumer WG runs pend_flush = fence.proxy.async + bar.sync(1+q, 128) + thread 0
red.release.gpu(doneA[i]). The release's MEMBAR stalls that warp, and with it the whole WG (wgmma is warpgroup-collective),
~0.35-0.7 us per A segment (R-lat32-5 marks; fence.release 1.76 us under a full ring, R-fnc-1). B segments use BRED and pay
nothing like it (board bs32: phase B matches stage + segment costs, phase A is +9.5 us over it, mb_11 marks).
VM_RH 1 (VM_WG5 >= 2 builds, i.e. w72): thread 0 of the WG pushes the doneA index into a per-WG smem ring (st.release.cta of the
ring count) instead; producer-WG warp NCW+2 (idle after gather_w5) polls the four rings (ld.acquire.cta) and issues the
red.release.gpu. Ordering: the WG's xB stores -> fence.proxy.async -> bar.sync -> st.release.cta -> helper ld.acquire.cta ->
red.release.gpu (cumulative) -> readers' acquire + fence.proxy.async + bulk copy: as shipped, one CTA-scope hop longer. After
phase A each WG pushes a sentinel (-2); the helper exits after four (bounded by WAIT_BUDGET, err code 3). Splits, epilogues and
phase B are untouched. At VM_RH 0 the source preprocesses to the base's.

    python -I tools/vrh_patches.py <file.py> --check
"""
import pathlib
import sys

RHQ = 32


def _rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, f"vrh anchor '{tag}' found {n} times"
    return s.replace(old, new)


DEFS = ("#ifndef VM_RH\n#define VM_RH 0   // R-vrh: 1 = doneA releases issued by producer-WG warp NCW+2 (VM_WG5 >= 2)\n#endif\n"
        "#if VM_RH && VM_WG5 < 2\n#error \"VM_RH needs the WG5 producer warpgroup (spare warp NCW+2)\"\n#endif\n")
HELPERS = r'''
#if VM_RH
__device__ __forceinline__ int rh_ld_acq(const int* p) {
int v;
asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];\n" : "=r"(v) : "r"(sa(p)) : "memory");
return v;
}
__device__ __forceinline__ void rh_st_rel(int* p, int v) {
asm volatile("st.release.cta.shared::cta.b32 [%0], %1;\n" :: "r"(sa(p)), "r"(v) : "memory");
}
__device__ __forceinline__ void rh_push(int q, int v) {   // thread 0 of WG q only (single writer of rhn[q])
extern __shared__ __align__(1024) uint8_t smem[];
Tab& T = *reinterpret_cast<Tab*>(smem + TAB_OFF);
const int k = T.rhn[q];
T.rhq[q][k & (VM_RHQ - 1)] = v;
rh_st_rel(&T.rhn[q], k + 1);
}
__device__ void rh_loop(const Args& A, Tab& T) {   // producer-WG warp NCW+2, lane 0
int seen0 = 0, seen1 = 0, seen2 = 0, seen3 = 0, done = 0;
const long long t0 = clock64();
while (done < 4) {
bool any = false;
#pragma unroll
for (int q = 0; q < 4; ++q) {
int& sn = q == 0 ? seen0 : q == 1 ? seen1 : q == 2 ? seen2 : seen3;
const int n = rh_ld_acq(&T.rhn[q]);
while (sn < n) {
const int idx = *reinterpret_cast<volatile int*>(&T.rhq[q][sn & (VM_RHQ - 1)]);
++sn;
any = true;
if (idx < 0) ++done;
else red_release_add(A.doneA + idx, 1);
}
}
if (!any) {
if (clock64() - t0 > WAIT_BUDGET) { atomicExch(A.err, 3); return; }
__nanosleep(64);
}
}
}
#endif
'''


def apply(s):
    s = _rep(s, "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n", DEFS + "#ifndef VM_DEFER\n#define VM_DEFER 0\n#endif\n", "defs")
    s = _rep(s, "unsigned long long bperm[JMAX];   // bit 9",
             f"#if VM_RH\n#define VM_RHQ {RHQ}\nint rhq[4][VM_RHQ];               // R-vrh: per-WG pending doneA indices (-2 = phase A done)\n"
             "int rhn[4];                       // R-vrh: per-WG push count (st.release.cta / ld.acquire.cta)\n#endif\n"
             "unsigned long long bperm[JMAX];   // bit 9", "tab")
    s = _rep(s, "if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);",
             "#if VM_RH\nif (((wp << 5) | lane) == 0) rh_push(q, pd.i);   // R-vrh: the release is issued by the helper warp\n#else\n"
             "if (((wp << 5) | lane) == 0) red_release_add(A.doneA + pd.i, 1);\n#endif", "pend_flush")
    s = _rep(s, "struct Pend { int i; };\n#endif\n", "struct Pend { int i; };\n#endif\n" + HELPERS, "helpers")
    s = _rep(s, "if (warp > NCW + 1) { if (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1); if (VM_W5B6) bar_arrive(6, VM_B6); return; }",
             "if (warp > NCW + 1) {\nif (A.flags & 64) gather_w5(A, T, ent, lane, warp - NCW - 1);\nif (VM_W5B6) bar_arrive(6, VM_B6);\n"
             "#if VM_RH\nif (warp == NCW + 2 && lane == 0) rh_loop(A, T);   // R-vrh\n#endif\nreturn;\n}", "helper warp")
    s = _rep(s, "for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }",
             "for (int i = 0; i < NS; ++i) { mbar_init(bars + 8 * i, 1); mbar_init(bars + 8 * (NS + i), NCW); }\n"
             "#if VM_RH\nfor (int k = 0; k < 4; ++k) T.rhn[k] = 0;   // R-vrh: before bar 9 (the producer WG waits on it)\n#endif", "init")
    s = _rep(s, "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();",
             "#if VM_RH\nif (wp == 0 && lane == 0) rh_push(q, -2);   // R-vrh: this WG's phase A is done (its last flush is pushed)\n#endif\n"
             "if (wp == 0 && lane == 0 && A.marks && q == 0) A.marks[(size_t)blockIdx.x * MKW + 6] = gtime();", "sentinel")
    return s


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    p = apply(t)
    print(f"[vrh] {sys.argv[1]}: +{len(p) - len(t)} B, all anchors applied once")

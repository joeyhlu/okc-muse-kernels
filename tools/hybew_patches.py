"""hybew_patches.py (R-hybew, HYB-prologue builder): HYB-EW (weight producer released at route end) + prologue / gather marks as
run-time arms on k3_hh_3tkv, one prepare. Follows R-hybpro-1 (notes/lab.md): HYB's first weight copy waits for WS_JOIN, i.e.
for the whole x gather (3.1-4.9 us after route), and the first x copy then queues behind the 108 KB of weight copies.

  python -I tools/hybew_patches.py [--base=kernels/lab/k3_hh_3tkv.py] [--out=kernels/lab/k3_hybew_ab.py] [--check]
  python -I tools/lead_gate.py file <out>

HYB = alygg77's WGS MoE engine (Apache-2.0, notice kept; T 49..320). The ship module (_HYB["mod"], unchanged source and flags)
stays the default; HYBP_ST selects an extra module built in prepare (PT1 job "hybp") from _HYB_MOE_CUDA with EDITS applied:
  16 = marks only (-DOKC_HYB_MK=1);  32 = EW only (-DOKC_HYBEW=1);  48 = EW + marks.
EW (OKC_HYBEW 1): the weight-producer warp (idle until WS_JOIN in the ship) joins bar 7 with the consumers' route arrive
(count NT + 32, unchanged), computes the job cost prefix warp-parallel (exclusive scan of the same uint32 terms as wgs_prefix:
bitwise the same S.wcp / wcpB / wtot) and the two ranges (wgs_range, lanes 0 / 1), then bar.arrive 8 (the x producer waits
there instead of computing the ranges itself) and bar.arrive 2 (counts toward WS_JOIN, NTL unchanged), and issues its first
WGS_NS copies while the consumers gather. Consumers and the x producer's issue loop are unchanged; all ranges are bitwise the
ship's, so outputs are too.
Marks (OKC_HYB_MK 1): 160 x 64 u64 buffer, slot k = %globaltimer, 32 + k = clock64 (k = 0..18), slot 31 = A stages |
B stages << 16 | nj << 32 | rows R << 48. k: 0 pre-wait, 1 wait done, 2 route done, 3 gather end (after tid 0's xready
release loop), 4 WS_JOIN passed, 5 ranges done (x producer, or the w producer under EW), 6 first weight copy issued, 7 first
x copy issued, 8 first weight stage landed / 9 first x stage landed (idle lane 1 of the w-producer warp), 10 first weight
slot freed (4th issue), 11 A end, 12 B end, 13 grid barrier, 14 kernel end; gather sub-marks (tid 0): 18 gather start (after
the CTA_SYNC that follows the jrow scan), 15 job walk done + loads issued, 16 loads returned + xA stores issued, 17 CTA_SYNC
after the stores passed (then the release loop; 3 = its end). 15 / 16 sit behind a compiler memory barrier so the stores
stay before the timer read.
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]
import hybpro_patches as HP  # noqa: E402  (shared Python glue: _hyb_call dispatch, PT1 job, prepare hook)
sys.argv = _argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
check, sub = HP.check, HP.sub

# ---------------- CUDA edits (raw text: asm "\n" stays two characters) ------------------------------------------------------
DEF_A = HP.MK_DEF_A
DEF_N = DEF_A + ("#ifndef OKC_HYB_MK\n#define OKC_HYB_MK 0\n#endif\n#ifndef OKC_HYBEW\n#define OKC_HYBEW 0\n#endif\n"
                 "#define MK(k) do { if (OKC_HYB_MK && p.dbg) { const unsigned long long t_ = gtimer(); "
                 "p.dbg[blockIdx.x * 64 + (k)] = t_; p.dbg[blockIdx.x * 64 + 32 + (k)] = (unsigned long long)clock64(); } } while (0)\n"
                 "#define MKB(k) do { asm volatile(\"\" ::: \"memory\"); MK(k); } while (0)\n")
JOIN_N = ("DBG(7);\nif (tid == 0) MK(3);\nWS_JOIN();\nDBG(6);\n"
          "if (tid == 0) { MK(4); if (OKC_HYB_MK && p.dbg) p.dbg[blockIdx.x * 64 + 31] = (unsigned long long)(S.wgr[1] - S.wgr[0]) "
          "| ((unsigned long long)(S.wgr[3] - S.wgr[2]) << 16) | ((unsigned long long)(S.njF + S.njS) << 32) "
          "| ((unsigned long long)S.jrow[S.njF + S.njS] << 48); }\n")
XP_A = r'''__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
if ((threadIdx.x & 31) == 0) wgs_ranges(p, S);
__syncwarp();
WS_JOIN();
'''
XP_N = r'''__device__ void wgs_xproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_HYBEW
asm volatile("bar.sync 8, 64;\n" ::: "memory");
#else
asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
if ((threadIdx.x & 31) == 0) { wgs_ranges(p, S); MK(5); }
#endif
__syncwarp();
WS_JOIN();
'''
WP_A = ("__device__ void wgs_ranges(const Params& p, Smem& S);\n"
        "__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {\nWS_JOIN();\nif ((threadIdx.x & 31) != 0) return;\n")
WP_N = r'''__device__ void wgs_ranges(const Params& p, Smem& S);
__device__ void wgs_ranges_warp(Smem& S);
__device__ void wgs_wproducer(const Params& p, Smem& S, unsigned char* ring) {
#if OKC_HYBEW
asm volatile("bar.sync 7, %0;\n" ::"n"(NT + 32) : "memory");
wgs_ranges_warp(S);
if ((threadIdx.x & 31) == 0) MK(5);
__syncwarp();
asm volatile("bar.arrive 8, 64;\n" ::: "memory");
asm volatile("bar.arrive 2, %0;\n" ::"n"(NTL) : "memory");
#else
WS_JOIN();
#endif
#if OKC_HYB_MK
if ((threadIdx.x & 31) == 1 && p.dbg && (S.wgr[1] > S.wgr[0] || S.wgr[3] > S.wgr[2])) { mbar_wait(S.bars + BW_F, 0u); MK(8); mbar_wait(S.bars + BX_F, 0u); MK(9); }
#endif
if ((threadIdx.x & 31) != 0) return;
'''
RW_A = ("__device__ __noinline__ void wgs_ranges(const Params& p, Smem& S) {\nconst int nj = S.njF + S.njS;\nwgs_prefix(S, nj);\n"
        "wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);\nwgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}\n")
RW_N = RW_A + r'''__device__ __noinline__ void wgs_ranges_warp(Smem& S) {
const int lane = threadIdx.x & 31, nj = S.njF + S.njS;
uint32_t carry = 0, carryB = 0;
for (int j0 = 0; j0 < nj; j0 += 32) {
const int j = j0 + lane;
const uint32_t v = j < nj ? (uint32_t)wgs_cost(0, S.jC[j]) << 9 : 0u, vB = j < nj ? (uint32_t)wgs_cost(1, S.jC[j]) << 8 : 0u;
uint32_t in = v, inB = vB;
#pragma unroll
for (int d = 1; d < 32; d <<= 1) {
const uint32_t t = __shfl_up_sync(0xffffffffu, in, d), tB = __shfl_up_sync(0xffffffffu, inB, d);
if (lane >= d) { in += t; inB += tB; }
}
if (j < nj) { S.wcp[j] = (int)(carry + in - v); S.wcpB[j] = (int)(carryB + inB - vB); }
carry += __shfl_sync(0xffffffffu, in, 31);
carryB += __shfl_sync(0xffffffffu, inB, 31);
}
if (lane == 0) { S.wcp[nj] = (int)carry; S.wcpB[nj] = (int)carryB; S.wtot = carry; S.wtotB = carryB; }
__syncwarp();
if (lane < 2) wgs_range(S, nj, lane, blockIdx.x, gridDim.x, S.wgr[2 * lane], S.wgr[2 * lane + 1]);
__syncwarp();
}
'''
GS_A = "CTA_SYNC();\nwgs_gather(p, S);\n"
GS_N = "CTA_SYNC();\nif (tid == 0) MK(18);\nwgs_gather(p, S);\n"
GI_A = ("}\n}\n}\n}\n#pragma unroll\nfor (int k = 0; k < GB; ++k)\nif (d[k]) *d[k] = v[k];\n}\nCTA_SYNC();\n"
        "if (tid == 0) {\nint jj = 0;\n")
GI_N = ("}\n}\n}\n}\nif (tid == 0 && qb == q0) MKB(15);\n#pragma unroll\nfor (int k = 0; k < GB; ++k)\nif (d[k]) *d[k] = v[k];\n"
        "if (tid == 0 && qb == q0) MKB(16);\n}\nCTA_SYNC();\nif (tid == 0) MK(17);\nif (tid == 0) {\nint jj = 0;\n")
EDITS = ([(DEF_A, DEF_N), (HP.MK_WAIT_A, HP.MK_WAIT_N), (HP.MK_ROUTE_A, HP.MK_ROUTE_N), (HP.MK_JOIN_A, JOIN_N)] + HP.MK_END
         + [(XP_A, XP_N), (HP.MK_XI_A, HP.MK_XI_N), (HP.MK_WI_A, HP.MK_WI_N), (WP_A, WP_N), (RW_A, RW_N), (GS_A, GS_N), (GI_A, GI_N)])

# ---------------- Python glue (hybpro's, with the new keys / flags / buffer stride) --------------------------------------------
HYB_LAB = HP.HYB_LAB
for a, n in (("_HYBP_KEYS = (7, 16)\n", "_HYBP_KEYS = (16, 32, 48)\n"),
             ('    return s, list(HYB_FLAGS) + [f"-DOKC_HYBST={k & 7}"] + (["-DOKC_HYB_MK=1"] if k & 16 else [])\n',
              '    return s, list(HYB_FLAGS) + (["-DOKC_HYB_MK=1"] if k & 16 else []) + (["-DOKC_HYBEW=1"] if k & 32 else [])\n'),
             ("torch.zeros(_HYB_SLOTS * 32, dtype=torch.int64", "torch.zeros(_HYB_SLOTS * 64, dtype=torch.int64")):
    check(HYB_LAB.count(a) == 1, f"HYB_LAB anchor {a[:50]!r}")
    HYB_LAB = HYB_LAB.replace(a, n)
NOTE_NEW = ("helpers.\nModified by team joeyhlu (2026-10-31, tools/hybew_patches.py, lab only): optional builds with per-CTA\n"
            "timing marks (-DOKC_HYB_MK) and with the weight producer released at route end (-DOKC_HYBEW: it computes the\n"
            "job ranges itself, warp-parallel); HYB's arithmetic and ranges are unchanged.\n\"\"\"\nTHIRD_PARTY_NOTICES += _HYB_NOTICE\n")


def apply(s):
    lines = s.split(NL)
    check(lines[0] == "# okc: problem=muse_decode_layer" and lines[1].startswith("# nonce "), "header lines")
    lines[1] = "# nonce k3-hybew-ab (k3_hh_3tkv + R-hybew lab arms: HYBP_ST 16 = marks, 32 = EW, 48 = EW + marks; tools/hybew_patches.py)"
    s = NL.join(lines)
    i0, i1 = s.index('_HYB_MOE_CUDA = r"""'), s.index('_HYB_MOE_CPP = r"""')
    cu = s[i0:i1]
    for a, n in EDITS:   # dry run on a copy: the file keeps the ship CUDA text; _hybp_src applies EDITS at run time
        check(cu.count(a) == 1, f"CUDA anchor not unique: {a[:70]!r}")
        cu = cu.replace(a, n)
    check(cu.count("MK(") + cu.count("MKB(") >= 21 and "wgs_ranges_warp(S);" in cu, "marks / EW inserted")
    s = sub(s, HP.HYB_CALL, HP.HYB_CALL_NEW, "_hyb_call")
    s = sub(s, "def _hyb_on(m):\n", HYB_LAB + f"_HYBP_EDITS = {EDITS!r}\n" + "def _hyb_on(m):\n", "HYBP lab block")
    s = sub(s, HP.HYB_PREP_END, HP.HYB_PREP_END_NEW, "_hyb_prepare end")
    s = sub(s, HP.PT1_HYB, HP.PT1_HYB_NEW, "PT1 hyb job")
    s = sub(s, HP.HST.A_NOTE, NOTE_NEW, "HYB notice end")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tkv.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybew_ab.py")
    new = apply(base.read_bytes().decode("ascii"))
    if "check" in FLAGS:
        check(out.exists(), f"{out} missing")
        check(out.read_bytes().decode("ascii") == new, f"{out.name} differs from apply({base.name})")
        HP.gates(out, new)
        print(f"hybew_patches: --check OK ({out.name} == apply({base.name}))")
        return
    out.write_bytes(new.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hybew_patches: {base.name} -> {out.name} ({len(new):,} B, +{len(new) - base.stat().st_size:,} B)")
    HP.gates(out, new)


if __name__ == "__main__":
    main()

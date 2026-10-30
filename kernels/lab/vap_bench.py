"""vap_bench.py: R-vap (anti-phase warpgroup pairs in the VML RS-wgmma consumer) on vcs_bench's harness (k3_gc_18 VML head).

  python3 -u kernels/lab/vap_bench.py sass OUTDIR [--builds=bf2,w72]
  python3 -u kernels/lab/vap_bench.py run [--ntoks=16,32,64] [--rounds=9] [--units=3] [--out=box_out/vap/run.json]

Question (R-vcs-0 / R-vcs-5): a VML stage costs ~ MMA time + INT-pipe unpack (w72 N32 nosync 1943 clk vs dq-only 1354, MMA-only
1232). Is that a scheduling effect (all 4 WGs issue wgmma into one tensor queue and stall behind it in order, so their next
unpack waits) or a hardware conflict (RS wgmma and INT ops on one SMSP do not overlap at all)?
Diagnostic arms (nosync, no barriers, warm; V 1):
  M 4 split: WGs 0-1 MMA-only (raw code words as A), WGs 2-3 dequant-only, at the same time;
  M 5 hmma: WGs 0-1 MMA-only, WGs 2-3 idle;  M 6 hdq: WGs 2-3 dequant-only, WGs 0-1 idle;  M 7 hful: WGs 0-1 full, 2-3 idle.
  split ~ max(hmma, hdq) -> different warps overlap the pipes (scheduling); split ~ hmma + hdq -> hardware conflict.
Candidate (V 9 / 10 / 11 = turn of k64 / k32 / k128 per pair): FA3-style ping-pong over WG PAIRS (q 0-1 = stage half 0, q 2-3 =
half 1), 96 registers kept: per turn a pair unpacks its chunk, takes the turn (named barrier 12 + pair), issues the chunk's
HGMMAs in one commit group, hands the turn over, then wait_group 0. The other pair's unpack runs under this pair's HGMMAs.
Same k16 chain order into the same accumulator as the incumbent -> bitwise == inc.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools", ROOT / "tools/judge", ROOT / "kernels/muse_moe_w4a16", ROOT / "kernels/lab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vcs_bench as V  # noqa: E402

AP_DEV = r"""
// R-vap: anti-phase WG pairs. CH 0: k64 per turn (2 groups, 16 A regs), 1: k32 (1 group), 2: k128 (4 groups, 32 A regs).
template <int N, int M, int CH>
__device__ __forceinline__ void vcs_seg_ap(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp, int lane) {
constexpr bool SY = (M == 0);
constexpr int GPC = CH == 1 ? 1 : (CH == 0 ? 2 : 4), NCH = 8 / GPC;
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = 0.f;
uint32_t hb[GPC][8];
const int pr = q >> 1;
const int bme = 12 + pr, bot = 13 - pr;
const int tile = (q & 1) * 4 + wp;
const uint32_t cofs = (uint32_t)((q >> 1) * SW + tile * 2048 + lane * 16), sofs = (uint32_t)((q >> 1) * SW + 16384 + tile * 256 + lane * 4);
const uint32_t zo = (uint32_t)A.flags >> 30;
const uint32_t k16 = 16u + zo, k64 = VM_TRICKS_DQ ? 64u + zo : 64u, k4 = VM_TRICKS_DQ ? 4u + zo : 4u;
const uint32_t m10 = unp_ok32(1u << 10), m13 = unp_ok32(1u << 13);
const uint64_t dd = wg_desc_sa(sa(smem + NS * US), 4096u);
const uint32_t dhi = (uint32_t)(dd >> 32);
VRing r = vring_at(it);
uint4 c[4];
uint32_t sw0, sw1;
if (SY) vwait<0>(bars + 8 * r.ws, r.wph, A.err);
{
const uint4* cp = VCS_CP(r.ws);
const uint32_t* sp = VCS_SP(r.ws);
#pragma unroll
for (int i = 0; i < 4; ++i) c[i] = cp[32 * i];
sw0 = sp[0]; sw1 = sp[32];
}
if (SY) {
vwait<0>(bars + 8 * (2 * NS + r.ws), r.wph, A.err);
__syncwarp();
}
if (pr == 1) bar_arrive(12, 512);   // pair 0 takes the first turn
int s = 0;
#pragma unroll 1
do {
const bool more = s < 15;
const VRing r1 = vring_next(r);
const uint4* cp1 = VCS_CP(r1.ws);
const uint32_t* sp1 = VCS_SP(r1.ws);
const uint32_t dlo = (uint32_t)dd + (uint32_t)(SY ? r.ws : 0) * (uint32_t)(XS >> 4);
#pragma unroll
for (int ch = 0; ch < NCH; ++ch) {
#pragma unroll
for (int gi = 0; gi < GPC; ++gi) {
const int g = ch * GPC + gi, i = g >> 1, hf = g & 1;
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
vdec<0>(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hb[gi], k16, k64, k4, m10, m13);
if (SY && i == 0 && hf == 1) {   // every code word of this stage is in registers now
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + r.ws) + ((c[1].x ^ c[2].x ^ c[3].x ^ sw1) & zo));
}
if (more && hf == 1) {   // rolling refill of k64 step i's dead code words with stage s + 1's
if (i == 1) { if (SY) vwait<0>(bars + 8 * r1.ws, r1.wph, A.err); c[0] = cp1[0]; c[1] = cp1[32]; sw0 = sp1[0]; }
if (i == 2) c[2] = cp1[64];
if (i == 3) { c[3] = cp1[96]; sw1 = sp1[32]; }
}
}
bar_sync(bme, 512);
wg_fence();
#pragma unroll
for (int gi = 0; gi < GPC; ++gi) {
const int g = ch * GPC + gi, i = g >> 1, hf = g & 1;
mma2<N, false>(d, hb[gi], dlo + 64 * i + 32 * hf, dhi);
}
wg_commit();
if (!(pr == 1 && !more && ch == NCH - 1)) bar_arrive(bot, 512);   // balanced per segment
wg_wait<0>();
}
if (SY) {   // all of this stage's HGMMAs are complete: its x slot is free
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + r.ws));
if (more) vwait<0>(bars + 8 * (2 * NS + r1.ws), r1.wph, A.err);
__syncwarp();
}
r = r1;
} while (++s < 16);
it += 16;
}
// R-vap diagnostics: M 4 split (WGs 0-1 MMA-only, 2-3 dequant-only), 5 hmma, 6 hdq, 7 hful (WGs 0-1 full, 2-3 idle); nosync
template <int N, int MM>
__device__ __forceinline__ void vcs_one(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp, int lane) {
if constexpr (N <= 16) vcs_seg_m<N, MM, 0>(A, smem, bars, it, d, q, wp, lane);
else vcs_seg_rs<N, MM, 0, 0>(A, smem, bars, it, d, q, wp, lane);
}
template <int N, int M>
__device__ __forceinline__ void vcs_diag(const Args& A, uint8_t* smem, uint32_t bars, int& it, float* d, int q, int wp, int lane) {
const bool lo = q < 2;
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = 0.f;
if constexpr (M == 4) { if (lo) vcs_one<N, 3>(A, smem, bars, it, d, q, wp, lane); else vcs_one<N, 2>(A, smem, bars, it, d, q, wp, lane); }
if constexpr (M == 5) { if (lo) vcs_one<N, 3>(A, smem, bars, it, d, q, wp, lane); }
if constexpr (M == 6) { if (!lo) vcs_one<N, 2>(A, smem, bars, it, d, q, wp, lane); }
if constexpr (M == 7) { if (lo) vcs_one<N, 1>(A, smem, bars, it, d, q, wp, lane); }
}
"""

OLD_DISPATCH = ("else if constexpr (N <= 16) vcs_seg_m<N, M, VF<V>::F>(A, smem, bars, it, d, q, wp, lane);\n"
                "else vcs_seg_rs<N, M, VF<V>::G, VF<V>::F>(A, smem, bars, it, d, q, wp, lane);\n")
NEW_DISPATCH = ("else if constexpr (V >= 9) vcs_seg_ap<N, M, V - 9>(A, smem, bars, it, d, q, wp, lane);\n"
                "else if constexpr (M >= 4) vcs_diag<N, M>(A, smem, bars, it, d, q, wp, lane);\n" + OLD_DISPATCH)
OLD_END = "if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime(); if (q == 0) mk[6] = clock64(); }\n}\n}  // namespace vm"
NEW_END = ("if (mk && wp == 0 && lane == 0) { mk[1 + q] = gtime();   // R-vap: clk end = the LAST WG's end (idle WGs in M 5-7)\n"
           "atomicMax(reinterpret_cast<unsigned long long*>(mk + 6), (unsigned long long)clock64()); }\n}\n}  // namespace vm")
ANCHOR = "// V -> (G, F): 1 lean"


def patch():
    s = V.VCS_DEV
    for old in (OLD_DISPATCH, OLD_END, ANCHOR):
        assert s.count(old) == 1, old[:60]
    s = s.replace(ANCHOR, AP_DEV + ANCHOR).replace(OLD_DISPATCH, NEW_DISPATCH).replace(OLD_END, NEW_END)
    V.VCS_DEV = s
    V.BUILDS = {"bf2": (16, 32), "w72": (32, 64)}

    def kernels_of(build):
        out = []
        for n in V.BUILDS[build]:
            out.append((n, 0, 0))
            out += [(n, 1, m) for m in (1, 2, 3, 4, 5, 6, 7)]
            if n > 16:
                out += [(n, 9, 0), (n, 9, 1), (n, 10, 0), (n, 10, 1), (n, 11, 0), (n, 11, 1)]
        return out
    V.kernels_of = kernels_of
    V.ARMS = {"inc": (0, 0, 0), "lean_ns": (1, 1, 0), "dq_ns": (1, 2, 0), "mma_ns": (1, 3, 0),
              "split_ns": (1, 4, 0), "hmma_ns": (1, 5, 0), "hdq_ns": (1, 6, 0), "hful_ns": (1, 7, 0),
              "ap64": (9, 0, 0), "ap64_ns": (9, 1, 0), "ap32": (10, 0, 0), "ap32_ns": (10, 1, 0), "ap128": (11, 0, 0), "ap128_ns": (11, 1, 0)}
    V.STREAM = ("inc", "ap64", "ap32", "ap128")
    V.CHECKED = ("ap64", "ap32", "ap128")
    _orig_loop = V.loop_spec

    def loop_spec(n, v, m):
        if m in (2, 6):
            return "HMUL2", 64
        return _orig_loop(n, v, m)
    V.loop_spec = loop_spec


if __name__ == "__main__":
    patch()
    if len(sys.argv) > 2 and sys.argv[1] == "sass":
        V.sass_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "run":
        V.run_main(sys.argv[2:])
    else:
        print(__doc__)

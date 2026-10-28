"""mb_gen.py: kernels/muse_moe_w4a16/mb_<n>.py = one MoE board file with our two one-launch MoE kernels and an
in-prepare variant sweep (the vjeux iteration loop on the free MoE board checks; notes/study/moe_board_loop.md):

  * VMS (kernels/lab/vms_moe_lab.py, CUDA verbatim + a compile-time VMS_MAXT patch so one build can take T <= 32):
    one persistent launch, 36 KB full-K mma.sync units, jobs of <= 8 tokens;
  * VML (kernels/lab/vml_moe.py via tools/vml_patches.lab_block, + runtime flags bit 10 = 64-token chunks plus a
    remainder instead of balanced splits): one persistent RS-wgmma stream-K launch + vm_fin;
  * no K2A kernels: VML's ws13 / ws2 stage layouts come from K2A's prepack helpers only (copied verbatim from
    kernels/muse_moe_w4a16/vml_2.py); a torch FP32 path is the last-resort fallback.

prepare(): builds every compile-time variant in parallel threads (MB_BUILDS), prepacks experts [0, 32), fits VML stage-cost
tables from per-CTA marks on the judge GPU (board-like and layer-like routings), self-tests every arm against an FP32
reference (MB_CASES, two calls each, error flags, finite), then times every passing arm per bucket on cold-L2 CUDA graph
replays of board-like routings (generate_input's distribution, our own seeds) and of layer-like routings (real layer
count profiles from box_out/{vms,m32,sk}/run.json), prints one line per arm and routes each bucket to its fastest arm on the
board routings. Check only, never scored.

    python -I tools/mb_gen.py --n=1
"""
import ast
import hashlib
import json
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vml_patches  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 1))
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
VMS_LAB = ROOT / "kernels/lab/vms_moe_lab.py"
K2A_SRC = ROOT / "kernels/muse_moe_w4a16/vml_2.py"
LAYER = ROOT / "kernels/muse_decode_layer/k3_vf_10.py"

# ---------------------------------------------------------------- variant sets per generation
CONFIGS = {
    1: dict(
        builds={"s3": ("vms", 3, 16, 0), "s2": ("vms", 2, 16, 0), "w3": ("vms", 3, 32, 0), "w2": ("vms", 2, 32, 0),
                "lt": ("vml", ()), "lo": ("vml", ("-DVM_TRICKS=0",)), "ls": ("vml", ("-DVM_TRICKS_DQ=0",))},
        arms={1: ["s3", "s2", "w3"],
              8: ["s3", "s2", "w3", "lt:193:d"],
              16: ["s3", "s2", "w3", "w2", "lt:193:d", "lt:193:f"],
              32: ["w3", "w2", "lt:193:d", "lt:193:L", "lt:193:f", "lt:193:f2", "lt:193:g", "lt:1217:f", "lt:65:f",
                   "lo:193:f"],
              128: ["lt:193:d", "lt:193:L", "lt:193:f", "lt:193:f2", "lt:193:g", "lt:1217:f", "lt:65:f", "lt:195:f",
                    "lo:193:f", "ls:193:f"],
              256: ["lt:193:d", "lt:193:L", "lt:193:f", "lt:193:f2", "lt:193:g", "lt:1217:f", "lt:65:f", "lt:195:f",
                    "lo:193:f", "ls:193:f"]},
    ),
    # mb_2 (T >= 32 first): flags bit 11 (x -> L2 by every thread at entry), consumer gather (129) vs producer-warp gather
    # + self-fill (193), bit 10 chunks, in-prepare tuned tables (t) and layer-routing fits (g)
    2: dict(
        builds={"s3": ("vms", 3, 16, 0), "w3": ("vms", 3, 32, 0), "lt": ("vml", ()), "lo": ("vml", ("-DVM_TRICKS=0",))},
        arms={1: ["s3"],
              8: ["s3"],
              16: ["s3", "w3"],
              32: ["w3", "lt:193:f", "lt:2241:f", "lt:129:f", "lt:2177:f", "lt:193:t", "lt:2241:t", "lo:2241:f", "lt:193:L"],
              128: ["lt:193:f", "lt:2241:f", "lt:129:f", "lt:2177:f", "lt:3265:f", "lt:193:t", "lt:2241:t", "lo:2241:f",
                    "lt:193:L", "lt:2241:g"],
              256: ["lt:193:f", "lt:2241:f", "lt:129:f", "lt:2177:f", "lt:3265:f", "lt:193:t", "lt:2241:t", "lo:2241:f",
                    "lt:193:L", "lt:2241:g"]},
    ),
    # mb_3: flags bit 12 (finalize inside vm_main, no vm_fin), robust tuned tables (24 routings, from L), TRICKS_DQ 0 (ls,
    # adopted by R-vml-16), VMS K16 jobs (FL 128) at MAXT 32 for bs16 / bs32, per-CTA marks of the best arms
    3: dict(
        builds={"s3": ("vms", 3, 16, 0), "w3": ("vms", 3, 32, 0), "wk": ("vms", 3, 32, 128), "lt": ("vml", ()),
                "ls": ("vml", ("-DVM_TRICKS_DQ=0",)), "l0": ("vml", ("-DVM_FININ=0",))},
        arms={1: ["s3"],
              8: ["s3"],
              16: ["s3", "wk"],
              32: ["w3", "wk", "l0:193:L", "lt:193:L", "lt:4289:L", "ls:4289:L", "lt:5313:L", "lt:4289:t"],
              128: ["l0:193:L", "lt:193:L", "lt:4289:L", "ls:4289:L", "ls:193:L", "lt:5313:L", "lt:4289:t"],
              256: ["l0:193:L", "lt:193:L", "lt:4289:L", "ls:4289:L", "ls:193:L", "lt:5313:L", "lt:4289:t"]},
    ),
    # mb_4: VMS smem-size probes (does smem alone explain K16's -16 %? it gates 2-tile jobs at T=16 in the layer), K16 at
    # MAXT 16 (R-vms-6 build), VML without fin_in code (bit 12 killed), register cap 112 instead of ptxas' 96 (lr), bit 10
    # on L; paired ratios with standard errors; per-routing marks on board and layer-like routings
    4: dict(
        builds={"s3": ("vms", 3, 16, 0), "p16": ("vms", 3, 16, 0, 16384), "p64": ("vms", 3, 16, 0, 65536),
                "k16": ("vms", 3, 16, 128), "lt": ("vml", ("-DVM_FININ=0",)),
                "lr": ("vml", ("-DVM_FININ=0", "-DVM_LB=0", "-maxrregcount=112")),
                "ls": ("vml", ("-DVM_FININ=0", "-DVM_TRICKS_DQ=0"))},
        arms={1: ["s3", "p16", "p64"],
              8: ["s3", "p16", "p64", "k16"],
              16: ["s3", "p16", "p64", "k16"],
              32: ["lt:193:L", "lt:1217:L", "lr:193:L", "ls:193:L"],
              128: ["lt:193:L", "lt:1217:L", "lr:193:L", "lr:1217:L", "ls:193:L"],
              256: ["lt:193:L", "lt:1217:L", "lr:193:L", "lr:1217:L", "ls:193:L"]},
    ),
    # mb_5: VMS X2S (16-token jobs, second x tile in smem, 1-tile path unchanged) at bs8 / 16 (layer T=16 hot experts
    # re-stream today) and MAXT 32 for bs32; VML bit 13 (exact fast path of the balanced-size loop when an expert is one
    # job) vs bit 10, register cap 104 without launch bounds (112 did not launch), TRICKS_DQ 0 with bit 10
    5: dict(
        builds={"s3": ("vms", 3, 16, 0), "x2": ("vms", 3, 16, 1152), "x2w": ("vms", 3, 32, 1152),
                "lt": ("vml", ("-DVM_FININ=0",)), "lr": ("vml", ("-DVM_FININ=0", "-DVM_LB=0", "-maxrregcount=104")),
                "ls": ("vml", ("-DVM_FININ=0", "-DVM_TRICKS_DQ=0"))},
        arms={1: ["s3", "x2"],
              8: ["s3", "x2"],
              16: ["s3", "x2", "x2w"],
              32: ["lt:193:L", "lt:1217:L", "lt:8385:L", "lr:1217:L", "ls:1217:L", "x2w"],
              128: ["lt:193:L", "lt:1217:L", "lt:8385:L", "lr:1217:L", "ls:1217:L"],
              256: ["lt:193:L", "lt:1217:L", "lt:8385:L", "lr:1217:L", "ls:1217:L"]},
    ),
    # mb_6: VMS X2S with cost-weighted CTA unit ranges (VMS_WT: a 2-tile unit weighs 1.25 / 1.5 1-tile units) at bs8 / 16
    # (layer T=16 hot experts of 9-14 tokens stop re-streaming); VML bit 13 = the closed form of the job-size loop (all nj)
    6: dict(
        builds={"s3": ("vms", 3, 16, 0), "x2": ("vms", 3, 16, 1152), "xw5": ("vms", 3, 16, 1152, 0, 5),
                "xw6": ("vms", 3, 16, 1152, 0, 6), "lt": ("vml", ("-DVM_FININ=0",)),
                "ls": ("vml", ("-DVM_FININ=0", "-DVM_TRICKS_DQ=0"))},
        arms={1: ["s3"],
              8: ["s3", "xw5"],
              16: ["s3", "x2", "xw5", "xw6"],
              32: ["ls:193:L", "ls:8385:L", "ls:1217:L", "lt:8385:L"],
              128: ["ls:193:L", "ls:8385:L", "ls:1217:L", "lt:8385:L"],
              256: ["ls:193:L", "ls:8385:L", "ls:1217:L", "lt:8385:L"]},
    ),
    # mb_7: VMS X2S + VMS_WT with the cheap routing (weights scanned with the job counts, 4 lanes compute the CTA ranges)
    # at weights 1.5 / 1.75 / 2.0; VML closed form again (ls) as the T >= 32 control
    7: dict(
        builds={"s3": ("vms", 3, 16, 0), "xw6": ("vms", 3, 16, 1152, 0, 6), "xw7": ("vms", 3, 16, 1152, 0, 7),
                "xw8": ("vms", 3, 16, 1152, 0, 8), "ls": ("vml", ("-DVM_FININ=0", "-DVM_TRICKS_DQ=0"))},
        arms={1: ["s3", "xw6"],
              8: ["s3", "xw6"],
              16: ["s3", "xw6", "xw7", "xw8"],
              32: ["ls:193:L", "ls:8385:L"],
              128: ["ls:193:L", "ls:8385:L"],
              256: ["ls:193:L", "ls:8385:L"]},
    ),
    # mb_8: is X2S's 1-tile slowdown register allocation? the same X2S + WT 6 build capped at 149 / 152 registers (s3 uses
    # 149); bs1 / 8 measure the 1-tile path alone, bs16 the T=16 lever (port gate: layer-like r <= 0.94)
    8: dict(
        builds={"s3": ("vms", 3, 16, 0), "xw6": ("vms", 3, 16, 1152, 0, 6),
                "xr149": ("vms", 3, 16, 1152, 0, 6, ("-DVMS_LB=0", "-maxrregcount=149")),
                "xr144": ("vms", 3, 16, 1152, 0, 6, ("-DVMS_LB=0", "-maxrregcount=144")),
                "ls": ("vml", ("-DVM_FININ=0", "-DVM_TRICKS_DQ=0"))},
        arms={1: ["s3", "xw6", "xr149", "xr144"],
              8: ["s3", "xw6", "xr149", "xr144"],
              16: ["s3", "xw6", "xr149", "xr144"],
              32: ["ls:8385:L"],
              128: ["ls:8385:L"],
              256: ["ls:8385:L"]},
    ),
}
CFG = CONFIGS[N]

# ---------------------------------------------------------------- VMS: CUDA verbatim from the lab + VMS_MAXT
VMS_PATCHES = (
    ("#ifndef VMS_FL\n#define VMS_FL 0\n#endif\n",
     "#ifndef VMS_FL\n#define VMS_FL 0\n#endif\n#ifndef VMS_MAXT\n#define VMS_MAXT 16\n#endif\n#ifndef VMS_PAD\n#define VMS_PAD 0\n#endif\n"),
    ("#if VMS_SKT == 16\n  float4 red2[NRB][NW][32];\n#endif\n};\n",
     "#if VMS_SKT == 16\n  float4 red2[NRB][NW][32];\n#endif\n#if VMS_PAD > 0\n  unsigned char pad[VMS_PAD];   // mb probe: smem size only\n#endif\n};\n"),
    ("constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = 128, MAXJ = 64;",
     "constexpr int HD = 4096, ID = 4096, NE = 32, MAXP = VMS_MAXT * 8, MAXJ = 64, NRT = VMS_MAXT / 4;"),
    ("  int base[4][NE];\n", "  int base[NRT][NE];\n"),
    ("    int idv[4]; unsigned mm[4];\n", "    int idv[NRT]; unsigned mm[NRT];\n"),
    ("    for (int r = 0; r < 4; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }\n",
     "    for (int r = 0; r < NRT; ++r) { const int i = r * 32 + lane; idv[r] = i < m8 ? ((__ldcg(p.ids + i) - p.lo) & 255) : 255; }\n"),
    ("    for (int r = 0; r < 4; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }\n",
     "    for (int r = 0; r < NRT; ++r) { if (idv[r] >= NE || idv[r] < 0) idv[r] = -1; S.base[r][lane] = 0; }\n"),
    ("    __syncwarp();\n#pragma unroll\n    for (int r = 0; r < 4; ++r) {\n      mm[r] = __match_any_sync(",
     "    __syncwarp();\n#pragma unroll\n    for (int r = 0; r < NRT; ++r) {\n      mm[r] = __match_any_sync("),
    ("    int cr[4], tot = 0;\n#pragma unroll\n    for (int r = 0; r < 4; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }\n",
     "    int cr[NRT], tot = 0;\n#pragma unroll\n    for (int r = 0; r < NRT; ++r) { cr[r] = S.base[r][lane]; tot += cr[r]; }\n"),
    ("    for (int r = 0; r < 4; ++r) { S.base[r][lane] = run; run += cr[r]; }\n",
     "    for (int r = 0; r < NRT; ++r) { S.base[r][lane] = run; run += cr[r]; }\n"),
    ("    const unsigned lt = (1u << lane) - 1u;\n#pragma unroll\n    for (int r = 0; r < 4; ++r) {\n",
     "    const unsigned lt = (1u << lane) - 1u;\n#pragma unroll\n    for (int r = 0; r < NRT; ++r) {\n"),
    ('  TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");',
     '  TORCH_CHECK(m >= 1 && m <= VMS_MAXT, "vms: 1 <= m <= VMS_MAXT");'),
)

# VMS_X2S (with VMS_FL 128 = 16-token jobs): a 2-tile job's second x tile lives in shared memory (each lane's own 16 uint4,
# x2s[warp][k][lane], written and read by the same thread) instead of 64 registers, and both tiles use the 4 accumulator
# chains of the 1-tile path (2 each); 1-tile units run the 1-tile build's inline loop unchanged. K16 (x2 in registers,
# acc2) was x0.84 at bs16: this keeps the 1-tile path's register footprint (mb_4 p16 / p64 probes check the smem side).
X2S_CONSUME = r'''
#if VMS_X2S
// mb VMS_X2S: one unit for one warp, 2 n-tiles: tokens 0-7 from x registers, tokens 8-15 from this lane's smem slot
__device__ __forceinline__ void consume2s(const unsigned char* slot, const uint4 sc, const uint4 (&xr)[2 * NCH],
                                          const uint4* x2p, float (&acc)[4][4], uint32_t k16, int warp, int lane) {
#pragma unroll
  for (int cc = 0; cc < NCH; ++cc) {
    const uint4 w = *reinterpret_cast<const uint4*>(slot + (warp * NCH + cc) * 512 + lane * 16);
    const uint32_t sw = cc < 4 ? (cc < 2 ? sc.x : sc.y) : (cc < 6 ? sc.z : sc.w);
    const uint32_t sel = (cc & 1) ? 0x4242u : 0x4040u;
    const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);
    const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
    const uint32_t bb[8] = {xr[2 * cc].x, xr[2 * cc].y, xr[2 * cc].z, xr[2 * cc].w,
                            xr[2 * cc + 1].x, xr[2 * cc + 1].y, xr[2 * cc + 1].z, xr[2 * cc + 1].w};
    const uint4 y0 = x2p[(2 * cc) * 32], y1 = x2p[(2 * cc + 1) * 32];
    const uint32_t b2[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t qw = ww[s];
      const uint32_t M = 0x81C081C0u;
      const uint32_t r0 = qw & M;
      const uint32_t r1 = (qw * 16u) & M;
      const uint32_t r2 = __funnelshift_l(qw, qw, 10) & M;
      const uint32_t r3 = ((qw * 64u) & 0x80C080C0u) | (__funnelshift_l(qw, qw, 14) & 0x01000100u);
      const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);
      mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);
      mma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);
    }
  }
}
#endif
'''
WT_INV = r'''
#if VMS_WT
// mb VMS_WT: the unit index where CTA boundary c (0..G) falls when a unit of a 1-tile job weighs 4 and a unit of a 2-tile
// job VMS_WT quarter units (S.jW = per-job weight prefix); R = units per job in this phase (512 A / 256 B). Boundaries are
// monotone in c and the same function for c and c + 1, so the CTA ranges partition [0, njobs x R) exactly.
__device__ __forceinline__ int wt_inv(const Smem& S, int c, int R) {
  const int nj = S.njobs;
  if (c >= G) return nj * R;
  const int tg = (S.jW[nj] * R * c) / G;
  int lo = 0, hi = nj;
  while (hi - lo > 1) { const int mid = (lo + hi) >> 1; if (S.jW[mid] * R <= tg) lo = mid; else hi = mid; }
  const int w = S.jW[lo + 1] - S.jW[lo], off = tg - S.jW[lo] * R;
  const int u = (w == 4) ? ((off + 3) >> 2) : ((off + VMS_WT - 1) / VMS_WT);
  return min(lo * R + u, (lo + 1) * R);
}
#endif
'''
X2S_PATCHES = (
    ("#ifndef VMS_PAD\n#define VMS_PAD 0\n#endif\n",
     "#ifndef VMS_PAD\n#define VMS_PAD 0\n#endif\n#define VMS_X2S ((VMS_FL & 1024) != 0)   // mb: FL bit 1024 = smem second x tile (with FL 128)\n"
     "#ifndef VMS_WT\n#define VMS_WT 0   // mb: > 0 = cost-weighted CTA unit ranges, a 2-tile unit weighs VMS_WT / 4\n#endif\n"),
    ("#if VMS_PAD > 0\n  unsigned char pad[VMS_PAD];   // mb probe: smem size only\n#endif\n};\n",
     "#if VMS_PAD > 0\n  unsigned char pad[VMS_PAD];   // mb probe: smem size only\n#endif\n#if VMS_SKT == 16 && VMS_X2S\n"
     "  uint4 x2s[NW][2 * NCH][32];   // mb VMS_X2S: second x tile per lane\n#endif\n#if VMS_WT\n"
     "  int jW[MAXJ + 1];   // mb VMS_WT: per-job weight prefix (quarter units per unit)\n"
     "  int rng[4];   // mb VMS_WT: this CTA's a0, a1, b0, b1\n#endif\n};\n"),
    ("__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {\n",
     X2S_CONSUME + WT_INV + "#ifndef VMS_LB\n#define VMS_LB 1\n#endif\n#if VMS_LB\n"
     "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {\n#else\n"
     "__global__ void vms_kernel(Params p) {   // mb VMS_LB 0: no launch bounds, so -maxrregcount caps the registers\n#endif\n"),
    # VMS_WT routing: each lane's (expert's) job weights summed in closed form and scanned with the job counts (one shuffle
    # more per round), per-job prefix written in the lane's own job loop; lanes 0-3 of warp 0 then compute the CTA's four
    # range boundaries in parallel into S.rng before the existing __syncthreads (mb_6's version: +1.2 us of routing)
    ("    int inc = tot, nj = (tot + VMS_SKT - 1) / VMS_SKT, jinc = nj;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n"
     "      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);\n"
     "      if (lane >= d) { inc += a; jinc += b; }\n    }\n",
     "    int inc = tot, nj = (tot + VMS_SKT - 1) / VMS_SKT, jinc = nj;\n#if VMS_WT\n"
     "    const int wsum = nj ? (nj - 1) * (VMS_SKT > 8 ? VMS_WT : 4) + ((tot - VMS_SKT * (nj - 1)) > 8 ? VMS_WT : 4) : 0;\n"
     "    int winc = wsum;\n#endif\n"
     "#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n"
     "      const int a = __shfl_up_sync(0xffffffffu, inc, d), b = __shfl_up_sync(0xffffffffu, jinc, d);\n#if VMS_WT\n"
     "      const int cw = __shfl_up_sync(0xffffffffu, winc, d);\n      if (lane >= d) winc += cw;\n#endif\n"
     "      if (lane >= d) { inc += a; jinc += b; }\n    }\n"),
    ("    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + VMS_SKT * k; S.jC[joff + k] = min(VMS_SKT, tot - VMS_SKT * k); }\n",
     "    for (int k = 0; k < nj; ++k) { S.jE[joff + k] = lane; S.jS[joff + k] = off + VMS_SKT * k; S.jC[joff + k] = min(VMS_SKT, tot - VMS_SKT * k); }\n"
     "#if VMS_WT\n    { int wr = winc - wsum; for (int k = 0; k < nj; ++k) { wr += (min(VMS_SKT, tot - VMS_SKT * k) > 8) ? VMS_WT : 4; "
     "S.jW[joff + k + 1] = wr; } if (lane == 0) S.jW[0] = 0; }\n#endif\n"),
    ("    if (lane == 31) S.njobs = jinc;\n    __syncwarp();\n",
     "    if (lane == 31) S.njobs = jinc;\n    __syncwarp();\n#if VMS_WT\n"
     "    if (lane < 4) S.rng[lane] = wt_inv(S, c + (lane & 1), (lane < 2) ? RBA : RBB);   // a0, a1, b0, b1\n#endif\n"),
    ("  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;\n  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;\n"
     "  const int b0 = t0 - a0;\n  const int nA = a1 - a0, N = t1 - t0;\n",
     "#if VMS_WT\n  const int a0 = S.rng[0], a1 = S.rng[1];\n"
     "  const int b0 = S.rng[2], b1 = S.rng[3];\n  const int nA = a1 - a0, N = nA + (b1 - b0);\n"
     "  (void)NN;\n#else\n"
     "  const int a0 = c * NA / G, a1 = (c + 1) * NA / G;\n  const int t0 = c * NN / G, t1 = (c + 1) * NN / G;\n"
     "  const int b0 = t0 - a0;\n  const int nA = a1 - a0, N = t1 - t0;\n#endif\n"),
    ("#if VMS_SKT == 16\n    uint4 x2[2 * NCH];\n    bool two = false;\n#endif\n",
     "#if VMS_SKT == 16 && VMS_X2S\n    uint4* x2p = &S.x2s[warp][0][lane];\n#define X2(k) x2p[(k) * 32]\n    bool two = false;\n"
     "#elif VMS_SKT == 16\n    uint4 x2[2 * NCH];\n#define X2(k) x2[k]\n    bool two = false;\n#endif\n"),
    ("              x2[2 * cc] = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);\n"
     "              x2[2 * cc + 1] = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);\n",
     "              X2(2 * cc) = PB ? __ldcg(src + cc * 8) : __ldg(src + cc * 8);\n"
     "              X2(2 * cc + 1) = PB ? __ldcg(src + cc * 8 + 1) : __ldg(src + cc * 8 + 1);\n"),
    ("            for (int cc = 0; cc < 2 * NCH; ++cc) x2[cc] = make_uint4(0u, 0u, 0u, 0u);\n          }\n        }\n#endif\n",
     "            for (int cc = 0; cc < 2 * NCH; ++cc) X2(cc) = make_uint4(0u, 0u, 0u, 0u);\n          }\n        }\n#endif\n"),
    ("#if VMS_SKT == 16\n      float acc2[4][4];\n",
     "#if VMS_SKT == 16 && !VMS_X2S\n      float acc2[4][4];\n"),
    ("#elif !(VMS_FL & 2)\n#pragma unroll\n      for (int cc = 0; cc < NCH; ++cc) {\n",
     "#elif !(VMS_FL & 2)\n#if VMS_X2S\n      if (two) consume2s(slot, sc, xr, x2p, acc, k16, warp, lane);\n      else {\n#endif\n"
     "#pragma unroll\n      for (int cc = 0; cc < NCH; ++cc) {\n"),
    ("          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);\n        }\n      }\n#endif\n",
     "          mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);\n        }\n      }\n"
     "#if VMS_X2S\n      }\n#endif\n#endif\n"),
    ("      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];\n",
     "      o.w = acc[0][3] + acc[1][3] + acc[2][3] + acc[3][3];\n#if VMS_X2S\n"
     "      if (two) { o.x = acc[0][0] + acc[1][0]; o.y = acc[0][1] + acc[1][1]; o.z = acc[0][2] + acc[1][2]; o.w = acc[0][3] + acc[1][3]; }\n"
     "#endif\n"),
    ("        o2.x = acc2[0][0] + acc2[1][0] + acc2[2][0] + acc2[3][0];\n"
     "        o2.y = acc2[0][1] + acc2[1][1] + acc2[2][1] + acc2[3][1];\n"
     "        o2.z = acc2[0][2] + acc2[1][2] + acc2[2][2] + acc2[3][2];\n"
     "        o2.w = acc2[0][3] + acc2[1][3] + acc2[2][3] + acc2[3][3];\n",
     "#if VMS_X2S\n        o2.x = acc[2][0] + acc[3][0]; o2.y = acc[2][1] + acc[3][1]; o2.z = acc[2][2] + acc[3][2]; o2.w = acc[2][3] + acc[3][3];\n"
     "#else\n"
     "        o2.x = acc2[0][0] + acc2[1][0] + acc2[2][0] + acc2[3][0];\n"
     "        o2.y = acc2[0][1] + acc2[1][1] + acc2[2][1] + acc2[3][1];\n"
     "        o2.z = acc2[0][2] + acc2[1][2] + acc2[2][2] + acc2[3][2];\n"
     "        o2.w = acc2[0][3] + acc2[1][3] + acc2[2][3] + acc2[3][3];\n#endif\n"),
)

# VML: runtime flags bit 10 = experts above 64 tokens as 64-token chunks + a remainder (else balanced jobs)
FIN_IN = r'''// mb bit 12: the finalize inside vm_main after phase B (vm_fin not launched). Grid barrier on gbar[0] (arrivals, monotonic
// across launches) against (gen + 1) x G, gen = gbar[1] read by each CTA's thread 0 before it arrives (CTA 0 bumps it only
// after every CTA arrived); CTA 0 then resets xready / doneA / cnt (every CTA is past both phases). Tokens round-robin over
// CTAs, two per pass (consumer threads 0-255 / 256-511, named barriers 9 / 10), vm_fin's numerics with 2 routes in flight.
__device__ __forceinline__ void fin_in(const Args& A, uint8_t* scr) {
const int tid = threadIdx.x, half = tid >> 8, ht = tid & 255, warp = ht >> 5, lane = tid & 31;
int* s_r = reinterpret_cast<int*>(scr + half * 1024);
int* s_r8 = s_r + 8;
float* s_w = reinterpret_cast<float*>(s_r + 16);
float* s_w8 = s_w + 8;
int* s_n = s_r + 32;
float* s_ss = reinterpret_cast<float*>(s_r + 64);   // [2 par][8 warps][2 routes]
asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
bar_sync(8, 32 * NCW);
if (tid == 0) {
const int gen = ld_acquire(A.gbar + 1);
__threadfence();
atomicAdd(A.gbar, 1);
const int target = (gen + 1) * (int)gridDim.x;
const long long c0 = clock64();
while (ld_acquire(A.gbar) < target) {
__nanosleep(32);
if (clock64() - c0 > WAIT_BUDGET) { atomicExch(A.err, 3); break; }
}
}
bar_sync(8, 32 * NCW);
if (blockIdx.x == 0) {
for (int i = tid; i < JMAX; i += 32 * NCW) A.xready[i] = 0;
for (int i = tid; i < JMAX * 16; i += 32 * NCW) A.doneA[i] = 0;
for (int i = tid; i < 2 * SLOTS * 4; i += 32 * NCW) A.cnt[i] = 0;
if (tid == 0) atomicAdd(A.gbar + 1, 1);
}
float4 w4[4];
#pragma unroll
for (int m = 0; m < 4; ++m) w4[m] = __ldg(reinterpret_cast<const float4*>(A.nw + 1024 * m + 4 * ht));
const float4 z4 = make_float4(0.f, 0.f, 0.f, 0.f);
for (int tb = 2 * blockIdx.x; tb < A.M; tb += 2 * gridDim.x) {
const int t = tb + half;
const bool ok = t < A.M;
if (ht < 8) { s_r8[ht] = ok ? __ldcg(A.rmap + t * 8 + ht) : -1; s_w8[ht] = ok ? __ldg(A.wts + t * 8 + ht) : 0.f; }
bar_sync(9 + half, 256);
if (ht == 0) {
int n = 0;
#pragma unroll
for (int k = 0; k < 8; ++k) {
const int r = s_r8[k];
if (r >= 0) { s_r[n] = r; s_w[n] = s_w8[k]; ++n; }
}
*s_n = n;
}
bar_sync(9 + half, 256);
const int nloc = *s_n;
float a[4][4];
#pragma unroll
for (int m = 0; m < 4; ++m) { a[m][0] = 0.f; a[m][1] = 0.f; a[m][2] = 0.f; a[m][3] = 0.f; }
int par = 0;
for (int c0 = 0; c0 < nloc; c0 += 2, par ^= 1) {
float4 v[2][4];
#pragma unroll
for (int jj = 0; jj < 2; ++jj) {
const int r = (c0 + jj < nloc) ? s_r[c0 + jj] : -1;
#pragma unroll
for (int m = 0; m < 4; ++m)
v[jj][m] = (r >= 0) ? __ldcg(reinterpret_cast<const float4*>(A.y + (size_t)r * KDIM + 1024 * m + 4 * ht)) : z4;
}
float ss[2];
#pragma unroll
for (int jj = 0; jj < 2; ++jj) {
ss[jj] = 0.f;
#pragma unroll
for (int m = 0; m < 4; ++m) {
float f;
f = bf16r(v[jj][m].x) * w4[m].x; ss[jj] += f * f; f = bf16r(v[jj][m].y) * w4[m].y; ss[jj] += f * f;
f = bf16r(v[jj][m].z) * w4[m].z; ss[jj] += f * f; f = bf16r(v[jj][m].w) * w4[m].w; ss[jj] += f * f;
}
#pragma unroll
for (int o = 16; o > 0; o >>= 1) ss[jj] += __shfl_xor_sync(0xffffffffu, ss[jj], o);
}
if (lane == 0) { s_ss[(par * 8 + warp) * 2] = ss[0]; s_ss[(par * 8 + warp) * 2 + 1] = ss[1]; }
bar_sync(9 + half, 256);
#pragma unroll
for (int jj = 0; jj < 2; ++jj) {
if (c0 + jj < nloc) {
float sm = 0.f;
for (int w = 0; w < 8; ++w) sm += s_ss[(par * 8 + w) * 2 + jj];
const float fr = rsqrtf(sm * (1.f / 4096.f) + 1e-8f) * s_w[c0 + jj];
#pragma unroll
for (int m = 0; m < 4; ++m) {
a[m][0] += bf16r(v[jj][m].x) * w4[m].x * fr; a[m][1] += bf16r(v[jj][m].y) * w4[m].y * fr;
a[m][2] += bf16r(v[jj][m].z) * w4[m].z * fr; a[m][3] += bf16r(v[jj][m].w) * w4[m].w * fr;
}
}
}
}
if (ok) {
#pragma unroll
for (int m = 0; m < 4; ++m) {
__nv_bfloat162 o0 = __floats2bfloat162_rn(a[m][0], a[m][1]), o1 = __floats2bfloat162_rn(a[m][2], a[m][3]);
uint2 o;
o.x = *reinterpret_cast<uint32_t*>(&o0); o.y = *reinterpret_cast<uint32_t*>(&o1);
*reinterpret_cast<uint2*>(A.out + (size_t)t * KDIM + 1024 * m + 4 * ht) = o;
}
}
bar_sync(9 + half, 256);
}
}
'''

VML_PATCHES = (   # the lab block's CUDA is dedented
    ("\nif (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }\n",
     "\nif ((A.flags & 8192) && nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));   // bit 13: the loop's closed form (same jobs; tools/gen_k3_mb.py asserts it)\n"
     "else if (nj && !(A.flags & 1024)) { while (base > 1 && (base - 1) * nj >= tot) --base; }   // bit 10: 64-token chunks\n"),
    # bit 11: every thread of the grid prefetches its share of the x rows (128-B lines) into L2 at entry, while HBM idles
    # through the routing prologue (the weight stream starts only after routing)
    ('\nif (A.flags & 256) asm volatile("griddepcontrol.wait;\\n" ::: "memory");   // PDL: ids / x come from the previous kernel\n',
     '\nif (A.flags & 256) asm volatile("griddepcontrol.wait;\\n" ::: "memory");   // PDL: ids / x come from the previous kernel\n'
     'if (A.flags & 2048) {   // bit 11: x rows -> L2 by every thread, one 128-B line each, before routing\n'
     'const uint32_t nl = (uint32_t)A.M * (KDIM * 2 / 128);\n'
     'for (uint32_t l = blockIdx.x * NTH + tid; l < nl; l += gridDim.x * NTH)\n'
     'asm volatile("prefetch.global.L2 [%0];\\n" :: "l"(reinterpret_cast<const uint8_t*>(A.x) + (size_t)l * 128));\n'
     '}\n'),
    # bit 12: the finalize inside vm_main (no vm_fin launch): grid barrier, tokens round-robin over CTAs, two at a time
    ("__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n", FIN_IN + "__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n"),
    ("if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 1 + q] = gtime();\n}\n",
     "if (wp == 0 && lane == 0 && A.marks) A.marks[(size_t)blockIdx.x * 32 + 1 + q] = gtime();\n"
     "#ifndef VM_FININ\n#define VM_FININ 1\n#endif\n#if VM_FININ\nif (A.flags & 4096) fin_in(A, smem + NS * US);\n#endif\n}\n"),
    ("if ((mask & 2) && M > 0) {\n", "if ((mask & 2) && M > 0 && !(VM_FININ && (V[5] & 4096))) {\n"),
    # VM_LB 0: vm_main without __launch_bounds__ (ptxas then honours -maxrregcount, e.g. 112 = 64K / 576 rounded to 8,
    # instead of its 96-register choice for 576 threads)
    ("__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n",
     "#ifndef VM_LB\n#define VM_LB 1\n#endif\n#if VM_LB\n__global__ void __launch_bounds__(NTH, 1) vm_main(const Args A) {\n#else\n"
     "__global__ void vm_main(const Args A) {\n#endif\n"),
)

K2A_NAMES = ("K2A_BITPOS", "K2A31_BITPOS3", "_K2A_GRID", "_k2a_reencode_scales", "_k2a_tile_row_perm", "_k2a31_pack_words",
             "_k2a31_prepack_expert", "_k2a33_stage_repack", "_k2a_dequant", "_k2a_rel_err")


def _lab_strings(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            out[n.targets[0].id] = ast.literal_eval(n.value)
    assert set(out) == set(names), set(names) - set(out)
    return out


def _segments(path, names):
    src = path.read_text(encoding="utf-8")
    lines = src.split("\n")
    tree = ast.parse(src)
    got = {}
    for n in tree.body:
        nm = getattr(n, "name", None)
        if nm is None and isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
        if nm in names:
            got[nm] = "\n".join(lines[n.lineno - 1:n.end_lineno])
    assert set(got) == set(names), set(names) - set(got)
    return "\n\n".join(got[nm] for nm in names)


def _layer_vml_cost():
    t = LAYER.read_text(encoding="utf-8")
    key = "\nVML_COST = "
    i = t.index(key) + len(key)
    return ast.literal_eval(t[i:t.index("\n", i)])


def _layer_profiles():
    """real layer routings (sorted per-expert token counts), by T, distinct profiles in file order"""
    by = {}
    for f in ("box_out/vms/run.json", "box_out/vms6/run.json", "box_out/m32/run.json", "box_out/sk/run.json"):
        p = ROOT / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["routings"]:
            T = r.get("T") or r.get("bs")
            c = tuple(int(v) for v in r["counts"])
            if T in (1, 8, 16, 32, 128, 256) and c not in by.setdefault(T, []) and sum(c) <= 8 * T:
                by[T].append(c)
    out = {}
    for T, v in by.items():   # spread over router seeds: every 2nd entry first
        pick = v[::2][:4] + v[1::2][:4]
        out[T] = [list(c) for c in pick[:6]]
    return out


GLUE = r'''

# ================================================================ mb glue (tools/mb_gen.py)
MB_N = @N@
MB_BUILDS = @BUILDS@
MB_ARMS = @ARMS@
MB_LAYER_COST = @LCOST@
MB_LAYER_PROFILES = @LPROF@
MB_BUCKETS = ((1, 4, 1), (5, 12, 8), (13, 24, 16), (25, 48, 32), (49, 192, 128), (193, 320, 256))
MB_CASES = {1: ((1, 1, "normal"), (1, 2, "skewed"), (4, 3, "normal")),
            8: ((5, 4, "normal"), (8, 5, "normal"), (8, 6, "skewed"), (12, 7, "skewed")),
            16: ((13, 8, "normal"), (16, 9, "normal"), (16, 10, "skewed"), (16, 11, "hot"), (24, 12, "skewed")),
            32: ((25, 13, "normal"), (32, 14, "normal"), (32, 15, "skewed"), (32, 16, "hot"), (33, 17, "normal"),
                 (48, 18, "skewed")),
            128: ((49, 19, "normal"), (128, 20, "normal"), (128, 21, "skewed"), (192, 22, "skewed")),
            256: ((193, 23, "normal"), (256, 24, "normal"), (256, 25, "hot"), (257, 26, "normal"), (320, 27, "skewed"))}
MB_SEEDS = 6         # board-like timing routings per bucket (T <= 16)
MB_SEEDS_L = 16      # ... for T >= 32 (routing spread there is +-15 %)
MB_TUNE_N = 24       # routings per tuned-table objective (separate seeds from the timing set)
MB_REPS = 16         # cold-L2 graph replays per (arm, routing)
MB_FITS = 3          # routings per cost fit
MB_EPS = 1e-8
_MB = {"mods": {}, "berr": {}, "bsec": {}, "route": {}, "fb": {}, "costs": {}, "scr": {}, "t0": 0.0}


def _mb_bucket(m):
    for lo, hi, rep in MB_BUCKETS:
        if lo <= m <= hi:
            return rep
    return None


# ---------------------------------------------------------------- builds (threads: nvcc runs in parallel)
def _mb_build_vms(ns, maxt, fl, G, pad=0, wt=0, extra=()):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", f"-DVMS_NS={ns}", f"-DVMS_MAXT={maxt}", f"-DVMS_FL={fl}", f"-DVMS_PAD={pad}", f"-DVMS_WT={wt}"] + list(extra) + [
             "-Xptxas=-warn-spills",
             "-gencode=arch=compute_90a,code=sm_90a"]
    tag = hashlib.sha1((MB_VMS_CUDA + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"mbs_{tag}", cpp_sources=MB_VMS_CPP, cuda_sources=MB_VMS_CUDA, functions=["run", "attrs"],
                       extra_cuda_cflags=flags)


def _mb_build_one(name, G):
    import threading  # noqa: F401
    t1 = time.time()
    try:
        b = MB_BUILDS[name]
        if b[0] == "vms":
            _MB["mods"][name] = _mb_build_vms(b[1], b[2], b[3], G, b[4] if len(b) > 4 else 0, b[5] if len(b) > 5 else 0,
                                              b[6] if len(b) > 6 else ())
        else:
            _MB["mods"][name], _ = _vml_build(defines=b[1])
    except Exception as ex:  # noqa: BLE001
        _MB["berr"][name] = repr(ex)[-300:]
    _MB["bsec"][name] = time.time() - t1


def _mb_start_builds(G):
    import threading
    ths = []
    for name in MB_BUILDS:
        th = threading.Thread(target=_mb_build_one, args=(name, G), name=f"mb-{name}", daemon=True)
        th.start()
        ths.append(th)
    return ths


# ---------------------------------------------------------------- weights
def _mb_vms_pack_words(cu):
    nu = cu.shape[0]
    x = cu.view(nu, 2, 8, 8, 8, 4, 4, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).contiguous()
    x = x.view(nu, 8, 8, 8, 4, 4, 4, 2).to(torch.int64)
    word = torch.zeros(x.shape[:6], dtype=torch.int64, device=cu.device)
    for r in range(4):
        for h in range(2):
            cd = x[..., r, h]
            for b in range(4):
                word |= ((cd >> b) & 1) << MB_VMS_POS[r][h][b]
    word = torch.where(word >= 2 ** 31, word - 2 ** 32, word).to(torch.int32)
    return word.view(nu, 8192)


def _mb_vms_prepack_expert(w, sf, gate_up):
    n = w.shape[0]
    rows = torch.arange(n, device=w.device)
    if gate_up:
        rb, i = rows // 16, rows % 16
        rows = torch.where(i < 8, 8 * rb + i, n // 2 + 8 * rb + (i - 8))
    wr = w[rows]
    cu = torch.stack((wr & 15, wr >> 4), dim=-1).reshape(n // 16, 16, 4096)
    sb = sf.view(torch.uint8)[rows].view(n // 16, 2, 8, 8, 8, 4).permute(0, 3, 2, 5, 4, 1).contiguous().view(n // 16, 4096)
    words = _mb_vms_pack_words(cu).view(torch.uint8).view(n // 16, 32768)
    return torch.cat((words, sb), dim=1).contiguous()


def _mb_prepack(w, dev):
    """experts [0, 32): VMS units (scale bytes must sit in 0x08..0x7E for its PRMT decode) and VML / K2A v3.1 stages"""
    E = 32
    pk = {}
    b13, b2 = w["w13_sf"][:E].view(torch.uint8), w["w2_sf"][:E].view(torch.uint8)
    if bool(((b13 >= 8) & (b13 <= 0x7E)).all()) and bool(((b2 >= 8) & (b2 <= 0x7E)).all()):
        wA = torch.empty((E, 512, 36864), dtype=torch.uint8, device=dev)
        wB = torch.empty((E, 256, 36864), dtype=torch.uint8, device=dev)
        for e in range(E):
            wA[e] = _mb_vms_prepack_expert(w["w13"][e], w["w13_sf"][e], True)
            wB[e] = _mb_vms_prepack_expert(w["w2"][e], w["w2_sf"][e], False)
        two62 = float(2 ** 62)
        pk["vms"] = {"wA": wA, "wB": wB, "gsA": (w["w13_gs"][:E].float() * two62).contiguous(),
                     "gsB": (w["w2_gs"][:E].float() * two62).contiguous(), "nw": (1.0 + w["post_norm_w"].float()).contiguous()}
    else:
        print("[mb] VMS off: block-scale bytes outside 0x08..0x7E", flush=True)
    ws13 = torch.empty((E, 64, 16, 18432), dtype=torch.uint8, device=dev)
    ws2 = torch.empty((E, 32, 16, 18432), dtype=torch.uint8, device=dev)
    for e in range(E):
        w13q, s13q = _k2a31_prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=K2A31_BITPOS3)
        w2q, s2q = _k2a31_prepack_expert(w["w2"][e], w["w2_sf"][e], gate_up=False, bitpos=K2A31_BITPOS3)
        ws13[e] = _k2a33_stage_repack(w13q, s13q)
        ws2[e] = _k2a33_stage_repack(w2q, s2q)
    pk["vml"] = {"ws13": ws13, "ws2": ws2, "gsc13": (w["w13_gs"][:E].float() * 256.0).contiguous(),
                 "gsc2": (w["w2_gs"][:E].float() * 256.0).contiguous(),
                 "nw": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()}
    torch.cuda.synchronize()
    return pk


def _mb_scratch(name, dev, G):
    b = MB_BUILDS[name]
    if b[0] == "vms":
        mp = 8 * b[2]
        return {"inter": torch.zeros((mp, 4096), dtype=torch.bfloat16, device=dev),
                "ybuf": torch.zeros((mp, 4096), dtype=torch.float32, device=dev),
                "ss": torch.zeros((2 * mp,), dtype=torch.float32, device=dev),
                "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
                "nomarks": torch.empty((0,), dtype=torch.int64, device=dev), "maxt": b[2]}
    sc = _vml_alloc(dev, G)
    sc["maxt"] = _vml_MMAX
    return sc


def _mb_reset(name):
    sc = _MB["scr"][name]
    for k, v in sc.items():
        if isinstance(v, torch.Tensor) and k not in ("nomark", "nomarks", "xA", "xB", "y", "inter", "ybuf"):
            v.zero_()
    torch.cuda.synchronize()


# ---------------------------------------------------------------- calls
def _mb_arm(spec):
    p = spec.split(":")
    return {"spec": spec, "b": p[0], "kind": MB_BUILDS[p[0]][0], "flags": int(p[1]) if len(p) > 1 else 0,
            "ck": p[2] if len(p) > 2 else None}


def _mb_cost(arm, rep):
    if arm["kind"] != "vml":
        return None
    ck = arm["ck"]
    if ck in (None, "d"):
        return None
    if ck == "L":
        return MB_LAYER_COST.get(rep)
    return _MB["costs"].get((rep, arm["b"], arm["flags"], ck))


def _mb_call(arm, st, x, ids, tw, out, rep, marks=None, cost=None):
    sc = _MB["scr"][arm["b"]]
    mod = _MB["mods"][arm["b"]]
    m = x.shape[0]
    if arm["kind"] == "vms":
        pk = st["pk"]["vms"]
        mod.run(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"],
                out, sc["nomarks"] if marks is None else marks, m, MB_EPS, 0, 0, 0)
        return out
    return _vml_call(mod, st["pk"]["vml"], sc, x, ids, tw, 0, marks=marks, out=out, flags=arm["flags"],
                     cost=_mb_cost(arm, rep) if cost is None else cost)


def _mb_err(arm):
    sc = _MB["scr"][arm["b"]]
    return int(sc["ctr"][2]) if arm["kind"] == "vms" else int(sc["err"][0])


def _mb_torch(data, st):
    """FP32 torch fallback (correct, slow): only when no kernel arm passed for this shape"""
    w = st["w"]
    x = data["hidden_states"].float()
    ids = data["topk_ids"].long()
    lo = int(data["config"]["local_start"])
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in torch.unique(ids).tolist():
        if not lo <= e < lo + 32:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok] @ _k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = (inter @ _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T) * (1.0 + w["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + MB_EPS)
        out.index_add_(0, tok, y * data["topk_weights"][tok, slot].float().unsqueeze(1))
    return out.to(torch.bfloat16)


def kernel_fn(data, state):
    x = data["hidden_states"]
    m = x.shape[0]
    rep = _mb_bucket(m)
    arm = _MB["route"].get(rep)
    if arm is not None and int(data["config"]["local_start"]) == 0:
        if m > _MB["scr"][arm["b"]]["maxt"]:
            arm = _MB["fb"].get(rep)
        if arm is not None:
            x = x.contiguous()
            if x.data_ptr() % 16:
                x = x.clone()
            out = torch.empty((m, 4096), dtype=torch.bfloat16, device=x.device)
            return _mb_call(arm, state, x, data["topk_ids"].to(torch.int32).contiguous(),
                            data["topk_weights"].to(torch.float32).contiguous(), out, rep)
    return _mb_torch(data, state)


# ---------------------------------------------------------------- inputs (synthetic; prepare only)
def _mb_board_input(bs, seed, dev, dist="normal"):
    """generate_input's routing distribution (normal: uniform + 0.03 x Student-t popularity; skewed: + rand^6 x 4;
    hot: 8 experts take every token), busiest GPU relabelled to experts 0-31, our own seeds"""
    gen = torch.Generator(device=dev).manual_seed(7919 * seed + 13 * bs + 1)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    scores = torch.rand((bs, 256), generator=gen, device=dev)
    if dist == "normal":
        z = torch.randn((256,), generator=gen, device=dev)
        u = torch.rand((256,), generator=gen, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    elif dist == "skewed":
        scores = scores + torch.rand((256,), generator=gen, device=dev).pow(6) * 4
    else:
        scores[:, :8] += 10.0
    top = scores.topk(8, dim=-1)
    ids = top.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32).contiguous(),
            "topk_weights": (top.values / top.values.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "local_start": 0, "local_count": 32}}


def _mb_layer_input(bs, counts, seed, dev):
    """a routing with the given per-expert token counts (a real layer profile), experts placed at random local ids"""
    import random
    rnd = random.Random(1009 * seed + bs)
    loc = list(range(32))
    rnd.shuffle(loc)
    free = [8] * bs
    ids = [[-1] * 8 for _ in range(bs)]
    for k, c in enumerate(sorted(counts, reverse=True)):
        e = loc[k]
        order = sorted(range(bs), key=lambda t: (-free[t], rnd.random()))[:c]
        for t in order:
            ids[t][8 - free[t]] = e
            free[t] -= 1
    for t in range(bs):
        rem = rnd.sample(range(32, 256), 8)
        for s in range(8):
            if ids[t][s] < 0:
                ids[t][s] = rem[s]
    gen = torch.Generator(device=dev).manual_seed(4241 * seed + bs)
    x = torch.randn((bs, 4096), generator=gen, device=dev).to(torch.bfloat16)
    tw = torch.rand((bs, 8), generator=gen, device=dev) + 0.05
    return {"hidden_states": x, "topk_ids": torch.tensor(ids, dtype=torch.int32, device=dev),
            "topk_weights": (tw / tw.sum(-1, keepdim=True)).float().contiguous(),
            "config": {"bs": bs, "local_start": 0, "local_count": 32}}


class _MbRef:
    """FP32 reference with dequantized expert weights cached for the self-tests"""
    def __init__(self, w):
        self.w, self.c = w, {}

    def deq(self, e):
        if e not in self.c:
            w = self.w
            self.c[e] = (_k2a_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                         _k2a_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.c[e]

    def __call__(self, d):
        w = self.w
        x = d["hidden_states"].float()
        ids = d["topk_ids"].long()
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in torch.unique(ids).tolist():
            if not 0 <= e < 32:
                continue
            tok, slot = (ids == e).nonzero(as_tuple=True)
            a, b = self.deq(e)
            h = x[tok] @ a.T
            inter = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
            y = (inter @ b.T) * (1.0 + w["post_norm_w"].float())
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + MB_EPS)
            out.index_add_(0, tok, y * d["topk_weights"][tok, slot].float().unsqueeze(1))
        return out.to(torch.bfloat16)


# ---------------------------------------------------------------- VML cost fit from per-CTA marks (public t_fit2 idea)
def _mb_fit(st, arm, rep, ds, base, G, flush, sink, stream):
    marks = torch.zeros((G * 32,), dtype=torch.int64, device=st["dev"])
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    res = []
    for pb in (0, 1):
        X, Y = [], []
        for d in ds:
            marks.zero_()
            with torch.cuda.stream(stream):
                torch.sum(flush, dim=0, out=sink)
                _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:d["hidden_states"].shape[0]],
                         rep, marks=marks, cost=base)
            torch.cuda.synchronize()
            mv = marks.view(G, 32).cpu().double()
            t0 = mv[:, 0].min()
            aend = mv[:, 6] - t0
            end = mv[:, 1:5].max(1).values - t0
            X.append(torch.cat([torch.ones(G, 1, dtype=torch.float64), mv[:, 8 + 8 * pb:16 + 8 * pb]], 1))
            Y.append(aend if pb == 0 else end - aend)
        X, Y = torch.cat(X), torch.cat(Y) / 1e3
        seen = (X[:, 1:] > 0).any(0)
        ks = [k for k in range(8) if bool(seen[k])]
        coef = torch.linalg.lstsq(X[:, [0] + [1 + k for k in ks]], Y.unsqueeze(1)).solution.squeeze(1)
        c = {k: float(coef[i + 1]) for i, k in enumerate(ks)}
        ref = max(c, key=lambda k: float(X[:, 1 + k].sum())) if c else 0
        b0 = list(_vml_COST) if base is None else list(base[8 * pb:8 * pb + 8])
        tab = list(b0)
        if c and c[ref] > 0:
            for k, v in c.items():
                if v > 0:
                    tab[k] = int(min(255, max(16, round(v / c[ref] * b0[ref]))))
        res += tab
    return res


def _mb_eager_us(st, arm, rep, ds, cost, flush, sink, stream, reps=3):
    """mean over routings of the median of `reps` cold-L2 eager calls (the flush keeps the GPU busy while Python launches)"""
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    evs = []
    with torch.cuda.stream(stream):
        for _ in range(reps):
            for i, d in enumerate(ds):
                torch.sum(flush, dim=0, out=sink)
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:d["hidden_states"].shape[0]],
                         rep, cost=cost)
                e1.record()
                evs.append((i, e0, e1))
    torch.cuda.synchronize()
    per = {}
    for i, e0, e1 in evs:
        per.setdefault(i, []).append(e0.elapsed_time(e1) * 1e3)
    return sum(sorted(v)[len(v) // 2] for v in per.values()) / len(per)


def _mb_ntiles(ds, flags):
    """cost-table entries (A: nt - 1, B: 8 + nt - 1) that the jobs of these routings use (VML job rule, bit 10 = chunks)"""
    ks = set()
    for d in ds:
        ids = d["topk_ids"].long()
        c = torch.bincount(ids[(ids >= 0) & (ids < 32)].flatten(), minlength=32).tolist()
        for tot in c:
            if tot == 0:
                continue
            nj = (tot + 63) // 64
            base = 64 if (flags & 1024) else -(-tot // nj)
            for k in range(nj):
                nt = (min(base, tot - k * base) + 7) // 8
                ks.update((nt - 1, 8 + nt - 1))
    return sorted(ks)


def _mb_tune(st, arm, rep, ds, start, seen, flush, sink, stream, budget_s=12.0):
    """coordinate descent on the stage-cost table entries of the n-tiles present (x0.92 / x1.08 steps, keep a step only if
    the mean cold-L2 time over `ds` drops by > 0.3 %), inside a time budget; returns (table, us start, us end, steps kept)"""
    t1 = time.time()
    cur = list(start)
    best = _mb_eager_us(st, arm, rep, ds, cur, flush, sink, stream, reps=1)
    b0, kept = best, 0
    for _ in range(2):
        for k in seen:
            for f in (0.92, 1.08):
                if time.time() - t1 > budget_s:
                    return cur, b0, best, kept
                cand = list(cur)
                cand[k] = int(min(255, max(16, round(cur[k] * f))))
                if cand[k] == cur[k]:
                    continue
                v = _mb_eager_us(st, arm, rep, ds, cand, flush, sink, stream, reps=1)
                if v < best * 0.997:
                    cur, best, kept = cand, v, kept + 1
                    break
    return cur, b0, best, kept


def _mb_marks(st, arm, rep, ds, G, flush, sink, stream):
    """per-CTA phase marks of one arm on cold-L2 eager calls -> one compact line (p50 / max over CTAs and routings, us after
    the earliest CTA entry). VML: route, gather (x ready), A end (WG 0), end (last WG), B doneA wait, A xready wait (both
    durations, x-producer warp). VMS: routed, first unit landed, first B wait start / end, compute end, helper A end / end,
    grid barrier passed, final pass done."""
    vms = arm["kind"] == "vms"
    W = 64 if vms else 32
    marks = torch.zeros((G * W,), dtype=torch.int64, device=st["dev"])
    out = torch.empty((max(d["hidden_states"].shape[0] for d in ds), 4096), dtype=torch.bfloat16, device=st["dev"])
    per = []   # one dict per routing: mark name -> per-CTA values (us after that routing's earliest CTA entry)
    for d in ds:
        m = d["hidden_states"].shape[0]
        if m > _MB["scr"][arm["b"]]["maxt"]:
            continue
        marks.zero_()
        with torch.cuda.stream(stream):
            torch.sum(flush, dim=0, out=sink)
            _mb_call(arm, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out[:m], rep, marks=marks)
        torch.cuda.synchronize()
        mv = marks.view(G, W).cpu().double()
        t0 = float(mv[:, 0].min())
        acc = {}
        if vms:
            cols = {"route": 1, "first": 2, "bw0": 3, "bw1": 4, "comp": 5, "hA": 6, "hend": 7, "bar": 8, "fin": 9}
            for k, c in cols.items():
                acc[k] = [(float(x) - t0) / 1e3 for x in mv[:, c] if x > 0]
        else:
            for k, c in (("route", 7), ("gather", 5), ("Aend", 6)):
                acc[k] = [(float(x) - t0) / 1e3 for x in mv[:, c] if x > 0]
            acc["end"] = [(float(x) - t0) / 1e3 for x in mv[:, 1:5].max(1).values]
            acc["Bwait"] = [float(x) / 1e3 for x in mv[:, 24]]
            acc["Xwait"] = [float(x) / 1e3 for x in mv[:, 26]]
        per.append(acc)
    _mb_reset(arm["b"])

    def q(k):
        p50, mx = [], []
        for acc in per:
            v = sorted(acc.get(k, []))
            if v:
                p50.append(v[len(v) // 2])
                mx.append(v[-1])
        return f"{sum(p50) / len(p50):.1f}/{sum(mx) / len(mx):.1f}" if p50 else "-"
    return " ".join(f"{k} {q(k)}" for k in (per[0] if per else {}))


# ---------------------------------------------------------------- prepare
def _mb_rel(o, ref):
    tot, row = _k2a_rel_err(o, ref)
    return tot, row


def prepare(weights):
    _MB["t0"] = t0 = time.time()
    dev = weights["w13"].device
    G = torch.cuda.get_device_properties(dev).multi_processor_count
    ths = _mb_start_builds(G)
    w = {k: weights[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_norm_w")}
    st = {"dev": dev, "G": G, "w": w, "pk": {}}
    try:
        st["pk"] = _mb_prepack(w, dev)
    except Exception as ex:  # noqa: BLE001
        print(f"[mb] prepack raised {ex!r}"[:300], flush=True)
    tp = time.time() - t0
    for th in ths:
        th.join()
    bl = " ".join(f"{k}={v:.0f}s" for k, v in sorted(_MB["bsec"].items()))
    print(f"[mb] v{MB_N}: G={G} prepack {tp:.0f}s builds {bl} errors {_MB['berr'] or 'none'}", flush=True)
    for name, mod in _MB["mods"].items():
        if MB_BUILDS[name][0] == "vms":
            try:
                print(f"[mb] build {name} {MB_BUILDS[name]}: regs/local/smem/G/NS {list(mod.attrs())}", flush=True)
            except Exception:  # noqa: BLE001
                pass
    for name in list(_MB["mods"]):
        need = "vms" if MB_BUILDS[name][0] == "vms" else "vml"
        if need not in st["pk"]:
            del _MB["mods"][name]
            continue
        _MB["scr"][name] = _mb_scratch(name, dev, G)
    try:
        _mb_sweep(st, G)
    except Exception as ex:  # noqa: BLE001
        print(f"[mb] sweep raised {ex!r}"[:400], flush=True)
    print(f"[mb] routes { {r: a['spec'] for r, a in _MB['route'].items()} } fallbacks "
          f"{ {r: a['spec'] for r, a in _MB['fb'].items()} } prepare {time.time() - t0:.0f}s", flush=True)
    torch.cuda.synchronize()
    return st


def _mb_sweep(st, G):
    dev = st["dev"]
    ref = _MbRef(st["w"])
    flush = torch.ones((256 << 20) // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)
    stream = torch.cuda.Stream()
    for rep, specs in MB_ARMS.items():
        arms = [_mb_arm(s) for s in specs if s.split(":")[0] in _MB["mods"]]
        if not arms:
            continue
        board = [_mb_board_input(rep, 100 + i, dev) for i in range(MB_SEEDS if rep <= 16 else MB_SEEDS_L)]
        layer = [_mb_layer_input(rep, c, 200 + i, dev) for i, c in enumerate(MB_LAYER_PROFILES.get(rep, []))]
        # 1. cost fits (VML arms with ck f / f2 / g) and tuned tables (t: coordinate descent from f on separate routings)
        tune = [_mb_board_input(rep, 300 + i, dev) for i in range(MB_TUNE_N)]
        for a in arms:
            if a["kind"] != "vml" or a["ck"] not in ("f", "f2", "g", "t"):
                continue
            key = (rep, a["b"], a["flags"])
            try:
                if a["ck"] in ("f", "f2") or (a["ck"] == "t" and rep not in MB_LAYER_COST):
                  if key + ("f",) not in _MB["costs"]:
                    _MB["costs"][key + ("f",)] = _mb_fit(st, a, rep, board[:MB_FITS], None, G, flush, sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "f2" and key + ("f2",) not in _MB["costs"]:
                    _MB["costs"][key + ("f2",)] = _mb_fit(st, a, rep, board[:MB_FITS], _MB["costs"][key + ("f",)], G, flush,
                                                          sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "g" and key + ("g",) not in _MB["costs"] and layer:
                    _MB["costs"][key + ("g",)] = _mb_fit(st, a, rep, layer[:MB_FITS], None, G, flush, sink, stream)
                    _mb_reset(a["b"])
                if a["ck"] == "t" and key + ("t",) not in _MB["costs"]:
                    t_start = list(MB_LAYER_COST[rep]) if rep in MB_LAYER_COST else _MB["costs"][key + ("f",)]
                    tab, u0, u1, kept = _mb_tune(st, a, rep, tune, t_start, _mb_ntiles(tune, a["flags"]), flush, sink, stream)
                    _MB["costs"][key + ("t",)] = tab
                    print(f"[mb] b{rep} tune {a['b']}:{a['flags']}: {u0:.1f} -> {u1:.1f} us (kept {kept})", flush=True)
                    _mb_reset(a["b"])
            except Exception as ex:  # noqa: BLE001
                print(f"[mb] b{rep} fit {a['spec']} raised {ex!r}"[:300], flush=True)
                _mb_reset(a["b"])
        for k, v in _MB["costs"].items():
            if k[0] == rep:
                print(f"[mb] b{rep} cost {k[1]}:{k[2]}:{k[3]} A {v[:8]} B {v[8:]}", flush=True)
        # 2. self-tests
        good = {a["spec"]: True for a in arms}
        for bs, seed, dist in MB_CASES[rep]:
            d = _mb_board_input(bs, seed, dev, dist)
            r = ref(d)
            out = torch.empty((bs, 4096), dtype=torch.bfloat16, device=dev)
            for a in arms:
                if not good[a["spec"]] or bs > _MB["scr"][a["b"]]["maxt"]:
                    continue
                if a["kind"] == "vml" and a["ck"] not in (None, "d", "L") and _mb_cost(a, rep) is None:
                    good[a["spec"]] = False
                    continue
                try:
                    for k in range(2):
                        o = _mb_call(a, st, d["hidden_states"], d["topk_ids"], d["topk_weights"], out, rep)
                        torch.cuda.synchronize()
                        tot, row = _mb_rel(o, r)
                        e = _mb_err(a)
                        ok = tot < 1e-2 and row < 5e-2 and e == 0 and bool(torch.isfinite(o.float()).all())
                        if not ok:
                            print(f"[mb] b{rep} FAIL {a['spec']} bs={bs} {dist} call {k}: rel {tot:.2e} row {row:.2e} err {e}",
                                  flush=True)
                            good[a["spec"]] = False
                            _mb_reset(a["b"])
                            break
                except Exception as ex:  # noqa: BLE001
                    print(f"[mb] b{rep} FAIL {a['spec']} bs={bs} raised {ex!r}"[:300], flush=True)
                    good[a["spec"]] = False
                    _mb_reset(a["b"])
        ok_arms = [a for a in arms if good[a["spec"]]]
        if not ok_arms:
            print(f"[mb] b{rep}: no arm passed", flush=True)
            continue
        # 3. timing: cold-L2 CUDA graph replays, rotated order
        sets = [("brd", board), ("lay", layer)]
        graphs = []
        for a in ok_arms:
            for sname, ds in sets:
                for i, d in enumerate(ds):
                    m = d["hidden_states"].shape[0]
                    if m > _MB["scr"][a["b"]]["maxt"]:
                        continue
                    out = torch.empty((m, 4096), dtype=torch.bfloat16, device=dev)
                    fn = (lambda a=a, d=d, out=out: _mb_call(a, st, d["hidden_states"], d["topk_ids"], d["topk_weights"],
                                                             out, rep))
                    with torch.cuda.stream(stream):
                        fn()
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=stream):
                            fn()
                    torch.cuda.synchronize()
                    graphs.append((a["spec"], sname, i, g))
        ts = {(s, n, i): [] for s, n, i, _ in graphs}
        for rd in range(MB_REPS):
            k0 = (rd * 7) % len(graphs)
            evs = []
            with torch.cuda.stream(stream):
                for s, n, i, g in graphs[k0:] + graphs[:k0]:
                    torch.sum(flush, dim=0, out=sink)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    evs.append(((s, n, i), e0, e1))
            torch.cuda.synchronize()
            for key, e0, e1 in evs:
                ts[key].append(e0.elapsed_time(e1) * 1e3)
        del graphs
        med = {k: sorted(v)[len(v) // 2] for k, v in ts.items()}
        res = {}
        for a in ok_arms:
            for sname, _ in sets:
                v = [x for (s, n, i), x in med.items() if s == a["spec"] and n == sname]
                res[(a["spec"], sname)] = (sum(v) / len(v), min(v), max(v), len(v)) if v else None
        for a in ok_arms:
            # error flags after the timing replays (a bounded wait that expired leaves a flag, never a hang)
            if _mb_err(a) != 0:
                print(f"[mb] b{rep} {a['spec']}: error flag {_mb_err(a)} after timing -> excluded", flush=True)
                good[a["spec"]] = False
                _mb_reset(a["b"])
        ok_arms = [a for a in ok_arms if good[a["spec"]]]
        best = min((a for a in ok_arms if res[(a["spec"], "brd")] is not None), key=lambda a: res[(a["spec"], "brd")][0],
                   default=None)
        vml_ok = [a for a in ok_arms if a["kind"] == "vml" and res[(a["spec"], "brd")] is not None]
        fb = min(vml_ok, key=lambda a: res[(a["spec"], "brd")][0]) if vml_ok else None
        ref_arm = ok_arms[0]

        def paired(a, sname):
            """mean over routings of (this arm / reference arm) per-routing medians, and its standard error"""
            rs = [x / med[(ref_arm["spec"], n, i)] for (s, n, i), x in med.items()
                  if s == a["spec"] and n == sname and (ref_arm["spec"], n, i) in med]
            if not rs:
                return "-"
            mu = sum(rs) / len(rs)
            se = (sum((r - mu) ** 2 for r in rs) / max(1, len(rs) - 1) / len(rs)) ** 0.5
            return f"r{mu:.4f}+-{se:.4f}"
        print(f"[mb] b{rep} ref {ref_arm['spec']}; r = arm / ref per routing (lower is faster), mean +- se", flush=True)
        for a in sorted(ok_arms, key=lambda a: res[(a["spec"], "brd")][0] if res[(a["spec"], "brd")] else 1e9):
            rb, rl = res[(a["spec"], "brd")], res[(a["spec"], "lay")]
            print(f"[mb] b{rep} {a['spec']:<10} brd {rb[0]:7.1f} {paired(a, 'brd')} n{rb[3]}"
                  + (f" | lay {rl[0]:7.1f} {paired(a, 'lay')} n{rl[3]}" if rl else " | lay -"), flush=True)
        bad = [s for s, v in good.items() if not v]
        if bad:
            print(f"[mb] b{rep} failed: {bad}", flush=True)
        if best is not None:
            _MB["route"][rep] = best
        if fb is not None:
            _MB["fb"][rep] = fb
        diag = [a for a in (best, fb) if a is not None]
        vms_ok = [a for a in ok_arms if a["kind"] == "vms" and res[(a["spec"], "brd")] is not None]
        if vms_ok:
            diag += vms_ok if rep == 16 else [min(vms_ok, key=lambda a: res[(a["spec"], "brd")][0])]
        seen_d = set()
        for a in diag:
            if a["spec"] in seen_d:
                continue
            seen_d.add(a["spec"])
            try:
                print(f"[mb] b{rep} marks brd {a['spec']}: {_mb_marks(st, a, rep, board[:3], G, flush, sink, stream)}",
                      flush=True)
                if layer and (a["kind"] == "vml" or rep == 16):
                    print(f"[mb] b{rep} marks lay {a['spec']}: {_mb_marks(st, a, rep, layer[:3], G, flush, sink, stream)}",
                          flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[mb] b{rep} marks {a['spec']} raised {ex!r}"[:200], flush=True)
                _mb_reset(a["b"])
        torch.cuda.synchronize()
    del flush
    ref.c.clear()
'''


def _fmt(d):
    """dict literal, one key per line (no line over 400 chars)"""
    nl = chr(10)
    return "{" + nl + "".join(f"    {k!r}: {v!r}," + nl for k, v in d.items()) + "}"


def main():
    lab = _lab_strings(VMS_LAB, ("CUDA_SRC", "CPP_SRC", "POS"))
    cuda = lab["CUDA_SRC"]
    for a, b in VMS_PATCHES + X2S_PATCHES:
        assert cuda.count(a) == 1, a[:70]
        cuda = cuda.replace(a, b)
    cpp = lab["CPP_SRC"]
    vml, _ = vml_patches.lab_block()
    for a, b in VML_PATCHES:
        assert vml.count(a) == 1, a[:70]
        vml = vml.replace(a, b)
    k2a = _segments(K2A_SRC, K2A_NAMES)
    head = ("# okc: problem=muse_moe_w4a16\n"
            f'"""mb_{N}: MoE board loop file (tools/mb_gen.py; notes/study/moe_board_loop.md). VMS (one-launch mma.sync, jobs of\n'
            "<= 8 tokens) + VML (one-launch RS-wgmma stream-K) with an in-prepare variant sweep per bucket on the judge GPU;\n"
            'our own implementations of the public vjeux structure (ideas only). Check only, never scored."""\n'
            "import hashlib\nimport time\n\nimport torch\n\n")
    s = head
    s += "# ======== K2A v3.1 prepack helpers (VML's ws13 / ws2 stage layouts; verbatim from vml_2.py)\n" + k2a + "\n\n"
    s += ("# ======== VMS (kernels/lab/vms_moe_lab.py CUDA, + VMS_MAXT)\nMB_VMS_CUDA = r'''" + cuda + "'''\n"
          "MB_VMS_CPP = r'''" + cpp + "'''\nMB_VMS_POS = " + repr(lab["POS"]) + "\n\n")
    s += "# ======== VML (kernels/lab/vml_moe.py via tools/vml_patches.lab_block, + flags bit 10)\n" + vml + "\n"
    glue = (GLUE.replace("@N@", str(N)).replace("@BUILDS@", _fmt(CFG["builds"])).replace("@ARMS@", _fmt(CFG["arms"]))
            .replace("@LCOST@", _fmt(_layer_vml_cost())).replace("@LPROF@", _fmt(_layer_profiles())))
    s += glue
    assert "'''" not in cuda and "'''" not in cpp
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    long_lines = [(i + 1, len(x)) for i, x in enumerate(s.split("\n")) if len(x) >= 1000]
    assert not long_lines, long_lines
    OUT.write_text(s, encoding="utf-8", newline="\n")
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

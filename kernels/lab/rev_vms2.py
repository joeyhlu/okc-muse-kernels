"""REV VMS2 (independent review, notes/study/independent_review_oct28.md s13; request R-rev-5): 16-token VMS jobs that keep
the 1-tile body's speed.

Built on kernels/lab/vms_moe_lab.py (text patches; the team's file is not touched). The lab already has 16-token jobs
(VMS_FL 128: VMS_SKT 16, second x tile x2 in registers, consume<true>, red2). That build lost ~20 % per unit (R-vms-6)
because 9 warps (8 compute + 1 helper) put 3 warps on one SMSP, which caps ptxas at 168 registers, too few for two x tiles
(2 x 64 registers). New switch here:
  VMS_NOH 1: no helper warp (256 threads = 2 warps per SMSP -> 255-register cap). Each unit's K reduction, epilogue and
             doneA credit run inline in compute warp (q % 8) right after the per-unit compute barrier, i.e. inside the
             slack that HBM-bound units leave (consumer ~0.75 us vs ~1.17 us of stream per unit, vm_small s4). The
             partial-sum buffers need no pfull / pempty handshake: buffer q & 3 is rewritten only at unit q + 4, after
             barriers q + 1 .. q + 3, which the epilogue warp must reach first.
Builds for R-rev-5: base (shipped lab VMS, fl 0), h1 (fl 0 + NOH: helper removed, 1-tile jobs), k16 (fl 128: the team's
16-token build, reference), k16h (fl 128 + NOH: this design).
"""
import hashlib
import time

import vms_moe_lab as V

_EPI = r"""
#if VMS_NOH
// one warp: unit q's K reduction over the NW partials, epilogue and doneA credit (the helper warp's loop body)
__device__ __forceinline__ void epi_unit(const Params& p, Smem& S, int q, int nA, int a0, int a1, int b0, int lane,
                                         bool mk, long long* mrow) {
  const int bq = q & (NRB - 1);
  const int g = lane >> 2, t = lane & 3;
  float* ssc = p.ss + (S.gen & 1) * MAXP;
  float4 sm = S.red[bq][0][lane];
#pragma unroll
  for (int w = 1; w < NW; ++w) { const float4 u = S.red[bq][w][lane]; sm.x += u.x; sm.y += u.y; sm.z += u.z; sm.w += u.w; }
#if VMS_SKT == 16
  const int jq = q < nA ? ((a0 + q) >> 9) : ((b0 + (q - nA)) >> 8);
  const int ntile = S.jC[jq] > 8 ? 2 : 1;
  float4 sm2 = make_float4(0.f, 0.f, 0.f, 0.f);
  if (ntile == 2) {
#pragma unroll
    for (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }
  }
#else
  const int ntile = 1;
  const float4 sm2 = sm;
#endif
  if (q < nA) {
    const int v = a0 + q, j = v >> 9, rb = v & 511;
    const int cnt = S.jC[j], st = S.jS[j];
    const float gs = __ldg(p.gsA + S.jE[j]);
    for (int tile = 0; tile < ntile; ++tile)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 8 * tile + 2 * t + h;
      const float4 ms = tile ? sm2 : sm;
      if (tk < cnt) {
        const float ga = (h ? ms.y : ms.x) * gs, up = (h ? ms.w : ms.z) * gs;
        const float sv = __fdividef(ga, 1.f + __expf(-ga)) * up;
        p.inter[(size_t)(st + tk) * ID + rb * 8 + g] = __float2bfloat16(sv);
      }
    }
    if (q == nA - 1 || ((v + 1) >> 9) != j) {     // this CTA's last unit of job j: credit its units
      __threadfence();
      __syncwarp();
      if (lane == 0) atomicAdd(p.ctr + 8 + j, min(a1, (j + 1) * RBA) - max(a0, j * RBA));
      if (mk && lane == 0 && q == nA - 1) mrow[6] = gt();
    }
  } else {
    const int v = b0 + (q - nA), j = v >> 8, rb = v & 255;
    const int cnt = S.jC[j], st = S.jS[j];
    const float gs = __ldg(p.gsB + S.jE[j]);
    const int n0 = rb * 16 + g;
    const float w0 = __ldg(p.nw + n0) * gs, w1 = __ldg(p.nw + n0 + 8) * gs;
    for (int tile = 0; tile < ntile; ++tile)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = 8 * tile + 2 * t + h;
      const float4 ms = tile ? sm2 : sm;
      const float y0 = (h ? ms.y : ms.x) * w0, y1 = (h ? ms.w : ms.z) * w1;
      float sq = y0 * y0 + y1 * y1;
      if (tk < cnt) {
        p.ybuf[(size_t)(st + tk) * HD + n0] = y0;
        p.ybuf[(size_t)(st + tk) * HD + n0 + 8] = y1;
      }
      sq += __shfl_xor_sync(0xffffffffu, sq, 4);
      sq += __shfl_xor_sync(0xffffffffu, sq, 8);
      sq += __shfl_xor_sync(0xffffffffu, sq, 16);
      if (g == 0 && tk < cnt) atomicAdd(ssc + st + tk, sq);
    }
  }
}
#endif
"""


def _patch(src, old, new):
    assert src.count(old) == 1, (src.count(old), old[:80])
    return src.replace(old, new)


SRC = V.CUDA_SRC
SRC = _patch(SRC, "#define VMS_DW ((VMS_FL & 64) != 0)", "#ifndef VMS_NOH\n#define VMS_NOH 0\n#endif\n#define VMS_DW ((VMS_FL & 64) != 0)")
SRC = _patch(SRC, "NTL = NT + 32,", "NTL = NT + (VMS_NOH ? 0 : 32),")
SRC = _patch(SRC, "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {",
             _EPI + "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {")
SRC = _patch(SRC, "      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));\n",
             "#if !VMS_NOH\n      mbar_wait(&S.pempty[bq], (uint32_t)(((q / NRB) & 1) ^ 1));\n#endif\n")
SRC = _patch(SRC, "#else\n      if (tid == 0) mbar_arrive(&S.pfull[bq]);\n#endif\n#endif\n      if (++sl == NS) { sl = 0; par ^= 1u; }",
             "#else\n#if VMS_NOH\n      if (warp == q % NW) epi_unit(p, S, q, nA, a0, a1, b0, lane, mk, mrow);\n#else\n"
             "      if (tid == 0) mbar_arrive(&S.pfull[bq]);\n#endif\n#endif\n#endif\n      if (++sl == NS) { sl = 0; par ^= 1u; }")
SRC = _patch(SRC, "    if (mk && tid == 0) mrow[5] = gt();\n", "#if VMS_NOH\n    __threadfence();\n#endif\n    if (mk && tid == 0) mrow[5] = gt();\n")
CUDA_SRC = SRC
CPP_SRC = V.CPP_SRC
BUILDS = {"base": (0, 0), "h1": (0, 1), "k16": (128, 0), "k16h": (128, 1)}   # name -> (VMS_FL, VMS_NOH)


def build(G, b, ns=3, arch="sm_90a", verbose=False, nw=8):
    from torch.utils.cpp_extension import load_inline
    fl, noh = BUILDS[b]
    gen = f"-gencode=arch=compute_{arch[3:].rstrip('a')}{'a' if arch.endswith('a') else ''},code={arch}"
    flags = ["-O3", f"-DVMS_G={G}", f"-DVMS_NS={ns}", f"-DVMS_NW={nw}", f"-DVMS_FL={fl}", f"-DVMS_NOH={noh}",
             "-Xptxas=-warn-spills", gen]
    tag = hashlib.sha1((CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"rev_vms2_{b}_{tag}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["run", "attrs"],
                      extra_cuda_cflags=flags, verbose=verbose)
    return mod, time.time() - t0

"""vlc_patches.py: VMS_LC = last-arriver finish at M == 1 on the layer VMS (moe redesign, R-vlc; bs1 tail, mb_22 marks).
Ship end: every CTA fences + adds its arrival, tid 0 polls the arrival counter until all G arrived (grid barrier, observed one L2
round trip after the last add), CTA 0 resets the next launch's counters, CTAs 0..3 combine one 1024-column quarter each. bs1 marks:
last helper end 26.1-26.4 -> barrier passed 26.8-27.2 -> final pass 27.7-28.1 us. VMS_LC 1, when M == 1: the add returns the old
count; the CTA whose add completes the generation fences (acquire), resets the counters and combines all 4096 columns itself
(256 threads x 4 float4, routes summed in the ship's order -> bitwise equal), every other CTA exits after its add. Saves the poll
observation RTT and the barrier skew; the combine's loads are all issued in one batch. M > 1 runs the ship's path. At VMS_LC 0
the source preprocesses to the base's.
    python -I tools/vlc_patches.py <file.py>
"""
import pathlib
import sys

OLD_SM = "  int njobs, gen;\n"
NEW_SM = "  int njobs, gen;\n#if VMS_LC\n  int last;   // R-vlc: this CTA completed the arrival generation\n#endif\n"
OLD_END = "  const int gen = S.gen;\n"
NEW_END = r"""  const int gen = S.gen;
#if VMS_LC
  if (p.m == 1) {   // R-vlc: last arriver resets + combines; the others exit after arriving
    if (tid == 0) {
      __threadfence();
      const int old = atomicAdd(p.ctr + 1, 1);
      const int lst = old == (gen + 1) * G - 1;
      if (lst) __threadfence();
      S.last = lst;
      if (mk) mrow[8] = gt();
    }
    __syncthreads();
    if (!S.last) {
      if (mk && tid == 0) mrow[9] = gt();
      return;
    }
    if (tid < MAXJ) p.ctr[8 + tid] = 0;
    if (tid < MAXP) p.ss[((gen + 1) & 1) * MAXP + tid] = 0.f;
    if (tid == 0) atomicAdd(p.ctr, 1);
    if (tid < NT) {
      const float* ssl = p.ss + (gen & 1) * MAXP;
      float4 a[4];
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) a[qq] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int pr = S.pos[s];
        if (pr >= 0) {
          const float cf = __ldg(p.tw + s) * rsqrtf(__ldcg(ssl + pr) * (1.f / HD) + p.eps);
          const float4* yp = reinterpret_cast<const float4*>(p.ybuf + (size_t)pr * HD) + tid;
#pragma unroll
          for (int qq = 0; qq < 4; ++qq) {
            const float4 y = __ldcg(yp + qq * 256);
            a[qq].x += cf * y.x; a[qq].y += cf * y.y; a[qq].z += cf * y.z; a[qq].w += cf * y.w;
          }
        }
      }
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        __nv_bfloat162 o0 = __floats2bfloat162_rn(a[qq].x, a[qq].y), o1 = __floats2bfloat162_rn(a[qq].z, a[qq].w);
        uint2 pk; pk.x = *reinterpret_cast<uint32_t*>(&o0); pk.y = *reinterpret_cast<uint32_t*>(&o1);
        *reinterpret_cast<uint2*>(p.out + qq * 1024 + tid * 4) = pk;
      }
    }
    if (mk && tid == 0) mrow[9] = gt();
    return;
  }
#endif
"""
DEF_ANCHOR = "#ifndef VMS_G\n"
DEFS = "#ifndef VMS_LC\n#define VMS_LC 0   // R-vlc: 1 = last-arriver reset + combine at M == 1\n#endif\n"


def apply(s):
    for old, tag in ((OLD_SM, "smem"), (OLD_END, "end"), (DEF_ANCHOR, "defines")):
        assert s.count(old) == 1, f"vlc anchor '{tag}' found {s.count(old)} times"
    s = s.replace(OLD_SM, NEW_SM).replace(OLD_END, NEW_END)
    return s.replace(DEF_ANCHOR, DEFS + DEF_ANCHOR)


if __name__ == "__main__":
    t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
    print(f"[vlc] {sys.argv[1]}: +{len(apply(t)) - len(t)} B, all anchors applied")

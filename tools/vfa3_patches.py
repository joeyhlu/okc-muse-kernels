"""vfa3_patches.py: VFA3 = V^T transpose of the VFA verify-full kernel moved from the consumer warpgroups to the producer warps
(FA3 FP8 style). Apply AFTER fa16_patches.apply (anchors are the FA16-patched text).

PVT: producer warp w (0..3) transposes keys 16 w .. 16 w + 15 of BOTH heads of tile j (same ldmatrix.trans + prmt + st.shared
as the consumer had) into V^T buffer j & 1, after issuing the loads of tile j + LAG: wait full[j % ST] (V landed), wait vte[j & 1]
(PV(j - 2) done by both consumer warpgroups), transpose, arrive empt (the V half of the stage is read; empt counts NCONS + NPROD),
fence.proxy.async, arrive vtf[j & 1] (count NPROD). Consumer: no V^T code; before PV(i) it waits vtf[i & 1] (replaces the
fence + named barrier), after PV(i)'s wait it arrives vte[i & 1]. Barriers vtf[2] / vte[2] live in the 128-B barrier block
(full[ST] + empt[ST] + 4 <= 14 x 8 B). No smem growth. setmaxnreg producer / consumer = PREG / CREG (pool 80 x 384).
Lab arms (VFA_DIAG, lab build): 16 = FA16-VSM, 17 = NOVT (consumer skips V^T + fence + barrier: wrong output, timing bound only),
18 = PVT LAG 2, 19 = PVT LAG 1, 20 = PVT LAG 3.
"""

NL = chr(10)


def _one(t, old, new, what, n=1):
    c = t.count(old)
    assert c == n, f"vfa3_patches: anchor '{what}' found {c} times (want {n})"
    return t.replace(old, new)


def apply(t, pvt=False, lag=2, preg=40, creg=96, lab=True):
    assert 1 <= lag <= 3 and preg % 8 == 0 and creg % 8 == 0 and preg * 128 + creg * 256 <= 80 * 384
    # helpers: compile-time PVT switch / lag, after the pdl helpers
    anchor = '__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }' + NL
    t = _one(t, anchor, anchor
             + f"template <int DIAG> constexpr bool pvt_on() {{ return (DIAG >= 18 && DIAG <= 20) || (DIAG == 0 && {'true' if pvt else 'false'}); }}   // VFA3-PVT" + NL
             + f"template <int DIAG> constexpr int pvt_lag() {{ return DIAG == 19 ? 1 : (DIAG == 20 ? 3 : (DIAG == 18 ? 2 : {int(lag)})); }}" + NL
             + f'#define VFA_PVT_PREG "{int(preg)}"' + NL + f'#define VFA_PVT_CREG "{int(creg)}"' + NL, "vfa pdl_trigger")
    # producer: template DIAG, transposer
    t = _one(t, "template <int ST, bool O32>" + NL + "__device__ __forceinline__ void producer(",
             "template <int ST, bool O32, int DIAG>" + NL + "__device__ __forceinline__ void producer(", "vfa producer template")
    t = _one(t, "  const uint32_t so = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);" + NL + "  int n = 0;" + NL,
             "  const uint32_t so = sa(smem) + r0 * ROWB + ((ch ^ (r0 & 7)) << 4);" + NL + "  int n = 0;" + NL + PROD_VT, "vfa producer so")
    t = _one(t, "      cp_async_mbar_arrive(full + st);" + NL,
             "      cp_async_mbar_arrive(full + st);" + NL
             + "      if constexpr (PVT) { if (n >= LAG) vt_tile(n - LAG); }   // VFA3-PVT: tile n - LAG landed long ago" + NL, "vfa producer arrive")
    i = t.index("__device__ __forceinline__ void producer(")
    end = "      for (int k = 0; k < 4; ++k) pg[k] = pgn[k];" + NL + "    }" + NL + "  }" + NL + "}" + NL
    j = t.index(end, i)
    assert t.count(end) == 1, "vfa3_patches: producer end"
    t = t[:j] + end[:-2] + "  if constexpr (PVT) { for (int j = (n > LAG ? n - LAG : 0); j < n; ++j) vt_tile(j); }   // VFA3-PVT: drain" + NL + "}" + NL + t[j + len(end):]
    # consumer: switches, V^T block off, barrier -> vtf wait, vte arrive after PV
    t = _one(t, "  constexpr uint32_t PBK = (DIAG == 6) ? 0x02020202u : PCHK;" + NL,
             "  constexpr uint32_t PBK = (DIAG == 6) ? 0x02020202u : PCHK;" + NL
             + "  constexpr bool NOVT = (DIAG == 17);   // VFA3 lab bound: no V^T at all (wrong output)" + NL
             + "  constexpr bool PVT = pvt_on<DIAG>();   // VFA3-PVT: V^T written by the producer warps" + NL
             + "  uint64_t* const vtf = empt + ST;" + NL + "  uint64_t* const vte = vtf + 2;" + NL, "vfa consumer PBK")
    t = _one(t, "        if (DIAG != 2) {   // this tile's V -> V^T (this head, buffer buf) while the QK wgmma runs" + NL,
             "        if (DIAG != 2 && !NOVT && !PVT) {   // this tile's V -> V^T (this head, buffer buf) while the QK wgmma runs" + NL, "vfa V^T if")
    sync = ("fence_proxy_async();                                   // V^T (st.shared) -> async proxy" + NL
            + "{ind}bar_named(1 + h, 128);                                 // all 4 warps' V^T rows written" + NL)
    for ind in ("          ",):
        old = ind + sync.format(ind=ind)
        new = (ind + "if constexpr (PVT) {   // VFA3-PVT: the producer's V^T of this tile (fence by the writers)" + NL
               + ind + "  mbar_wait(vtf + buf, ((uint32_t)n >> 1) & 1u);" + NL
               + ind + "} else if constexpr (!NOVT) {" + NL
               + ind + "  " + sync.format(ind=ind + "  ") + ind + "}" + NL)
        t = _one(t, old, new, "vfa V^T sync", n=2)
    t = _one(t, "      buf ^= 1;" + NL + "      if (++st_c == ST) { st_c = 0; ph_c ^= 1u; }",
             "      if constexpr (PVT) mbar_arrive(vte + buf);   // VFA3-PVT: PV(n) done (waited), V^T buffer free" + NL
             + "      buf ^= 1;" + NL + "      if (++st_c == ST) { st_c = 0; ph_c ^= 1u; }", "vfa buf toggle")
    # kernel: barrier counts, setmaxnreg, producer call
    t = _one(t, "    mbar_init(empt + tid, NCONS);" + NL,
             "    mbar_init(empt + tid, NCONS + (pvt_on<DIAG>() ? NPROD : 0));" + NL + "  }" + NL
             + "  if (tid < 2) {   // VFA3-PVT V^T barriers: vtf[2] (producer writes), vte[2] (consumer PV done)" + NL
             + "    mbar_init(full + 2 * ST + tid, NPROD);" + NL + "    mbar_init(full + 2 * ST + 2 + tid, NCONS);" + NL, "vfa mbar init")
    t = _one(t, '    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\\n" ::: "memory");' + NL
             + "    producer<ST, O32>(p, smem, seg, nseg, full, empt, tid - NCONS, pair);" + NL,
             '    if constexpr (pvt_on<DIAG>()) asm volatile("setmaxnreg.dec.sync.aligned.u32 " VFA_PVT_PREG ";\\n" ::: "memory");' + NL
             + '    else asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\\n" ::: "memory");' + NL
             + "    producer<ST, O32, DIAG>(p, smem, seg, nseg, full, empt, tid - NCONS, pair);" + NL, "vfa producer call")
    t = _one(t, '    asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\\n" ::: "memory");' + NL,
             '    if constexpr (pvt_on<DIAG>()) asm volatile("setmaxnreg.inc.sync.aligned.u32 " VFA_PVT_CREG ";\\n" ::: "memory");' + NL
             + '    else asm volatile("setmaxnreg.inc.sync.aligned.u32 104;\\n" ::: "memory");' + NL, "vfa consumer setmaxnreg")
    if lab:
        t = _one(t, "      case 6: vfa_go<ST, 6, true>(p, grid, stream, pdl); return;" + NL,
                 "      case 6: vfa_go<ST, 6, true>(p, grid, stream, pdl); return;" + NL
                 + "".join(f"      case {d}: vfa_go<ST, {d}, true>(p, grid, stream, pdl); return;" + NL for d in (16, 17, 18, 19, 20)),
                 "vfa diag 6")
    return t


PROD_VT = r"""  constexpr bool PVT = pvt_on<DIAG>();   // VFA3-PVT: this warp's 16 keys of both heads -> V^T (consumer's mapping, w = q >> 5)
  constexpr int LAG = pvt_lag<DIAG>();
  uint64_t* const vtf = empt + ST;
  uint64_t* const vte = vtf + 2;
  const int vlane = q & 31, vw = q >> 5, vg = vlane >> 2;
  const uint32_t vld0 = sa(smem) + off_v<ST>() + (16 * vw + 8 * ((vlane >> 3) & 1) + (vlane & 7)) * ROWB + (((vlane >> 4) ^ (vlane & 7)) << 4);
  const uint32_t vsta = sa(smem) + off_vt<ST>() + 128 * vg + ((vw ^ (vg & 3)) << 4) + 4 * (vlane & 3) + ((vg >= 4) ? 64u : 0u);
  const uint32_t vsa = (vg >= 4) ? 0x7531u : 0x6420u, vsb = vsa ^ 0x1111u;
  auto vt_tile = [&](int j) {
    const int st = j % ST, bf = j & 1;
    mbar_wait(full + st, (uint32_t)(j / ST) & 1u);
    if (j >= 2) mbar_wait(vte + bf, (uint32_t)((j >> 1) - 1) & 1u);
    const uint32_t vs0 = vld0 + (uint32_t)(st * STAGE_BYTES), d0 = vsta + (uint32_t)(bf * VT_BYTES);
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const uint32_t vsrc = vs0 ^ (64u * hh), da = d0 + (uint32_t)(hh * 2 * VT_BYTES), db = da ^ 64u;
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t r[4];
        ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
        sts32(da + 1024 * (2 * u), prmt(r[0], r[1], vsa));
        sts32(db + 1024 * (2 * u), prmt(r[0], r[1], vsb));
        sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], vsa));
        sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], vsb));
      }
    }
    mbar_arrive(empt + st);   // the V half of stage st read
    fence_proxy_async();      // V^T (st.shared) -> async proxy (wgmma)
    mbar_arrive(vtf + bf);
  };
  (void)vt_tile; (void)vtf; (void)vte;
"""

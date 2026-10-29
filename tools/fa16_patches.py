"""fa16_patches.py: compile-time edits of the VFA block (verify full attention, q_len 8, 2 <= B <= 256) of a k3_gc_9-family layer.

FA16-PEX: part of the softmax exponentials P = e4m3(2^x) computed on the FMA pipe (Cody-Waite split, degree-3 polynomial on
[-0.5, 0.5], exponent by integer add; x clamped to [-125, 126]) instead of MUFU.EX2, so the 4 consumer warpgroups per SM stop
queueing on the SFU (16 / clk / SM). P keeps its precision: the same f32 value (rel err ~1e-4 vs 2^-22 for ex2.approx, both far
below the e4m3 step of 2^-4) goes through the same cvt.rn.satfinite.e4m3x2.f32. PEX = emulated elements of the 32 per thread
and tile, spread by (k * PEX) % 32 < PEX.
Lab arms (VFA_DIAG, lab build only): 7 = no exponentials (x itself, garbage, timing only), 8..11 = PEX 4 / 8 / 12 / 16.
"""

NL = chr(10)


def _one(t, old, new, what):
    n = t.count(old)
    assert n == 1, f"fa16_patches: anchor '{what}' found {n} times"
    return t.replace(old, new)


EX2P = r'''__device__ __forceinline__ float ex2p(float x) {   // FA16-PEX: 2^x on the FMA pipe (x in [-125, 126])
  x = fminf(fmaxf(x, -125.f), 126.f);
  const float t = x + 12582912.f;                     // 1.5 x 2^23: round(x) in the low mantissa bits
  const float f = x - (t - 12582912.f);               // [-0.5, 0.5]
  const float p = fmaf(fmaf(fmaf(0.05550411f, f, 0.24022652f), f, 0.69314718f), f, 1.0f);
  return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
'''

PACK_OLD = '''        auto pack = [&]() {   // P = e4m3(ex2(s * ssc - offset)), keys permuted into the PV A-fragment order
#pragma unroll
          for (int kk = 0; kk < 2; ++kk) {
            const float* sv = sc + 16 * kk;
            pa[kk][0] = pk4(ex2(fmaf(sv[0], ssc, -cr0)), ex2(fmaf(sv[1], ssc, -cr0)), ex2(fmaf(sv[4], ssc, -cr0)), ex2(fmaf(sv[5], ssc, -cr0)));
            pa[kk][1] = pk4(ex2(fmaf(sv[2], ssc, -cr1)), ex2(fmaf(sv[3], ssc, -cr1)), ex2(fmaf(sv[6], ssc, -cr1)), ex2(fmaf(sv[7], ssc, -cr1)));
            pa[kk][2] = pk4(ex2(fmaf(sv[8], ssc, -cr0)), ex2(fmaf(sv[9], ssc, -cr0)), ex2(fmaf(sv[12], ssc, -cr0)), ex2(fmaf(sv[13], ssc, -cr0)));
            pa[kk][3] = pk4(ex2(fmaf(sv[10], ssc, -cr1)), ex2(fmaf(sv[11], ssc, -cr1)), ex2(fmaf(sv[14], ssc, -cr1)), ex2(fmaf(sv[15], ssc, -cr1)));
          }
        };'''

PACK_NEW = '''        auto e2 = [&](float x, int k) {   // FA16-PEX: element k of 32 on the FMA pipe iff (k * PEX) % 32 < PEX
          if (DIAG == 7) return x;
          return (((k * PEX) & 31) < PEX) ? ex2p(x) : ex2(x);
        };
        auto q4 = [&](float a, float b, float c, float d, int k) -> uint32_t {   // FA16-H16: exps as f16x2, same e4m3 P
          if constexpr (H16) return pk4h(a, b, c, d);
          else return pk4(e2(a, k), e2(b, k + 1), e2(c, k + 2), e2(d, k + 3));
        };
        auto packh = [&](int kk) {   // P = e4m3(ex2(s * ssc - offset)) of keys 32 kk .. +31, permuted into the PV A-fragment order
          const float* sv = sc + 16 * kk;
          const int q = 16 * kk;
          pa[kk][0] = q4(fmaf(sv[0], ssc, -cr0), fmaf(sv[1], ssc, -cr0), fmaf(sv[4], ssc, -cr0), fmaf(sv[5], ssc, -cr0), q);
          pa[kk][1] = q4(fmaf(sv[2], ssc, -cr1), fmaf(sv[3], ssc, -cr1), fmaf(sv[6], ssc, -cr1), fmaf(sv[7], ssc, -cr1), q + 4);
          pa[kk][2] = q4(fmaf(sv[8], ssc, -cr0), fmaf(sv[9], ssc, -cr0), fmaf(sv[12], ssc, -cr0), fmaf(sv[13], ssc, -cr0), q + 8);
          pa[kk][3] = q4(fmaf(sv[10], ssc, -cr1), fmaf(sv[11], ssc, -cr1), fmaf(sv[14], ssc, -cr1), fmaf(sv[15], ssc, -cr1), q + 12);
        };
        auto pack = [&]() { packh(0); packh(1); };'''

PK4H = r'''__device__ __forceinline__ uint32_t pk4h(float a, float b, float c, float d) {   // FA16-H16: 4 x e4m3(2^x) via f16x2, a low byte
  uint32_t r;
  asm("{\n.reg .b32 h0, h1;\n.reg .b16 lo, hi;\ncvt.rn.f16x2.f32 h0, %2, %1;\ncvt.rn.f16x2.f32 h1, %4, %3;\n"
      "ex2.approx.f16x2 h0, h0;\nex2.approx.f16x2 h1, h1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 lo, h0;\ncvt.rn.satfinite.e4m3x2.f16x2 hi, h1;\nmov.b32 %0, {lo, hi};\n}"
      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}
'''

PVS_HEAD = '''        const bool was_first = first;
        auto bump = [&]() {   // FA16-PVS: offsets up to this tile's row maxima, state rescaled (rare)
          float m0 = neg_inf(), m1 = neg_inf();
#pragma unroll
          for (int nt8 = 0; nt8 < 8; ++nt8) {
            m0 = fmaxf(m0, fmaxf(sc[4 * nt8], sc[4 * nt8 + 1]));
            m1 = fmaxf(m1, fmaxf(sc[4 * nt8 + 2], sc[4 * nt8 + 3]));
          }
          m0 = quad_max(m0) * ssc;
          m1 = quad_max(m1) * ssc;
          const float n0 = fmaxf(m0 - POF, cr0), n1 = fmaxf(m1 - POF, cr1);
          const float a0 = ex2(cr0 - n0), a1 = ex2(cr1 - n1);
#pragma unroll
          for (int t = 0; t < 9; ++t) { acc[4 * t] *= a0; acc[4 * t + 1] *= a0; acc[4 * t + 2] *= a1; acc[4 * t + 3] *= a1; }
          cr0 = n0; cr1 = n1;
        };
        auto ovf = [&](int kk) -> bool {
          constexpr uint32_t PB = PBK;
          const uint32_t ov = ((pa[kk][0] + PB) | (pa[kk][1] + PB) | (pa[kk][2] + PB) | (pa[kk][3] + PB)) & 0x80808080u;
          return __any_sync(FULL, ov != 0u);
        };
        if constexpr (!PVS) {
'''

PVS_TAIL = '''        } else {   // FA16-PVS: PV of keys 0..31 in flight while keys 32..63 are exponentiated
          packh(0);
          if (!was_first && DIAG != 4 && ovf(0)) { bump(); packh(0); }
          first = false;
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 4; ++e) VFA_RR(pa[0][e]);
          fence_proxy_async();                                   // V^T (st.shared) -> async proxy
          bar_named(1 + h, 128);                                 // all 4 warps' V^T rows written
          const uint32_t vl = vtlo + (uint32_t)(buf * (VT_BYTES >> 4));
          wg_fence();
          wg_pv(acc, pa[0], HI64 | vl);
          wg_commit();
          packh(1);
          if (!was_first && DIAG != 4 && ovf(1)) {   // rare: PV(keys 0..31) done, then rescale and repack keys 32..63
            wg_wait0();
#pragma unroll
            for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
            bump();
            packh(1);
          }
#pragma unroll
          for (int e = 0; e < 4; ++e) VFA_RR(pa[1][e]);
          wg_fence();
          wg_pv(acc, pa[1], HI64 | (vl + 2u));
          wg_commit();
          wg_wait0();                                            // nothing in flight across the loop edge
#pragma unroll
          for (int e = 0; e < 36; ++e) VFA_RF(acc[e]);
#pragma unroll
          for (int e = 0; e < 8; ++e) VFA_RR(pa[e >> 2][e & 3]);
        }
'''

TAIL_A = '        pack();' + NL + '        if (!first && DIAG != 4) {   // lazy offset'
TAIL_B = '      buf ^= 1;' + NL + '      if (++st_c == ST) { st_c = 0; ph_c ^= 1u; }'



def apply(t, pex=0, h16=False, pvs=False, pp=False, vsm=False, lab=True):
    """t: layer text carrying the VFA block. pex / h16 / pvs: the shipped DIAG 0 defaults. lab: add the DIAG 7..14 arms to vfa_diag
    (7 no exps, 8..11 PEX 4..16, 12 H16, 13 PVS, 14 H16 + PVS)."""
    ex2l = '__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }' + NL
    t = _one(t, ex2l + '__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }' + NL,
             ex2l + '__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }' + NL + EX2P, "vfa ex2 / neg_inf")
    pk_end = '      : "=r"(r) : "f"(a), "f"(b), "f"(c), "f"(d));' + NL + '  return r;' + NL + '}' + NL
    i = t.index("__device__ __forceinline__ uint32_t pk4(float a, float b, float c, float d)")
    j = t.index(pk_end, i) + len(pk_end)
    t = t[:j] + PK4H + t[j:]
    t = _one(t, "  constexpr uint32_t PBK = (DIAG == 6) ? 0x02020202u : PCHK;" + NL,
             "  constexpr uint32_t PBK = (DIAG == 6) ? 0x02020202u : PCHK;" + NL
             + f"  constexpr int PEX = (DIAG >= 8 && DIAG <= 11) ? 4 * (DIAG - 7) : {int(pex) if True else 0};   // FA16-PEX (lab 8..11)" + NL
             + f"  constexpr bool H16 = (DIAG == 12 || DIAG == 14) || (DIAG == 0 && {'true' if h16 else 'false'});   // FA16-H16" + NL
             + f"  constexpr bool PVS = (DIAG == 13 || DIAG == 14) || (DIAG == 0 && {'true' if pvs else 'false'});   // FA16-PVS" + NL,
             "vfa PBK")
    t = _one(t, PACK_OLD, PACK_NEW, "vfa pack")
    # FA16-VSM (DIAG 16): the V^T per-thread constants (V row address, V^T address, prmt selector) kept in smem (4 KB, one 16-B slot
    # per consumer thread) and re-read each tile, instead of the ~25 instructions ptxas rematerializes per tile at 104 registers
    vsc = f"(DIAG == 16 || (DIAG == 0 && {'true' if vsm else 'false'}))"
    t = _one(t, "template <int ST> constexpr int smem_bytes() { return off_misc<ST>() + 64 + 1024; }",
             "template <int ST> constexpr int off_adr() { return off_misc<ST>() + 64; }" + NL
             + "template <int ST> constexpr int smem_bytes() { return off_adr<ST>() + 4096 + 1024; }   // FA16-VSM slots, 1 KB align slack last",
             "vfa smem_bytes")
    t = _one(t, "  const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);" + NL,
             "  const uint32_t vst_a = vst + ((g >= 4) ? 64u : 0u), vst_b = vst + ((g >= 4) ? 0u : 64u);" + NL
             + f"  constexpr bool VSM = {vsc};" + NL
             + "  const uint32_t vslot = sa(smem) + off_adr<ST>() + (uint32_t)tid * 16u;" + NL
             + '  if (VSM) asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(vslot), "r"(vld0), "r"(vst_a), "r"(vsel_a), "r"(0u) : "memory");' + NL,
             "vfa vst_a")
    old_vt = """          const uint32_t vsrc = vld0 + (uint32_t)(st_c * STAGE_BYTES);
          const uint32_t da = vst_a + (uint32_t)(buf * VT_BYTES), db = vst_b + (uint32_t)(buf * VT_BYTES);
#pragma unroll
          for (int u = 0; u < 2; ++u) {
            uint32_t r[4];
            ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
            sts32(da + 1024 * (2 * u), prmt(r[0], r[1], vsel_a));
            sts32(db + 1024 * (2 * u), prmt(r[0], r[1], vsel_b));
            sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_a));
            sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], vsel_b));
          }
"""
    new_vt = """          if constexpr (VSM) {   // FA16-VSM: constants from this thread's smem slot (db = da ^ 64, selector b = a ^ 0x1311)
            uint32_t x0, x1, x2, x3;
            asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(vslot) : "memory");
            const uint32_t vsrc = x0 + (uint32_t)(st_c * STAGE_BYTES);
            const uint32_t da = x1 + (uint32_t)(buf * VT_BYTES), db = da ^ 64u, sA = x2, sB = x2 ^ 0x1311u;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
              uint32_t r[4];
              ldsm_x4_t(r, u ? (vsrc ^ 32u) : vsrc);
              sts32(da + 1024 * (2 * u), prmt(r[0], r[1], sA));
              sts32(db + 1024 * (2 * u), prmt(r[0], r[1], sB));
              sts32(da + 1024 * (2 * u + 1), prmt(r[2], r[3], sA));
              sts32(db + 1024 * (2 * u + 1), prmt(r[2], r[3], sB));
            }
          } else {
""" + old_vt + "          }" + NL
    t = _one(t, old_vt, new_vt, "vfa V^T block")
    # FA16-PP (DIAG 15): exponential-phase ping-pong of the two consumer warpgroups (named barriers 4 / 5, 256 threads)
    ppc = f"(DIAG == 15 || (DIAG == 0 && {'true' if pp else 'false'}))"
    t = _one(t, "__device__ __forceinline__ void bar_named(int id, int n) {",
             '__device__ __forceinline__ void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }' + NL
             + "__device__ __forceinline__ void bar_named(int id, int n) {", "vfa bar_named")
    t = _one(t, "  int n = 0, st_c = 0, buf = 0;" + NL,
             "  int n = 0, st_c = 0, buf = 0;" + NL + f"  constexpr bool PP = {ppc};" + NL
             + "  if (PP && h == 1) bar_arrive(4, 256);   // FA16-PP: warpgroup 0 exponentiates first" + NL, "vfa consumer counters")
    t = _one(t, "        pack();" + NL + "        if (!first && DIAG != 4) {   // lazy offset",
             "        if (PP) bar_named(4 + h, 256);   // FA16-PP: my turn" + NL
             + "        pack();" + NL + "        if (!first && DIAG != 4) {   // lazy offset", "vfa pack call")
    t = _one(t, "        first = false;" + NL + "        if (DIAG != 2) {",
             "        if (PP) bar_arrive(5 - h, 256);   // FA16-PP: the other warpgroup's turn" + NL
             + "        first = false;" + NL + "        if (DIAG != 2) {", "vfa first=false")
    t = _one(t, "    consumer<ST, DIAG>(p, smem, seg, sN, nseg, full, empt, misc, tid, pair, pairs);" + NL,
             "    consumer<ST, DIAG>(p, smem, seg, sN, nseg, full, empt, misc, tid, pair, pairs);" + NL
             + f"    if ({ppc} && tid < 128) bar_named(4, 256);   // FA16-PP: consume warpgroup 1's last hand-over" + NL, "vfa consumer call")
    a = t.index(TAIL_A)
    b = t.index(TAIL_B, a)
    assert t[b - 8:b] == '      }' + NL, 'fa16_patches: tail brace'
    b -= 8   # keep the closing brace of the DIAG != 1 branch
    blk = t[a:b]
    assert t.count(blk) == 1 and blk.count("wg_pv(") == 2, "fa16_patches: tail block"
    t = t[:a] + PVS_HEAD + blk + PVS_TAIL + t[b:]
    if lab:
        t = _one(t, "      case 6: vfa_go<ST, 6, true>(p, grid, stream, pdl); return;" + NL,
                 "      case 6: vfa_go<ST, 6, true>(p, grid, stream, pdl); return;" + NL
                 + "".join(f"      case {d}: vfa_go<ST, {d}, true>(p, grid, stream, pdl); return;" + NL for d in (12, 13, 14, 15, 16)),
                 "vfa diag 6")
    return t

"""sfx2_patches.py: SFX2 (R-sfx-3, notes/lab.md) text patches on a layer file (k3_gc_19 and later): the SFX data dependency
for the two other rings whose empty-barrier arrive issued with LDS write scoreboards open (tools/rsx_sass.py audit).

  VA  (va1_main, K1_CUDA_SRC): lane-0 empt arrive after tile_compute1<4> had the last V-fragment LDS.64s open (the HMMAs
      consuming them sink past the arrive). tile_compute1 gains `uint32_t* vdep = nullptr`; its last kk step xors the 8 V
      words into *vdep; the arrive address adds (vdep & opaque 0), opaque 0 = (uint32_t)p.page_shift >> 16 (log2 page).
  K2A (k1/k2_kernel, k1/k2_kernel36, K2A33_KERNEL_SRC; MoE fallback behind VMS / VML): the empty_a arrives after
      consume_kb (main loop + the NC == 1 cold loop) and hc_item had the last x LDS.128s open; refill is TMA (async
      proxy) = the SFX class. x words xored into kdep; arrive + (kdep & (uint32_t)P.hot_mode >> 30) (hot_mode: bits 0-1).
SFX2 defaults to 1 here (ship form); -DSFX2=0 gives the base code (SFX2_Z folds to 0u, the xors are dead).
In-order LDS return (ptxas's shared-SB DEPBAR counts assume it) covers each slot's earlier loads, as in SFX.
"""

PRE = "#ifndef SFX2\n#define SFX2 1\n#endif\n#define SFX2_Z(e) ((SFX2) ? (uint32_t)(e) : 0u)\n"


def _sub(s, old, new, n=1):
    if s.count(old) != n:
        raise SystemExit(f"[sfx2_patches] anchor count {s.count(old)} != {n}: {old[:70]!r}")
    return s.replace(old, new)


def _sub_after(s, before, old, new):
    """replace the first `old` after the (unique) anchor `before`"""
    if s.count(before) != 1:
        raise SystemExit(f"[sfx2_patches] anchor count {s.count(before)} != 1: {before[:70]!r}")
    i = s.index(before) + len(before)
    j = s.index(old, i)
    return s[:j] + new + s[j + len(old):]


ARR = "if (lane == 0) mbar_arrive(empty_a + 8 * slot);"
ARR_K = "if (lane == 0) mbar_arrive(empty_a + 8 * slot + SFX2_Z(kdep & ((uint32_t)P.hot_mode >> 30)));   // R-sfx2"


def patch(t):
    # headers (K1 first in the v10 combined build; K2A33 also builds alone on the fallback paths)
    t = _sub(t, 'K1_CUDA_SRC = r"""\n', 'K1_CUDA_SRC = r"""\n' + PRE)
    t = _sub(t, 'K2A33_KERNEL_SRC = r"""\n', 'K2A33_KERNEL_SRC = r"""\n' + PRE)
    # VA
    t = _sub(t, "const int key_lim, const int seq, const int W) {",
             "const int key_lim, const int seq, const int W, uint32_t* vdep = nullptr) {")
    t = _sub(t, "    v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01, w23);\n",
             "    v_frags(Vbuf, kw + 16 * kk + 2 * t, h, g, w01, w23);\n"
             "    if (SFX2 && vdep && kk == NT / 2 - 1)   // R-sfx2: w01[0] / w23[0] read all four last V LDS.64s (va..vd)\n"
             "      *vdep ^= w01[0] ^ w23[0];\n")
    old = "      tile_compute1<4>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, ts + TILE > cend, ts, cend, seq, 0);\n"
    t = _sub(t, old, "      uint32_t vdep = 0u;\n" + old.replace("seq, 0);", "seq, 0, &vdep);"))
    va = r'''if (lane == 0) asm volatile("{\n.reg .b64 s;\nmbarrier.arrive.shared.b64 s, [%0];\n}\n" ::"r"(va_sa(empt + st)) : "memory");'''
    t = _sub(t, va, va.replace('va_sa(empt + st))', 'va_sa(empt + st) + SFX2_Z(vdep & ((uint32_t)p.page_shift >> 16)))') + "   // R-sfx2")
    # K2A consume_kb: optional x-dependency out
    t = _sub(t, "int kloc, int nch, int t, float (*acc)[4]) {", "int kloc, int nch, int t, float (*acc)[4], uint32_t* kdep = nullptr) {")
    t = _sub(t, "      const uint4 x0 = xp[0], x1 = xp[1];\n",
             "      const uint4 x0 = xp[0], x1 = xp[1];\n      if (SFX2 && kdep) *kdep ^= x0.w ^ x1.w;   // R-sfx2\n")
    xr = "const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST;\n"
    t = _sub(t, xr, xr + "      uint32_t kdep = 0u;   // R-sfx2\n", n=2)
    c1 = "consume_kb<NC>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, nch, t, (i < 2 || NC > 1) ? acc : acc2);"
    t = _sub(t, c1, c1[:-2] + ", (i == 3) ? &kdep : nullptr);")
    t = _sub_after(t, c1[:-2] + ", (i == 3) ? &kdep : nullptr);", ARR, ARR_K)
    c2 = "consume_kb<1>(wv[i], sv[i >> 1], i & 1, xrow, i * 64, 1, t, (i < 2) ? acc : acc2);"
    t = _sub(t, c2, c2[:-2] + ", (i == 3) ? &kdep : nullptr);")
    t = _sub_after(t, c2[:-2] + ", (i == 3) ? &kdep : nullptr);", ARR, ARR_K)
    # K2A hc_item
    xh = "const __nv_bfloat16* xrow = reinterpret_cast<const __nv_bfloat16*>(st + SW) + g * XST + 16 * t;\n"
    t = _sub(t, xh, xh + "    uint32_t kdep = 0u;   // R-sfx2\n")
    xv = "const uint4 xv = *reinterpret_cast<const uint4*>(xrow + q * 8 * XST + i * 64 + 8 * hf);   // x0 / x1\n"
    t = _sub(t, xv, xv + "          if (SFX2 && i == 3) kdep ^= xv.w;   // R-sfx2: last k64 step's x (in-order LDS return)\n")
    t = _sub_after(t, "if (SFX2 && i == 3) kdep ^= xv.w;   // R-sfx2", ARR, ARR_K)
    return t

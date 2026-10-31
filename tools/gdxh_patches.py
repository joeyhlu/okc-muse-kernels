"""gdxh_patches.py (R-gdxh): token-half X (XH) for the MT >= 2 GD kernels of kernels/lab/k3_hh_3tekv.py.

  patch(GD_CUDA_SRC) -> source in which XP bit 8 selects XH = 2: the unit loop holds the x fragments of HALF the token tiles at a time
  (X[XB][MT/2][2] = 16 registers at MT 4 instead of 32), the second half is loaded after the first half's MMAs have issued, into the same
  registers. Everything else (decode, weight reload, patch loop, accumulators, epilogue, GC cat code) is untouched; XP without bit 8 compiles
  to the shipped kernel, byte for byte (XH = 1, MTH = MT).

Why (R-gdncu-0, SASS of the final file): the MT 4 routes spill (qkv 96 regs 72 B st / 212 B ld; pooler / unpooler 128 regs 24 / 360 B; the x base
pointers and the escape state are reloaded every unit) and issue the NEXT unit's three weight loads after 15 of the 16 HMMAs of the current unit
('H L H L H5 X H8 X H W3 X6'), while the spill-free MT 2 / MT 1 kernels issue them right after the first HMMA ('H X3 W3 H7 X'). At ST 1 that
serialises the DRAM flight behind the whole MMA phase. Halving the live X registers is the smallest change that gives the allocator the 12
registers it needs to keep the reload ahead of the MMAs.
"""

OLD = [
    # (anchor, replacement) pairs; every anchor must occur exactly once
    ("constexpr int XB = (XP & 1) ? 2 : 1;\n",
     "constexpr int XB = (XP & 1) ? 2 : 1;\n"
     "constexpr int XH = (((XP & 8) != 0) && ((XP & 1) == 0) && MT >= 2) ? 2 : 1;   // token halves per unit (R-gdxh)\n"
     "constexpr int MTH = MT / XH;\n"),
    ("uint4 X[XB][MT][2];\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MT; ++mt) {\n"
     "X[0][mt][0] = ldxk<XP>(xb + xo[mt]);\n"
     "X[0][mt][1] = ldxk<XP>(xb + xo[mt] + 1);\n"
     "}\n",
     "uint4 X[XB][MTH][2];\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MTH; ++mt) {\n"
     "X[0][mt][0] = ldxk<XP>(xb + xo[mt]);\n"
     "X[0][mt][1] = ldxk<XP>(xb + xo[mt] + 1);\n"
     "}\n"),
    ("if (s + 1 < KU) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MT; ++mt) {\n"
     "X[(s + 1) & (XB - 1)][mt][0] = ldxk<XP>(GC_XP(s + 1) + xo[mt]);\n",
     "if (s + 1 < KU) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MTH; ++mt) {\n"
     "X[(s + 1) & (XB - 1)][mt][0] = ldxk<XP>(GC_XP(s + 1) + xo[mt]);\n"),
    ("} else if (s > 0) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MT; ++mt) {\n"
     "X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[mt]);\n",
     "} else if (s > 0) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MTH; ++mt) {\n"
     "X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[mt]);\n"),
    ("#pragma unroll\n"
     "for (int q = 0; q < 4; ++q) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MT; ++mt) {\n"
     "const uint4 xv = X[xs][mt][q >> 1];\n"
     "const uint32_t b0 = (q & 1) ? xv.z : xv.x;\n"
     "const uint32_t b1 = (q & 1) ? xv.w : xv.y;\n"
     "mma16816(acc[q % NC][mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);\n"
     "}\n"
     "}\n",
     "#pragma unroll\n"
     "for (int h = 0; h < XH; ++h) {\n"
     "if (h > 0) {   // second token half: its x goes into the registers of the first half (free once those MMAs have issued)\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MTH; ++mt) {\n"
     "X[0][mt][0] = ldxk<XP>(GC_XP(s) + xo[h * MTH + mt]);\n"
     "X[0][mt][1] = ldxk<XP>(GC_XP(s) + xo[h * MTH + mt] + 1);\n"
     "}\n"
     "}\n"
     "#pragma unroll\n"
     "for (int q = 0; q < 4; ++q) {\n"
     "#pragma unroll\n"
     "for (int mt = 0; mt < MTH; ++mt) {\n"
     "const uint4 xv = X[xs][mt][q >> 1];\n"
     "const uint32_t b0 = (q & 1) ? xv.z : xv.x;\n"
     "const uint32_t b1 = (q & 1) ? xv.w : xv.y;\n"
     "mma16816(acc[q % NC][h * MTH + mt], a[4 * q], a[4 * q + 1], a[4 * q + 2], a[4 * q + 3], b0, b1);\n"
     "}\n"
     "}\n"
     "}\n"),
]


def patch(src):
    out = src
    for old, new in OLD:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"[gdxh] anchor found {n}x, expected 1: {old[:80]!r}")
        out = out.replace(old, new)
    return out

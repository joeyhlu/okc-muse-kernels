"""sfx_patches.py: SFX (R-sfx-0, notes/lab.md) text patch on a layer's VML CUDA source (k3_gc_18 and later).

SFX defaults to 0 = the shipped code (same SASS). SFX=1 orders stage_m's (mma.sync path, N <= 16) two empty-barrier arrives
after the data return of the shared loads they release (gc_18 SASS: SYNCS.ARRIVE is issued with the LDS / LDSM write
scoreboards still open; only the operand-read barriers are waited, __syncwarp is BRA.DIV + NOP):
  weight-empty  arrive moved after the x-full wait (+ __syncwarp), address data-dependent on all six code / scale words
                ((c0.x ^ c1.x ^ c2.x ^ c3.x ^ sw0 ^ sw1) & opaque 0) -> ptxas must wait the LDS scoreboards first.
  x-empty       address data-dependent on b0 of every ldmatrix of the last k64 step (i == 3; in-order LDS/LDSM return,
                which ptxas itself relies on for shared-scoreboard DEPBAR counts, covers the earlier steps).
opaque 0 = (uint32_t)A.flags >> 30 (flags < 2^30, as UNP SMQ's k16q). Producers prod_w / prod_x are separate warps and the
RS stage<N> already releases its weight slot after the x-full wait, so the move adds no wait cycle.
"""

PRE = """#ifndef SFX
#define SFX 0
#endif
"""

W_OLD = ("const uint32_t sw0 = sp[0], sw1 = sp[32];\n__syncwarp();\nif (lane == 0) mbar_arrive(bars + 8 * (NS + ws));\n"
         "mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);\n__syncwarp();\n")
W_NEW = ("const uint32_t sw0 = sp[0], sw1 = sp[32];\n#if SFX   // R-sfx: weight slot released after its LDS data returned\n"
         "const uint32_t zom = (uint32_t)A.flags >> 30;   // opaque 0\n"
         "mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);\n__syncwarp();\n"
         "if (lane == 0) mbar_arrive(bars + 8 * (NS + ws) + ((c[0].x ^ c[1].x ^ c[2].x ^ c[3].x ^ sw0 ^ sw1) & zom));\n"
         "__syncwarp();\nuint32_t xdep = 0u;\n#else\n"
         "__syncwarp();\nif (lane == 0) mbar_arrive(bars + 8 * (NS + ws));\n"
         "mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);\n__syncwarp();\n#endif\n")

B_OLD = ("mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);\n}\n#endif\n}\n}\n__syncwarp();\n"
         "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));\n++it;\n}\n")
B_NEW = ("mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);\n#if SFX\nif (i == 3) xdep ^= b0;\n#endif\n}\n#endif\n}\n}\n"
         "__syncwarp();\n#if SFX   // R-sfx: x slot released after the last k64 step's ldmatrix data returned\n"
         "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs) + (xdep & zom));\n#else\n"
         "if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));\n#endif\n++it;\n}\n")


def _sub(s, old, new, n=1):
    if s.count(old) != n:
        raise SystemExit(f"[sfx_patches] anchor count {s.count(old)} != {n}: {old[:60]!r}")
    return s.replace(old, new)


def patch(src):
    s = PRE + src
    s = _sub(s, W_OLD, W_NEW)
    s = _sub(s, B_OLD, B_NEW)
    return s

"""dequant_diet_check.py ($0, CPU, notes/study/dequant_diet.md): exhaustive exactness of the shipped VML lane math
(UNP L2 bit placement + per-k64 scale pair `sgp = prmt(sw) * 16 + 0x75807580` + HMUL2.BF16 RN) against the NVFP4 reference
value e2m1 * e4m3 (* the 2^-8 that K2A_W_SHIFT = 256 puts back in the epilogue), over every E4M3 scale byte (NaN excluded,
after _k2a_reencode_scales) x every 8-bit code byte (two codes = one bf16x2 lane pair), plus the same check for any candidate
lane formula registered in CANDS. Pure Python (fractions), no torch / GPU.

    py -3 -I tools/dequant_diet_check.py
"""
from fractions import Fraction

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e4m3_value(b):
    s, e, m = b >> 7, (b >> 3) & 15, b & 7
    if e == 15 and m == 7:
        return None                                    # NaN
    v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
    return -v if s else v


def reencode(b):
    """port of k3 _k2a_reencode_scales for one byte: (bp, neg, zero)"""
    s, ex, m = b >> 7, (b >> 3) & 15, b & 7
    zero = ex == 0 and m == 0
    if ex == 0 and m != 0:
        bp = ((0, 1, 2, 2, 3, 3, 3, 3)[m] << 3) | (0, 0, 0, 4, 0, 2, 4, 6)[m]
    else:
        bp = ((ex + 3) << 3) | m
    if zero:
        bp = 24
    return bp, (s == 1) and not zero, zero


def bf16_value(h):
    s, e, m = h >> 15, (h >> 7) & 0xFF, h & 0x7F
    assert e != 0xFF
    v = Fraction(m, 128) * Fraction(2) ** -126 if e == 0 else (1 + Fraction(m, 128)) * Fraction(2) ** (e - 127)
    return -v if s else v


def bf16_rn(v):
    """round a Fraction to the nearest bf16 (ties to even); returns (bits, exact?)"""
    if v == 0:
        return 0, True
    s = 1 if v < 0 else 0
    a = -v if s else v
    e = 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    if e < -126:                                       # subnormal
        q = abs(v) / Fraction(2) ** -133
        n = int(q)
        r = q - n
        if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
            n += 1
        return (s << 15) | n, r == 0
    q = (a - 1) * 128
    n = int(q)
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
        n += 1
    if n == 128:
        n, e = 0, e + 1
    return (s << 15) | ((e + 127) << 7) | n, r == 0


def ship_lane(code, bp):
    """one bf16 lane of the shipped path: L2 places (s, e1, e0, m) at bits (15, 8, 7, 6); HMUL2 by the sgp lane."""
    w = ((code >> 3) << 15) | ((code & 7) << 6)
    sg = (bp << 4) + 0x7580
    return bf16_rn(bf16_value(w) * bf16_value(sg))


CANDS = {"ship": ship_lane, "neg_ctrl(mantissa lsb flipped)": lambda c, bp: ship_lane(c, bp ^ 1)}


def main():
    for name, fn in CANDS.items():
        total = bad = inexact = 0
        for sb in range(256):
            sv = e4m3_value(sb)
            if sv is None:
                continue
            bp, neg, zero = reencode(sb)
            for cb in range(256):
                for code in (cb & 15, cb >> 4):
                    c = 0 if zero else (code ^ 8 if neg else code)
                    bits, exact = fn(c, bp)
                    ref = (-1 if code >> 3 else 1) * Fraction(E2M1[code & 7]).limit_denominator() * sv * Fraction(1, 256)
                    total += 1
                    inexact += not exact
                    if bf16_value(bits) != ref:
                        bad += 1
                        if bad <= 5:
                            print(f"  MISMATCH {name}: scale 0x{sb:02x} code {code} -> {float(bf16_value(bits))} ref {float(ref)}")
        print(f"[{name}] lanes checked {total}, mismatches {bad}, rounded (inexact) products {inexact}")


if __name__ == "__main__":
    main()

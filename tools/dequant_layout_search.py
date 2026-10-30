"""dequant_layout_search.py ($0, CPU, notes/study/dequant_diet.md): exhaustive search of exact NVFP4 -> BF16 bit-placement
layouts by SASS pipe cost. A bf16x2 output register must hold (sign, e1, e0, m) of two codes at bits M = {15,8,7,6} + {31,24,23,22}
(forced: exponent offset 0 is the only one where code 000 -> 0 and 001 -> 0.5 come out exact, see the study note), every other
bit zero. Each output = (source op) & M, the sources must partition the code bits. Source ops and cost (ALU, FMA-heavy):
  id (the word itself) 1,0 | rotl r (SHF.L.W) 2,0 | shr k (SHF.R) 2,0 | PRMT byte permute 2,0 | shl k <= 6 (IMAD.SHL) 1,1 |
  carry-free gather ((w & P) * K, K = 2^a + 2^b [+ 2^c]) 2,1 | 2-word funnel window SHF.L.W(a, b, k) 2,0
Part 1: one 32-bit word (8 codes, 4 outputs). Part 2: two words (16 codes, 8 outputs) with funnel windows, <= 1 gather.

    py -3 -I tools/dequant_layout_search.py
"""
import itertools
import time

M = [6, 7, 8, 15, 22, 23, 24, 31]


def gathers():
    out = {}
    for nt in (2, 3):
        for ks in itertools.combinations(range(0, 26), nt):
            for assign in itertools.product(range(nt), repeat=8):
                src = [M[j] - ks[assign[j]] for j in range(8)]
                if min(src) < 0 or len(set(src)) < 8:
                    continue
                P = set(src)
                copies = [set(q + k for q in P if q + k < 32) for k in ks]
                ok, seen = True, set()
                for c in copies:                      # pairwise-disjoint shifted copies = no carries
                    if c & seen:
                        ok = False
                        break
                    seen |= c
                if ok:
                    for j, p in enumerate(M):         # each M bit hit only by its intended copy
                        if [i for i, c in enumerate(copies) if p in c] != [assign[j]]:
                            ok = False
                            break
                if ok:
                    out.setdefault(frozenset(P), ks)
    return out


def one_word(G):
    opts = {}

    def add(src, name, a, h):
        if src is None or len(set(src)) < 8 or min(src) < 0 or max(src) > 31:
            return
        k = frozenset(src)
        if k not in opts or (a + h, h) < (opts[k][1] + opts[k][2], opts[k][2]):
            opts[k] = (name, a, h)
    for r in range(1, 32):
        add([(p - r) % 32 for p in M], "rotl%d" % r, 2, 0)
    for k in range(1, 7):
        add([p - k for p in M], "shl%d(IMAD)" % k, 1, 1)
    for k in range(1, 32):
        add([p + k for p in M] if max(M) + k < 32 else None, "shr%d" % k, 2, 0)
    for sel in itertools.permutations(range(4)):
        add([sel[p >> 3] * 8 + (p & 7) for p in M], "prmt%s" % (sel,), 2, 0)
    for P, ks in G.items():
        add(sorted(P), "gather%s" % (ks,), 2, 1)
    full, Mf = frozenset(range(32)), frozenset(M)
    idx = {}
    for k, v in opts.items():
        if not (k & Mf):
            idx.setdefault(min(k), []).append((k, v))
    res = {}

    def rec(cov, ch):
        if cov == full:
            a, h = 1 + sum(c[1][1] for c in ch), sum(c[1][2] for c in ch)
            res.setdefault((a, h), []).append([c[1][0] for c in ch])
            return
        if len(ch) < 3:
            for x in idx.get(min(full - cov), []):
                if not (x[0] & cov):
                    rec(cov | x[0], ch + [x])
    rec(Mf, [])
    # layouts without an identity output cost >= 8 ALU-or-heavy ops: never better
    return res


def two_word(G):
    def mk(b):
        m = 0
        for x in b:
            m |= 1 << x
        return m
    S = {}
    for t in range(64):                               # funnel windows of the 64-bit ring [a | b]; t = 0 / 32 are the ids
        src = [(p - t) % 64 for p in M]
        S[mk(src)] = ("id" if t in (0, 32) else "win%d" % t, 1 if t in (0, 32) else 2, 0)
    for off in (0, 32):
        for r in range(1, 32):
            S.setdefault(mk([((p - r) % 32) + off for p in M]), ("rot", 2, 0))
    bylow = [[] for _ in range(64)]
    for m, v in S.items():
        bylow[(m & -m).bit_length() - 1].append((m, v))
    FULL = (1 << 64) - 1

    def count(cov):
        n = 0
        if cov == FULL:
            return 1
        low = ((~cov) & -(~cov)).bit_length() - 1
        for m, v in bylow[low]:
            if not (m & cov):
                n += count(cov | m)
        return n
    n0 = count(0)
    n1 = 0
    for P in G:
        for off in (0, 32):
            n1 += count(mk(p + off for p in P))
    return n0, n1


def main():
    t0 = time.time()
    G = gathers()
    print(f"carry-free 2-3 term gathers: {len(G)} source sets ({time.time() - t0:.0f} s)")
    res = one_word(G)
    print("1 word (ALU, heavy) -> layouts:")
    for k in sorted(res):
        print("  ", k, len(res[k]), res[k][:4])
    n0, n1 = two_word(G)
    print(f"2 words: tilings with 0 gathers {n0}, with exactly 1 gather {n1}  (so every 2-word layout needs >= 2 gathers"
          f" = (14 ALU, 2 heavy) like 2 x L2, or worse)" if n0 == n1 == 0 else f"2 words: 0-gather {n0}, 1-gather {n1}")


if __name__ == "__main__":
    main()

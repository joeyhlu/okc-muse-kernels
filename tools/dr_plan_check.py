"""dr_plan_check.py: CPU replay of the VM_DR integer plan (no GPU, no torch): route()'s 32-token job split, the cost
prefix (256 / 128 per DR stage), stage_at / split_point / phase() segment walk, prod_w / prod_x / self_fill indexing, the
fixup slot and counter protocol, and the doneA credits, transcribed from the k3_gc_6 + VM_DR CUDA. For random routings at
T = 25..48 (and a T = 192 stress) it asserts:
  - the CTA stage ranges partition [0, nj << 8) (A) and [0, nj << 7) (B), every unit's 16 stages are covered once;
  - every split phase-A unit has one owner CTA slot (cta_of of its first stage), no slot serves two split units, and the
    per-(slot, WG) stage counts reach exactly 16 (the epilogue runs once per WG);
  - doneA[j][U] receives 4 WGs x 2 = 8 credits for every phase-A DR unit = every B stage's x slice;
  - each weight-producer source block (K2A layout and VM_DRP repack) is in range and the four blocks of a stage are 4U..4U+3;
  - the self-filled stages are the CTA's first nf phase-A stages in its processing order, as prod_x skips them;
  - 32-bit products of the plan (cp[nj] * c) stay below 2^32.

    python -I tools/dr_plan_check.py [--draws=2000] [--seed=1]
"""
import random
import sys

COST = [128, 167, 183, 229, 128, 162, 184, 215]   # k3_gc_6 VML_COST[32] mapped by _vml_cost_ext for VM_NTM 4


def mdiv(a, m):
    assert 0 <= a < 1 << 32, a
    return (a * m) >> 40


def mag(d):
    return -(-(1 << 40) // d)


def route(counts):
    jobs = []   # (expert, size)
    for e, tot in enumerate(counts):
        nj = (tot + 31) // 32
        base = (tot + nj - 1) // nj if nj else 32
        for k in range(nj):
            jobs.append((e, min(base, tot - k * base)))
    return jobs


def plan(jobs, G):
    nj = len(jobs)
    nt = [((c + 7) >> 3) for _, c in jobs]
    cp = {0: [0], 1: [0]}
    for j in range(nj):
        cp[0].append(cp[0][-1] + 256 * COST[nt[j] - 1])
        cp[1].append(cp[1][-1] + 128 * COST[4 + nt[j] - 1])
    mg, magG = [mag(c) for c in COST], mag(G)

    def stage_at(pb, c):
        if nj == 0:
            return 0
        if c >= G:
            return nj << (8 - pb)
        x = mdiv(cp[pb][nj] * c, magG)
        lo, hi = 0, nj
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if cp[pb][mid] <= x:
                lo = mid
            else:
                hi = mid
        st = mdiv(x - cp[pb][lo], mg[4 * pb + nt[lo] - 1])
        st = min(st, (256 >> pb) - 1)
        return (lo << (8 - pb)) + st
    return {pb: [stage_at(pb, c) for c in range(G + 1)] for pb in (0, 1)}, nt


def split_point(flags, pb, g0, g1):
    ue = ((g1 - 1) >> 4) << 4
    return ue if (flags & 1) and pb == 0 and (g1 & 15) != 0 and ue > g0 else g0


def cta_of(b, s, G):
    lo, hi = 0, G
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if b[mid] <= s:
            lo = mid
        else:
            hi = mid
    return lo


def check(counts, G=132, flags=193, NX=3):
    jobs = route(counts)
    nj = len(jobs)
    assert nj <= 80, f"JMAX: {nj} jobs"
    assert all(1 <= c <= 32 for _, c in jobs)
    b, nt = plan(jobs, G)
    for pb in (0, 1):
        assert b[pb][0] == 0 and b[pb][G] == nj << (8 - pb) and all(b[pb][c] <= b[pb][c + 1] for c in range(G)), "ranges"
    cnt = {}     # (pb, slot, q) -> stages added by split shares, per owner unit
    owner = {}   # (pb, slot) -> unit
    cover = {0: [0] * (nj << 8), 1: [0] * (nj << 7)}
    done = [0] * (nj * 16)
    for c in range(G):
        for pb in (0, 1):
            g0, g1 = b[pb][c], b[pb][c + 1]
            sp = split_point(flags, pb, g0, g1)
            order, gs, ge = [], sp, g1
            while True:                                  # phase<PB>'s segment walk
                if gs >= ge:
                    if ge == g1 and sp > g0:
                        gs, ge = g0, sp
                    else:
                        break
                u, s0 = gs >> 4, gs & 15
                s1 = min(16, s0 + (ge - gs))
                j = (u >> 3) if pb else (u >> 4)
                ub = u & (7 if pb else 15)
                assert 0 <= j < nj and 0 <= ub < (8 if pb else 16)
                for s in range(s0, s1):
                    cover[pb][(u << 4) + s] += 1
                    order.append((u << 4) + s)
                whole = s0 == 0 and s1 == 16
                if not whole and not pb:                 # phase-A fixup (B: BRED red.add, no counters)
                    slot = cta_of(b[pb], u << 4, G)
                    assert owner.setdefault((pb, slot), u) == u, f"slot {slot} serves two split units"
                    for q in range(4):
                        cnt[(pb, slot, q)] = cnt.get((pb, slot, q), 0) + (s1 - s0)
                        if cnt[(pb, slot, q)] == 16:
                            done[j * 16 + ub] += 2       # the last arriver of WG q runs epilogue_dr
                            cnt[(pb, slot, q)] = 0
                elif not pb:
                    done[j * 16 + ub] += 8               # 4 WGs x 2 credits
                gs += s1 - s0
            # producers walk the same order (prod_w / prod_x: gs = sp + i wrapped)
            pw = [(sp + i) if sp + i < g1 else sp + i - (g1 - g0) for i in range(g1 - g0)]
            assert pw == order, "producer / consumer stage order differ"
            for gsx in pw:
                u = gsx >> 4
                j, ub, s = ((u >> 3), u & 7, gsx & 15) if pb else ((u >> 4), u & 15, gsx & 15)
                e = jobs[j][0]
                blocks = [(e * (32 if pb else 64) + 4 * ub + k) for k in range(4)]   # K2A layout blocks of the stage
                assert blocks[-1] < (e + 1) * (32 if pb else 64)
                drp = (e * (8 if pb else 16) + ub) * 16 + s                        # VM_DRP stage index
                assert drp < 32 * (8 if pb else 16) * 16
            if not pb:
                nf = min(NX - 1, g1 - g0) if flags & 64 else 0
                fill = [(sp + f) if sp + f < g1 else sp + f - (g1 - g0) for f in range(nf)]
                assert fill == order[:nf], "self_fill stages"
                assert all((g >> 8) == (g >> 4) >> 4 for g in fill)
    for pb in (0, 1):
        assert all(v == 1 for v in cover[pb]), f"phase {pb} coverage"
    assert all(v == 0 for v in cnt.values()), "fixup counters not reset"
    assert all(done[j * 16 + U] == 8 for j in range(nj) for U in range(16)), "doneA credits"
    return nj


def draw(rng, M, skew):
    counts = [0] * 32
    heat = [rng.random() ** 6 * 4 if skew else 0.03 * rng.gauss(0, 1) for _ in range(256)]
    for _ in range(M):
        sc = sorted(((rng.random() + heat[e], e) for e in range(256)), reverse=True)[:8]
        for _, e in sc:
            if e < 32:
                counts[e] += 1
    return counts


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    rng = random.Random(int(args.get("seed", 1)))
    n, jobs_max, tot_max = int(args.get("draws", 2000)), 0, 0
    for i in range(n):
        M = rng.randint(25, 48)
        counts = draw(rng, M, i % 2 == 1)
        if i % 7 == 0:                          # hot: every token on expert 0..7 (counts up to M)
            counts = [M if e < 8 else 0 for e in range(32)]
        G = rng.choice((132, 132, 114, 64, 16))
        jobs_max = max(jobs_max, check(counts, G))
        tot_max = max(tot_max, max(counts))
    for M in (192, 100):                        # JMAX stress (vml_run TORCH_CHECK M <= 192)
        jobs_max = max(jobs_max, check([M * 8 // 32 + (1 if e < (M * 8) % 32 else 0) for e in range(32)], 132))
    print(f"[dr_plan_check] OK: {n} routings (T 25..48, normal / skewed / hot, G 16..132) + T 100 / 192 stress; "
          f"max jobs {jobs_max} (JMAX 80), max expert tokens {tot_max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

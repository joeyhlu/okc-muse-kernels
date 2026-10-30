"""moe_deal_sim.py: $0 CPU makespan model of VML phase structure on MoE-board routings (moe redesign, R-deal-0).
Muse's ext_muse_bs32.md designs vs the shipped static plan, one model and one set of constants for all three:
  ship : A then B, each phase's cumulative-cost list cut into 132 contiguous stage ranges (splits allowed, VML_COST L
         tables), B of job j waits doneA(j) (+ hop).
  deal : #1 split-free whole-unit deal (A unit = 256 rows x full K = 16 stages, B unit = 256 out rows x full K = 16 stages),
         expert-interleaved list A(e0) A(e1) B(e0) A(e2) B(e1) ... (lag 1), every unit to the CTA free earliest (greedy
         list scheduling with modelled durations = an optimistic bound for any static deal); B(e) waits doneA(e) + hop.
  lag  : #2 the ship's cost-cut contiguous ranges over one global list with B(e) inserted after A(e + L) (L 2 / 4).
Constants (calibrated on mb_11's board bs32 marks: A 79.2 us = 58.2 stages + 4.6 segments + 1.85 splits, B 37.2 us):
  pace nt 1 1.15 us / stage, nt 2 1.50, nt >= 3 per in-layer tiers; A segment end 1.9, B segment end 0.64, split share 1.17,
  doneA hop 1.5, start 3.5 us. Output: median / p90 over routings of the modelled end max.
    python tools/moe_deal_sim.py [--bs=32] [--seeds=60]
"""
import statistics as st
import sys

PACE = [1.15, 1.50, 1.70, 1.81, 1.95, 2.05, 2.15, 2.25, 2.35]
SEG_A, SEG_B, SPLIT, HOP, START = 1.9, 0.64, 1.17, 1.5, 3.5
COST_A = [128, 167, 183, 229, 171, 182, 193, 204, 214]
COST_B = [128, 162, 184, 215, 171, 182, 193, 204, 214]
G = 132


def counts_for(bs, seed):
    sys.path.insert(0, "<repo>/problems/muse_moe_w4a16")
    import torch
    import reference as R
    d = R.generate_input(bs, "ep8", seed, device="cpu")
    ids = d["topk_ids"].long().flatten()
    loc = ids[ids < 32]
    c = torch.bincount(loc, minlength=32)
    return sorted([int(x) for x in c if x > 0], reverse=True)


def jobs_of(counts, jn=64):
    js = []
    for n in counts:
        while n > 0:
            k = min(n, jn)
            js.append((k + 7) // 8)
            n -= k
    return js


def cut(items, cost):
    """items = [(job, phase, nstages)] in list order; cumulative cost cut at c / G -> per CTA list of (job, phase, s0, s1)"""
    tot = sum(ns * cost(j, p) for j, p, ns in items)
    bounds = [tot * c / G for c in range(G + 1)]
    out = [[] for _ in range(G)]
    acc, c = 0.0, 0
    for j, p, ns in items:
        cs = cost(j, p)
        s = 0
        while s < ns:
            while c < G - 1 and acc >= bounds[c + 1] - 1e-9:
                c += 1
            room = int(round((bounds[c + 1] - acc) / cs)) if c < G - 1 else ns - s
            k = max(1, min(ns - s, room))
            out[c].append((j, p, s, s + k))
            acc += k * cs
            s += k
    return out


def run_ranges(rng, nts, nst_unit=16):
    """per-CTA sequential timeline with doneA waits; segments = (job, unit) pieces; split share = partial unit piece"""
    G_ = len(rng)
    t = [START] * G_
    adone = {}
    # pass 1: A pieces in list order per CTA until the first B piece (CTA order is the list order)
    seqs = rng
    pos = [0] * G_
    fin = [0.0] * G_
    # event loop: process CTAs round-robin until all done (B waits may need other CTAs' A first)
    pending = set(range(G_))
    guard = 0
    while pending and guard < 10 ** 6:
        guard += 1
        progressed = False
        for c in list(pending):
            while pos[c] < len(seqs[c]):
                j, p, s0, s1 = seqs[c][pos[c]]
                if p == 1:
                    if j not in adone or len(adone[j]) < A_UNITS_OF[j]:
                        break
                    t[c] = max(t[c], max(adone[j].values()) + HOP)
                n = s1 - s0
                pc = PACE[min(nts[j], len(PACE)) - 1]
                # unit pieces inside [s0, s1): units of 16 stages; a piece that is not a whole unit is a split share
                u0, u1 = s0 // nst_unit, (s1 - 1) // nst_unit
                for u in range(u0, u1 + 1):
                    a, b = max(s0, u * nst_unit), min(s1, (u + 1) * nst_unit)
                    t[c] += (b - a) * pc + (SEG_A if p == 0 else SEG_B) + (SPLIT if b - a < nst_unit else 0.0)
                    if p == 0:
                        adone.setdefault(j, {})
                        adone[j][(u, c)] = max(adone[j].get((u, c), 0.0), t[c])
                pos[c] += 1
                progressed = True
            if pos[c] >= len(seqs[c]):
                fin[c] = t[c]
                pending.discard(c)
        if not progressed:
            break
    return max(fin) if not pending else float("inf")


A_UNITS_OF = {}


def ship(nts):
    items_a = [(j, 0, 512) for j in range(len(nts))]
    items_b = [(j, 1, 256) for j in range(len(nts))]
    ra = cut(items_a, lambda j, p: COST_A[nts[j] - 1])
    rb = cut(items_b, lambda j, p: COST_B[nts[j] - 1])
    for j in range(len(nts)):
        A_UNITS_OF[j] = len({(u, c) for c in range(G) for (jj, p, s0, s1) in ra[c] if jj == j
                             for u in range(s0 // 16, (s1 - 1) // 16 + 1)})
    return run_ranges([ra[c] + rb[c] for c in range(G)], nts)


def lag(nts, L):
    items = []
    n = len(nts)
    for e in range(n + L):
        if e < n:
            items.append((e, 0, 512))
        if e - L >= 0:
            items.append((e - L, 1, 256))
    rr = cut(items, lambda j, p: (COST_A if p == 0 else COST_B)[nts[j] - 1])
    for j in range(n):
        A_UNITS_OF[j] = len({(u, c) for c in range(G) for (jj, p, s0, s1) in rr[c] if jj == j and p == 0
                             for u in range(s0 // 16, (s1 - 1) // 16 + 1)})
    return run_ranges(rr, nts)


def deal(nts):
    """greedy list scheduling of whole 16-stage units, expert-interleaved with lag 1 (optimistic static-deal bound)"""
    import heapq
    n = len(nts)
    order = []
    for e in range(n + 1):
        if e < n:
            order += [(e, 0, u) for u in range(32)]
        if e - 1 >= 0:
            order += [(e - 1, 1, u) for u in range(16)]
    free = [(START, c) for c in range(G)]
    heapq.heapify(free)
    afin = {}
    bq = []
    end = 0.0
    for j, p, u in order:
        tc, c = heapq.heappop(free)
        pc = PACE[min(nts[j], len(PACE)) - 1]
        if p == 1:
            tc = max(tc, max(afin[j]) + HOP)
        tc += 16 * pc + (SEG_A if p == 0 else SEG_B)
        if p == 0:
            afin.setdefault(j, []).append(tc)
        end = max(end, tc)
        heapq.heappush(free, (tc, c))
    return end


def main():
    bs = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--bs=")), 32))
    ns = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--seeds=")), 60))
    rows = []
    for seed in range(400, 400 + ns):
        nts = jobs_of(counts_for(bs, seed))
        r = {"ship": ship(nts), "deal": deal(nts), "lag2": lag(nts, 2), "lag4": lag(nts, 4),
             "floor": START + sum(768 * PACE[min(x, 9) - 1] for x in nts) / G}
        rows.append(r)
    for k in ("floor", "ship", "deal", "lag2", "lag4"):
        v = sorted(r[k] for r in rows)
        print(f"bs{bs} {k:5s}: median {st.median(v):6.1f} p90 {v[int(0.9 * len(v))]:6.1f} us"
              + ("" if k in ("floor", "ship") else f"   vs ship median {st.median(r[k] - r['ship'] for r in rows):+.1f} us"))


if __name__ == "__main__":
    main()

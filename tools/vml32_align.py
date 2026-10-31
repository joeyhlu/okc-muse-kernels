"""vml32_align.py: $0 CPU falsifier for INDEPENDENT_REVIEW_TOP30 s3.2 "VML T=32 split-point alignment" (2026-10-30).

Idea under test: move the VML phase-A cut points onto 16-stage unit edges so the head / tail split shares of a CTA merge
into the neighbouring segment (no share = no RED + CNT + resolver), charging the resulting stage imbalance to the plan.
Kill rule (review): split shares per CTA fall by < 1.5 at T=32, OR the max CTA share grows by > 1 stage.

Reuses the calibrated planner / timeline model of tools/plan_search.py (not edited; imported, its `cut` is wrapped).
Variants (all change the phase-A cuts only unless noted):
  ship     shipped planner (cost cut at floor(tot * c / G), stage granular, split_point rotation)
  snapW    a cut within W stages of a unit edge moves onto it (plan_search `snap`): W=1 is the literal "<= 1 stage
           imbalance" reading, W=8 aligns every cut (nearest edge) = VM_AW's A side (R-aw-1)
  aw       snap8 + B cut carrying the even A+B target (plan_search `bbal`)  = VM_AW as built in R-aw-1 (measured +14.8..+16.4)
  opt      best possible fully aligned A partition in the model: min-max cost over whole-unit ranges (binary search on
           the cap + greedy feasibility), then min sum of squares under that cap (DP); B cut as ship
  opt+bbal same A partition, B cut evens predicted A end + B (plan_search `bbal`)
Usage: python tools/vml32_align.py [--n=60] [--syn=120]
Board set: _mb_board_input(32, seeds 100..159) (60; the first 40 equal tools/plan_search_routings.json), JSON next to this file
(tools/vml32_align_routings.json, drawn on the local 3070, torch CUDA RNG offset emulated like plan_search_routings_gen.py).
Layer set: plan_search.layer(32) = MB_LAYER_PROFILES + box_out/*/run.json in-layer routings (3 expert-id placements each),
and a synthetic in-layer-like set (Zipf popularity, 61.7 routes, 8.33 distinct local experts = notes/study/lb_gc13.md [4]).
"""
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_search as ps  # noqa: E402

G = ps.G
T = 32
_orig_cut = ps.cut


# ---------------------------------------------------------------- aligned partitions
def make_opt(nts, tab, Gc):
    uc = []
    for nt in nts:
        uc += [16 * tab[nt - 1]] * 32
    n = len(uc)
    pre = [0]
    for c in uc:
        pre.append(pre[-1] + c)

    def feasible(M):
        i, used = 0, 0
        while i < n and used < Gc:
            j = i
            while j < n and pre[j + 1] - pre[i] <= M:
                j += 1
            if j == i:
                return False
            i, used = j, used + 1
        return i >= n

    lo, hi = max(uc), pre[-1]
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    M = lo
    # DP: min sum of squares of range costs under cap M; f[c][i] = best for first i units on first c CTAs
    INF = float("inf")
    kmax, j = 1, 0
    for i in range(n):   # widest window of whole units whose cost fits the cap
        j = max(j, i)
        while j < n and pre[j + 1] - pre[i] <= M:
            j += 1
        kmax = max(kmax, j - i)
    f = [INF] * (n + 1)
    f[0] = 0.0
    back = []
    for c in range(Gc):
        g = [INF] * (n + 1)
        bk = [0] * (n + 1)
        for i in range(n + 1):
            for k in range(0, min(i, kmax) + 1):
                if f[i - k] == INF:
                    continue
                w = pre[i] - pre[i - k]
                if w > M:
                    break
                v = f[i - k] + float(w) * w
                if v < g[i]:
                    g[i], bk[i] = v, k
        f = g
        back.append(bk)
    cuts = [0] * (Gc + 1)
    i = n
    cuts[Gc] = n
    for c in range(Gc - 1, -1, -1):
        i -= back[c][i]
        cuts[c] = i
    return [u * 16 for u in cuts], M


def cut2(nts, ns, tab, Gc, sg=0, snap=0, grp=0, targets=None):
    if isinstance(snap, tuple) and snap[0] == "opt" and ns == 512:
        return make_opt(nts, tab, Gc)[0]
    return _orig_cut(nts, ns, tab, Gc, sg, snap, grp, targets)


ps.cut = cut2


# ---------------------------------------------------------------- plan metrics (A phase), replicating simulate()'s cuts
def a_plan(plan, counts):
    jn = plan.get("jn", 64)
    nts = ps.jobs_of(counts, jn, plan.get("ta", True), plan.get("order", "id"))
    b8 = ps.MB_LAYER_COST[32]
    tabA, _ = ps.cost_ext(list(b8[:8]) + list(b8[8:]), max(8, max(nts), -(-jn // 8)))
    bA = ps.cut(nts, 512, tabA, G, plan.get("sg", 0), plan.get("snap", 0), plan.get("grp", 0))
    return nts, tabA, bA


def a_metrics(plan, counts, rot=True):
    nts, tabA, bA = a_plan(plan, counts)
    pA = [ps.pieces(bA[c], bA[c + 1], rot, 9) for c in range(G)]
    act = [c for c in range(G) if pA[c]]
    sh = [sum(1 for _, s0, s1 in pA[c] if not (s0 == 0 and s1 == 16)) for c in act]
    # head share: range starts mid-unit; tail share: range ends mid-unit (a single-unit range counts once)
    ht = 0
    for c in act:
        g0, g1 = bA[c], bA[c + 1]
        if (g0 & 15) and (g1 & 15) and (g0 >> 4) != ((g1 - 1) >> 4):
            ht += 1
    stg = [bA[c + 1] - bA[c] for c in act]
    cst = [sum((s1 - s0) * tabA[nts[u >> 5] - 1] for u, s0, s1 in pA[c]) / tabA[0] for c in act]   # nt1-stage equivalents
    return dict(shares=sum(sh) / len(act), ht=ht / len(act), segs=sum(len(pA[c]) for c in act) / len(act),
                maxstg=max(stg), maxcost=max(cst), meancost=sum(cst) / len(cst), nact=len(act), nunits=32 * len(nts),
                nj=len(nts))


# ---------------------------------------------------------------- routings
def board60():
    return json.load(open(os.path.join(HERE, "vml32_align_routings.json")))["32"]


def syn_layer(n, seed=7, target=8.33, routes=61.7):
    """Zipf popularity over 32 local experts tuned to 61.7 routes / 8.33 distinct (lb_gc13.md [4]); target / routes
    override it for the 'rich' bracket (more distinct experts = more units per CTA = finer alignment quantum)"""
    rng = random.Random(seed)
    best = None
    for a20 in range(8, 80):
        a = a20 / 20.0
        w = [(r + 1) ** -a for r in range(32)]
        s = sum(w)
        p = [x / s for x in w]
        dist = sum(1 - (1 - x) ** routes for x in p)
        if best is None or abs(dist - target) < abs(best[1] - target):
            best = (a, dist, p)
    a, dist, p = best
    out = []
    for _ in range(n):
        N = max(40, int(rng.gauss(routes, 8)))
        ids = rng.sample(range(32), 32)   # random placement of the popularity ranks onto local expert ids
        v = [0] * 32
        for _ in range(N):
            r = rng.random()
            acc = 0.0
            k = 31
            for i, x in enumerate(p):
                acc += x
                if r < acc:
                    k = i
                    break
            v[ids[k]] += 1
        out.append(v)
    return out, a, dist


PLANS = {
    "ship": {},
    "snap1": {"snap": 1}, "snap2": {"snap": 2}, "snap3": {"snap": 3}, "snap4": {"snap": 4}, "snap6": {"snap": 6},
    "snap8(all)": {"snap": 8},
    "aw(snap8+bbal)": {"snap": 8, "bbal": 1},
    "opt": {"snap": ("opt",)},
    "opt+bbal": {"snap": ("opt",), "bbal": 1},
}


def run_set(name, rs, P, a, C0):
    ship_end = [ps.simulate({}, r, T, P, a, C0) for r in rs]
    ship_m = [a_metrics({}, r) for r in rs]
    print(f"\n== {name}: {len(rs)} routings, ship end median {st.median(ship_end):.1f} us, jobs/routing "
          f"{st.mean(m['nj'] for m in ship_m):.1f}, units/CTA {st.mean(m['nunits'] / m['nact'] for m in ship_m):.2f}, "
          f"ship A shares/CTA {st.mean(m['shares'] for m in ship_m):.2f}, head+tail CTAs {st.mean(m['ht'] for m in ship_m) * 100:.0f} %, "
          f"A segs/CTA {st.mean(m['segs'] for m in ship_m):.2f}")
    print(f"{'plan':16s} {'shares/CTA':>11s} {'d shares':>9s} {'segs/CTA':>9s} {'maxA stg':>9s} {'d max stg (med/p90)':>20s} "
          f"{'d max cost(nt1 stg)':>20s} {'d end us med [p10,p90]':>26s}")
    rows = {}
    allends = {}
    for nm, pl in PLANS.items():
        ms = [a_metrics(pl, r) for r in rs]
        ends = [ps.simulate(pl, r, T, P, a, C0) for r in rs]
        de = sorted(e - s for e, s in zip(ends, ship_end))
        dmax = sorted(m['maxcost'] - s['maxcost'] for m, s in zip(ms, ship_m))
        dstg = sorted(m['maxstg'] - s['maxstg'] for m, s in zip(ms, ship_m))
        dsh = st.mean(m['shares'] - s['shares'] for m, s in zip(ms, ship_m))
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]   # noqa: E731
        rows[nm] = dict(shares=st.mean(m['shares'] for m in ms), dshares=dsh, segs=st.mean(m['segs'] for m in ms),
                        maxstg=st.mean(m['maxstg'] for m in ms), dstg=st.median(dstg), dstg90=q(dstg, .9),
                        dcost=st.median(dmax), dcost90=q(dmax, .9), de=st.median(de), de10=q(de, .1), de90=q(de, .9))
        allends[nm] = ends
        r = rows[nm]
        print(f"{nm:16s} {r['shares']:11.2f} {r['dshares']:+9.2f} {r['segs']:9.2f} {r['maxstg']:9.1f} "
              f"{r['dstg']:+9.1f} /{r['dstg90']:+6.1f}   {r['dcost']:+9.1f} /{r['dcost90']:+6.1f}   "
              f"{r['de']:+8.1f} [{r['de10']:+6.1f},{r['de90']:+6.1f}]")
    cand = ["ship", "snap1", "snap2", "snap3", "snap4", "snap6", "snap8(all)", "opt"]
    orc = sorted(min(allends[k][i] for k in cand) - ship_end[i] for i in range(len(rs)))
    print(f"{'oracle best-of':16s} (per routing, min over ship / snap1..8 / opt; selection on model noise = upper bound) "
          f"d end median {st.median(orc):+.1f} mean {st.mean(orc):+.1f} [{orc[len(orc) // 10]:+.1f},{orc[(9 * len(orc)) // 10]:+.1f}] us")
    rows["oracle"] = dict(de=st.median(orc), de10=orc[len(orc) // 10], de90=orc[(9 * len(orc)) // 10])
    return rows


def calib32(P):
    rs6 = ps.board(32, 6)
    ys = [st.mean(r[i] for r in ps.MEAS[32]) for i in range(6)]
    best = None
    for a100 in range(80, 131):
        a = a100 / 100.0
        e = [ps.simulate({}, rs6[i], 32, P, a, 0.0) for i in range(6)]
        C0 = st.mean(ys[i] - e[i] for i in range(6))
        sse = sum((ys[i] - e[i] - C0) ** 2 for i in range(6))
        if best is None or sse < best[0]:
            best = (sse, a, C0)
    return best[1], best[2], (best[0] / 6) ** 0.5


def sens(rb, lay):
    """how expensive must a split share be before alignment pays? RED / CNT / RB scaled by m, refit each time"""
    print("\n== sensitivity: split-share sub-steps (RED, CNT, RB) x m, refit per m; median d end vs ship, us")
    for m in (1.0, 2.0, 3.0, 4.0):
        sub = dict(ps.SUB["ps"])
        ov = {k: sub[k] * m for k in ("RED", "CNT", "RB")}
        P = dict(ps.DEF, subov=ov)
        a, C0, rms = calib32(P)
        line = f"m={m:.0f} (share {ov['RED'] + ov['CNT']:.1f} us, resolver +{ov['RB'] + sub['EPI'] + sub['PF']:.1f}; fit rms {rms:.2f})"
        for nm, rs in (("b32", rb), ("L32", lay)):
            ship = [ps.simulate({}, r, T, P, a, C0) for r in rs]
            for pn in ("snap2", "snap4", "snap6", "snap8(all)", "opt"):
                d = st.median(ps.simulate(PLANS[pn], r, T, P, a, C0) - s0 for r, s0 in zip(rs, ship))
                line += f" | {nm} {pn} {d:+.1f}"
        print(line)


def budget(name, rs, P, a, C0):
    """savings vs imbalance: mean per-CTA overhead (EPI / PF / RED / CNT / resolver / BEND) and the A-end spread / max"""
    print(f"\n== budget {name} (model detail, medians over routings)")
    print(f"{'plan':16s} {'ovh/CTA us':>11s} {'d ovh':>7s} {'Aend max':>9s} {'d Aend max':>11s} {'Aend max-p50':>13s} {'max-mean end':>13s} {'B wait/CTA':>11s} {'d end':>7s}")
    base = None
    for nm in ("ship", "snap1", "snap4", "snap6", "snap8(all)", "opt"):
        dd = [ps.simulate(PLANS[nm], r, T, P, a, C0, detail=True) for r in rs]
        if base is None:
            base = dd
        med = lambda k: st.median(x[k] for x in dd)   # noqa: E731
        print(f"{nm:16s} {med('ovm'):11.2f} {st.median(x['ovm'] - b['ovm'] for x, b in zip(dd, base)):+7.2f} {med('aend_max'):9.1f} "
              f"{st.median(x['aend_max'] - b['aend_max'] for x, b in zip(dd, base)):+11.1f} "
              f"{st.median(x['aend_max'] - x['aend_p50'] for x in dd):13.1f} {med('bal'):13.1f} {med('wait'):11.2f} "
              f"{st.median(x['end'] - b['end'] for x, b in zip(dd, base)):+7.1f}")


def main():
    n = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--n=")), 60))
    nsyn = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--syn=")), 120))
    P = dict(ps.DEF)
    # calibrate bucket 32 only (same procedure as ps.calibrate)
    rs6 = ps.board(32, 6)
    ys = [st.mean(r[i] for r in ps.MEAS[32]) for i in range(6)]
    best = None
    for a100 in range(80, 131):
        a = a100 / 100.0
        e = [ps.simulate({}, rs6[i], 32, P, a, 0.0) for i in range(6)]
        C0 = st.mean(ys[i] - e[i] for i in range(6))
        sse = sum((ys[i] - e[i] - C0) ** 2 for i in range(6))
        if best is None or sse < best[0]:
            best = (sse, a, C0)
    _, a, C0 = best
    print(f"calibration bucket 32: pace a {a:.2f}, C0 {C0:.1f} us, rms {(best[0] / 6) ** 0.5:.2f} us (plan_search: 1.06)")
    out = {}
    out["board"] = run_set("board bs32, seeds 100..159", board60()[:n], P, a, C0)
    lay = ps.layer(32)
    out["layer"] = run_set("in-layer T=32 (18 saved profiles x 3 id placements)", lay, P, a, C0)
    syn, za, zd = syn_layer(nsyn)
    out["syn"] = run_set(f"in-layer-like synthetic (Zipf a={za:.1f}, mean distinct {st.mean(sum(1 for x in r if x) for r in syn):.2f}, "
                         f"routes {st.mean(sum(r) for r in syn):.1f})", syn, P, a, C0)
    rich, ra, rd = syn_layer(nsyn, seed=11, target=11.0, routes=72.0)
    out["rich"] = run_set(f"in-layer-like 'rich' bracket (Zipf a={ra:.2f}, mean distinct {st.mean(sum(1 for x in r if x) for r in rich):.2f}, "
                          f"routes {st.mean(sum(r) for r in rich):.1f}; ~ the upper end of the lat32 marks' 2.9-3.8 A segs / CTA)", rich, P, a, C0)
    budget('board bs32', board60()[:n], P, a, C0)
    budget('in-layer T=32 (saved profiles)', lay, P, a, C0)
    sens(board60()[:n], lay)
    if os.environ.get("VML32_OUT"):
        json.dump(out, open(os.path.join(os.environ["VML32_OUT"], "vml32_align_out.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

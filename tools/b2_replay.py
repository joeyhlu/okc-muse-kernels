"""$0 CPU replay for Muse v2 B2 (hot-expert local-B tile) on the 16 measured in-layer d16 routings (R-v16m-1).

Model (per CTA, us): start at the measured first-unit time; A units in plan order at c1 (1-tile) / c2 (2-tile) us; job j's A
completes at the max over CTAs of the time they finish their last unit of job j; a B unit of job j may start at
max(CTA ready, A_done[j] + lat); kernel end = max CTA end + tail. Calibrated so 'today' (the T16 weighted plan, WT1 20 / WT2 30)
reproduces the measured fin max per routing. B2: jobs with > 8 tokens run A then own-dims B on a dedicated group of g CTAs
(no wait; B cost x (1 + ovh) for short-K + y red.add); fillers keep today's weighted plan on G - g CTAs; best g.
"""
import itertools

G, RBA, RBB, WT1, WT2 = 132, 512, 256, 20, 30
ROUT = [([12, 6, 5, 3, 2, 1], 55.8), ([13, 5, 4, 2, 1, 1, 1, 1], 67.1), ([12, 8, 4, 3, 2, 1, 1, 1, 1], 75.5), ([12, 10, 4, 3, 1, 1], 58.1),
        ([13, 9, 3, 2, 1, 1], 58.6), ([12, 7, 5, 4, 3, 2, 1, 1], 66.8), ([10, 6, 3, 2, 1, 1, 1, 1], 67.6), ([13, 4, 3, 2, 2, 1, 1], 60.9),
        ([13, 6, 3, 1], 41.0), ([13, 7, 4, 3, 1], 47.6), ([13, 7, 3, 3, 1, 1], 55.3), ([11, 10, 3, 1, 1, 1], 57.9),
        ([11, 10, 4, 1, 1, 1, 1], 63.5), ([11, 8, 4, 2, 2, 1], 55.0), ([11, 8, 3, 2, 2, 1, 1], 61.7), ([12, 11, 3, 3, 2, 1], 58.6)]


def split(w, per, x):   # the T16 plan's t16_split on a weight list w (per-job unit weight)
    jw = [0]
    for v in w:
        jw.append(jw[-1] + v)
    nj = len(w)
    lo = max(j for j in range(nj + 1) if per * jw[j] <= x)
    if lo >= nj:
        return nj * per
    return lo * per + (x - per * jw[lo]) // w[lo]


def wpre(w, per, v):
    j = v // per
    jw = sum(w[:j])
    return per * jw + (v - j * per) * (w[j] if j < len(w) else 0)


def plan(w, ncta):
    WJ = sum(w)
    out = []
    for c in range(ncta):
        a0, a1 = split(w, RBA, RBA * WJ * c // ncta), split(w, RBA, RBA * WJ * (c + 1) // ncta)
        T0, T1 = (RBA + RBB) * WJ * c // ncta, (RBA + RBB) * WJ * (c + 1) // ncta
        b0, b1 = split(w, RBB, T0 - wpre(w, RBA, a0)), split(w, RBB, T1 - wpre(w, RBA, a1))
        out.append((a0, a1, b0, b1))
    return out


def simulate(jobs2, ncta, c1, c2, t0, lat):   # jobs2: list of bools (2-tile?) per job; returns per-CTA ends
    w = [WT2 if t else WT1 for t in jobs2]
    pl = plan(w, ncta)
    cost = lambda j: c2 if jobs2[j] else c1  # noqa: E731
    adone = [0.0] * len(jobs2)
    tA = []
    for a0, a1, _, _ in pl:
        t = t0
        for v in range(a0, a1):
            j = v // RBA
            t += cost(j)
            adone[j] = max(adone[j], t)
        tA.append(t)
    ends = []
    for (a0, a1, b0, b1), t in zip(pl, tA):
        for v in range(b0, b1):
            j = v // RBB
            t = max(t, adone[j] + lat) + cost(j)
        ends.append(t)
    return ends, tA, adone


def jobs_of(cnt):
    js = []
    for c in cnt:
        while c > 0:
            js.append(c > 8 if c >= 9 else False)
            c -= 16
    return js


def main():
    best = None
    for c1, c2, lat, tail in itertools.product((1.10, 1.12, 1.14), (1.30, 1.39, 1.48), (0.7, 1.0), (3.3,)):
        err = []
        for cnt, fin in ROUT:
            ends, _, _ = simulate(jobs_of(cnt), G, c1, c2, 4.1, lat)
            err.append(max(ends) + tail - fin)
        mae = sum(abs(e) for e in err) / len(err)
        if best is None or mae < best[0]:
            best = (mae, c1, c2, lat, tail, err)
    mae, c1, c2, lat, tail, err = best
    print(f"calibration: c1 {c1} c2 {c2} lat {lat} tail {tail}: MAE {mae:.2f} us; per-routing residuals {[round(e, 1) for e in err]}")
    tot_today, tot_b2 = 0.0, 0.0
    for ovh in (0.0, 0.15, 0.3):
        dsum, rows = 0.0, []
        for cnt, fin in ROUT:
            js = jobs_of(cnt)
            ends, tA, adone = simulate(js, G, c1, c2, 4.1, lat)
            today = max(ends) + tail
            hot = [j for j, t in enumerate(js) if t]
            fil = [t for t in js if not t]
            best_b2 = None
            for g in range(4, 80):
                hot_work = sum((RBA + RBB * (1 + ovh)) * c2 for _ in hot) / g
                hot_end = 4.1 + hot_work
                fe, _, _ = simulate(fil, G - g, c1, c2, 4.1, lat) if fil else ([0.0], None, None)
                e = max(hot_end, max(fe)) + tail
                if best_b2 is None or e < best_b2[0]:
                    best_b2 = (e, g)
            dsum += best_b2[0] - today
            rows.append((cnt[0], round(today, 1), round(best_b2[0], 1), best_b2[1]))
        print(f"B2 own-dims overhead {ovh:.2f}: mean delta {dsum / len(ROUT):+.2f} us over {len(ROUT)} routings; (hot cnt, today, B2, g) {rows[:6]}")
    # where today's wait sits: hot-B CTAs vs filler-B CTAs
    cnt = ROUT[0][0]
    js = jobs_of(cnt)
    ends, tA, adone = simulate(js, G, c1, c2, 4.1, lat)
    print("routing 0 A-done per job (us):", [round(x, 1) for x in adone], "2-tile:", js, "A end p50/max:",
          round(sorted(tA)[len(tA) // 2], 1), round(max(tA), 1), "CTA end max:", round(max(ends), 1))


if __name__ == "__main__":
    main()

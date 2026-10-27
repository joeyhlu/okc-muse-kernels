"""t1_spread_model.py: CPU model of K2A vid 0's K1 / K2 end spread at T <= 8 and of three proposed fixes (parent 2026-10-27):
(a) per-SM dual-ended split, (b) half-size stages, (c) 3 CTAs / SM.

Calibration data (measured, box_out/t1/run.json + run2.json, 24 in-layer marked replays, ship): per-CTA duration from the
p50 first copy (K1) / the hand-off (K2) to the CTA's end (consumer, epilogue and producer exits), min / p50 / max, and stages:
  T=1 K1 8.9 / 11.6 / 17.0 us (7-8 stages), K2 5.8 / 7.2 / 10.3 (3-4); T=8 K1 28.2 / 35.0 / 40.0 (31-32), K2 13.1 / 18.6 / 22.7 (15-16).
The raw per-CTA records were not saved (only these quantiles), so the intra-SM structure comes from other measurements:
  * paired CTAs on one SM: older 0.82-1.02, younger 1.17-1.36 us / 18.4 KB stage (notes/study/moe_bs32_fresh.md, sa5, NC1 cold
    consumer, layer routings) -> asymmetry a = (t_y - t_o) / (t_y + t_o) ~ 0.16;
  * an SM left with one CTA runs at LAMBDA = 0.75-0.8 of the pair's throughput (R-m32-1).
Model per SM: CTA work w (stages, Stream-K equal split), paired per-stage times t (1 -+ a) / f_s, f_s ~ lognormal(sigma) per SM,
a fixed per-CTA cost c0 (ramp + the last-contributor finalize); once a CTA ends its partner runs at the lone rate.
sigma and c0 are fitted to the measured p50 / max; the per-stage time t to the p50. Then each fix is applied to the same SMs.

    python -I tools/t1_spread_model.py
"""
import math
import random
import statistics

MEAS = {  # (stages per CTA (mean), measured min, p50, max of the CTA duration in us)
    ("K1", 1): (2048 / 264, 8.9, 11.6, 17.0), ("K2", 1): (1024 / 264, 5.8, 7.2, 10.3),
    ("K1", 8): (8192 / 264, 28.2, 35.0, 40.0), ("K2", 8): (4096 / 264, 13.1, 18.6, 22.7),
}
A, LAMBDA, NSM = 0.16, 0.78, 132


def stages(total, G):
    return [total * (i + 1) // G - total * i // G for i in range(G)]


def sm_end_pair(w_o, w_y, t_o, t_y):
    """two CTAs from t = 0; the first to finish leaves the other at the lone rate (LAMBDA x the pair throughput)."""
    e_o, e_y = w_o * t_o, w_y * t_y
    pair = 1 / t_o + 1 / t_y
    t_lone = 1 / (LAMBDA * pair)
    if e_o <= e_y:
        done = e_o / t_y
        return e_o, e_o + (w_y - done) * t_lone
    done = e_y / t_o
    return e_y + (w_o - done) * t_lone, e_y


def simulate(w_mean, t, sigma, c0, mode="ship", seed=1, spill=0.0, ovh=0.0):
    rng = random.Random(seed)
    G = 2 * NSM
    total = round(w_mean * G)
    ends = []
    for s in range(NSM):
        f = math.exp(rng.gauss(0.0, sigma))
        to, ty = t * (1 - A) / f, t * (1 + A) / f
        if mode == "ship":
            ws = stages(total, G)
            e1, e2 = sm_end_pair(ws[s], ws[s + NSM], to, ty)
            ends += [e1 + c0, e2 + c0]
        elif mode == "dual":   # (a): both CTAs drain the SM's 2w stages from both ends and finish together (+1 meeting finalize)
            ws = stages(total, G)
            e = (ws[s] + ws[s + NSM]) / (1 / to + 1 / ty)
            ends += [e + c0, e + c0 + 0.5]   # one CTA also finalizes the 2-way meeting item (~0.5 us beyond today's)
        elif mode == "half":   # (b): stages of 9.2 KB, twice as many; per-stage fixed overhead ovh
            ws = stages(2 * total, G)
            th = 0.5 * (1 + ovh)
            e1, e2 = sm_end_pair(ws[s] * th, ws[s + NSM] * th, to, ty)
            ends += [e1 + c0, e2 + c0]
        elif mode == "3cta":   # (c): 3 CTAs / SM; bulk HBM-bound at the same SM throughput; age asymmetry -A / 0 / +A
            ws = stages(total, 3 * NSM)
            pair = 1 / to + 1 / ty                  # SM throughput (stages / us), unchanged while >= 2 CTAs are resident
            ts = [3 * (1 + d) / pair * (1 + spill) for d in (-A, 0.0, A)]   # per-CTA per-stage time with 3 sharing
            left = [ws[s], ws[s + NSM], ws[s + 2 * NSM]]
            now, alive = 0.0, [0, 1, 2]
            while alive:
                n = len(alive)
                rate = {i: (1 / ts[i]) * (3 / n if n == 2 else 1.0) for i in alive}   # 2 left: same SM throughput
                if n == 1:
                    rate = {alive[0]: LAMBDA * pair / (1 + spill)}
                dt, j = min((left[i] / rate[i], i) for i in alive)
                for i in alive:
                    left[i] -= rate[i] * dt
                now += dt
                ends.append(now + c0)
                alive.remove(j)
    ends.sort()
    return ends[0], ends[len(ends) // 2], ends[-1]


def fit(key):
    w, mn, p50, mx = MEAS[key]
    best = None
    for sigma in [x / 50 for x in range(0, 11)]:
        for c0 in [x / 2 for x in range(0, 11)]:
            # per-stage time from the p50: younger-ish CTAs set the p50; solve t by bisection
            lo, hi = 0.3, 3.0
            for _ in range(18):
                t = 0.5 * (lo + hi)
                s = [simulate(w, t, sigma, c0, seed=k)[1] for k in range(1)]
                if statistics.mean(s) > p50:
                    hi = t
                else:
                    lo = t
            r = [simulate(w, t, sigma, c0, seed=k) for k in range(2)]
            sm = [statistics.mean(x[i] for x in r) for i in range(3)]
            err = (sm[0] - mn) ** 2 + (sm[2] - mx) ** 2
            if best is None or err < best[0]:
                best = (err, sigma, c0, t, sm)
    return best


def main():
    print("phase  T | fit: sigma c0 t(us/stage) -> model min/p50/max (measured) | max end: ship, (a) dual, (b) half +2/5 % ovh, "
          "(c) 3 CTAs/SM spill 0/5/10 %")
    tot = {1: [0.0] * 6, 8: [0.0] * 6}
    for key in MEAS:
        err, sigma, c0, t, sm = fit(key)
        w = MEAS[key][0]
        mx = lambda **kw: statistics.mean(simulate(w, t, sigma, c0, seed=k, **kw)[2] for k in range(8))  # noqa: E731
        base = mx(mode="ship")
        arms = [mx(mode="dual"), mx(mode="half", ovh=0.02), mx(mode="half", ovh=0.05), mx(mode="3cta"), mx(mode="3cta", spill=0.05),
                mx(mode="3cta", spill=0.10)]
        for i, x in enumerate(arms):
            tot[key[1]][i] += x - base
        print(f"{key[0]} {key[1]:2d} | {sigma:.3f} {c0:.2f} {t:.3f} -> {sm[0]:.1f}/{sm[1]:.1f}/{sm[2]:.1f} ({MEAS[key][1]}/{MEAS[key][2]}/"
              f"{MEAS[key][3]}) | {base:.2f}, " + ", ".join(f"{x - base:+.2f}" for x in arms))
    for T, v in tot.items():
        print(f"T={T}: K1 + K2 change (us): (a) {v[0]:+.2f}, (b) {v[1]:+.2f} / {v[2]:+.2f}, (c) {v[3]:+.2f} / {v[4]:+.2f} / {v[5]:+.2f}")


if __name__ == "__main__":
    main()

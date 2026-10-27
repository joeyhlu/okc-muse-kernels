"""L3 step-0 (notes/study/beyond_vjeux.md): offline replay of VMS per-CTA raw marks (R-vms-7, box_out/vms7/run.json, arm n8)
under (a) the shipped order (A share then B share, contiguous bands), (b) a two-wave job order (A of wave-1 jobs, A of wave-2
jobs, B of wave 1, B of wave 2; balanced shares per wave), (c) a dynamic pool for each CTA's last K units, (d) the last
units' epilogue on the compute warps (no helper tail). Per CTA the measured per-unit time u_c and start (first unit landed)
are kept; job-switch cost s_sw and the doneA signal latency d_sig are fitted so (a) reproduces the measured B waits / ends.

    python -I tools/vms_l3_replay.py [box_out/vms7/run.json] [--arm=n8] [--sw=0.7] [--pool=2]
"""
import json
import statistics
import sys

G = 132
RBA, RBB = 512, 256
I_ENTRY, I_ROUTE, I_LAND, I_BW0, I_BW1, I_CEND, I_ACRED, I_HEND, I_BAR, I_END, I_N = range(11)


def jobs_of(cnt_by_id, skt=8):
    out = []
    for e, c in enumerate(cnt_by_id):
        k = 0
        while c - k > 0:
            out.append((e, min(skt, c - k)))
            k += skt
    return out


def shares(nj, c, G=G):
    NA, NN = nj * RBA, nj * (RBA + RBB)
    a0, a1 = c * NA // G, (c + 1) * NA // G
    t0, t1 = c * NN // G, (c + 1) * NN // G
    return a0, a1, t0 - a0, t1 - a1


def seq_units(nj, c, waves):
    """the CTA's unit sequence as (phase, job) for a list of waves (each a list of job indices, contiguous)"""
    seq = []
    if len(waves) == 1:
        a0, a1, b0, b1 = shares(nj, c)
        seq += [("A", v // RBA) for v in range(a0, a1)] + [("B", v // RBB) for v in range(b0, b1)]
        return seq
    segA, segB = [], []
    for w in waves:
        n = len(w)
        a0, a1, b0, b1 = shares(n, c)
        segA += [("A", w[0] + v // RBA) for v in range(a0, a1)]
        segB.append([("B", w[0] + v // RBB) for v in range(b0, b1)])
    return segA + [u for s in segB for u in s]


def simulate(nj, start, unit, waves, sw, dsig, pool=0, wsbl_gain=0.0, hepi=0.0):
    """event simulation: CTAs run their sequences; a B unit of job j waits for doneA[j] = last A unit of j done + dsig.
    pool > 0: each CTA's last `pool` units go to a shared pool taken by the earliest-free CTA (greedy)."""
    seqs = [seq_units(nj, c, waves) for c in range(G)]
    pooled = []
    if pool:
        for c in range(G):
            k = min(pool, len(seqs[c]))
            pooled += seqs[c][len(seqs[c]) - k:]
            seqs[c] = seqs[c][:len(seqs[c]) - k]
    # A completion per job needs every CTA's A units of that job: iterate (A units never wait)
    t = [start[c] for c in range(G)]
    last_job = [None] * G
    doneA = {}
    pos = [0] * G
    timeline = [[] for _ in range(G)]
    # pass 1: A units (all A units precede B units in every sequence)
    for c in range(G):
        for ph, j in seqs[c]:
            if ph != "A":
                break
            if last_job[c] != ("A", j):
                t[c] += sw if last_job[c] is not None else 0.0
                last_job[c] = ("A", j)
            t[c] += unit[c]
            doneA[j] = max(doneA.get(j, 0.0), t[c])
            pos[c] += 1
    pooledA = [u for u in pooled if u[0] == "A"]
    if pooledA:   # pooled A units: greedy on the earliest-free CTA
        for ph, j in pooledA:
            c = min(range(G), key=lambda x: t[x])
            t[c] += unit[c] + sw
            doneA[j] = max(doneA.get(j, 0.0), t[c])
    waits = []
    for c in range(G):
        for ph, j in seqs[c][pos[c]:]:
            if last_job[c] != ("B", j):
                ready = doneA.get(j, 0.0) + dsig
                waits.append(max(0.0, ready - t[c]))
                t[c] = max(t[c], ready) + sw
                last_job[c] = ("B", j)
            t[c] += unit[c]
    for ph, j in [u for u in pooled if u[0] == "B"]:
        c = min(range(G), key=lambda x: t[x])
        t[c] = max(t[c], doneA.get(j, 0.0) + dsig) + sw + unit[c]
    end = [t[c] + (hepi - wsbl_gain) for c in range(G)]
    return max(end), statistics.median(waits) if waits else 0.0


def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "box_out/vms7/run.json")
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    arm, sw0, pool = args.get("arm", "n8"), float(args.get("sw", 0.7)), int(args.get("pool", 2))
    recs = json.load(open(path))
    recs = recs["routings"] if isinstance(recs, dict) else recs
    rows_out = []
    for r in recs:
        raw = (r.get("raw") or {}).get(arm)
        if not raw:
            continue
        mk = raw[0]
        t0 = min(x[I_ENTRY] for x in mk if x[I_ENTRY] > 0)
        us = lambda x: (x - t0) / 1e3  # noqa: E731
        jobs = jobs_of(r["cnt_by_id"])
        nj = len(jobs)
        start = [us(x[I_LAND]) for x in mk]
        bw = [us(x[I_BW1]) - us(x[I_BW0]) if x[I_BW0] > 0 and x[I_BW1] > 0 else 0.0 for x in mk]
        unit = [max(0.3, (us(x[I_CEND]) - us(x[I_LAND]) - bw[i]) / max(1, x[I_N] - 1)) for i, x in enumerate(mk)]
        meas_cend = max(us(x[I_CEND]) for x in mk)
        meas_hend = max(us(x[I_HEND]) for x in mk)
        meas_bar = statistics.median(us(x[I_BAR]) for x in mk)
        hepi = meas_hend - meas_cend
        # fit the doneA signal latency so the shipped order reproduces the measured last consumer end
        best = None
        for dsig in [i * 0.1 for i in range(0, 31)]:
            e, wmed = simulate(nj, start, unit, [list(range(nj))], sw0, dsig)
            err = abs(e - meas_cend)
            if best is None or err < best[0]:
                best = (err, dsig, e, wmed)
        _, dsig, e_ship, w_ship = best
        half = max(1, nj // 2)
        waves = [list(range(half)), list(range(half, nj))] if nj >= 2 else [list(range(nj))]
        e_two, w_two = simulate(nj, start, unit, waves, sw0, dsig)
        e_two0, _ = simulate(nj, start, unit, waves, 0.0, dsig)
        e_pool, _ = simulate(nj, start, unit, [list(range(nj))], sw0, dsig, pool=pool)
        e_both, _ = simulate(nj, start, unit, waves, sw0, dsig, pool=pool)
        rec = {"T": r["T"], "layer": r["layer"], "rs": r["rs"] % 100000, "jobs": nj, "meas_cend": round(meas_cend, 2),
               "model_ship": round(e_ship, 2), "dsig": round(dsig, 1), "bw_meas_med": round(statistics.median(bw), 2),
               "bw_model_med": round(w_ship, 2), "two_wave": round(e_two - e_ship, 2), "two_wave_free_switch": round(e_two0 - e_ship, 2),
               "pool": round(e_pool - e_ship, 2), "two_wave+pool": round(e_both - e_ship, 2),
               "helper_tail": round(hepi, 2), "bar_after_hend": round(meas_bar - meas_hend, 2)}
        rows_out.append(rec)
        print(" ".join(f"{k}={v}" for k, v in rec.items()))
    print("SUMMARY (median per T; d = model change of the last consumer end vs the shipped order, us)")
    for T in sorted({x["T"] for x in rows_out}):
        rr = [x for x in rows_out if x["T"] == T]
        med = lambda k: round(statistics.median(x[k] for x in rr), 2)  # noqa: E731
        print(f"  T={T:2d} n={len(rr)} model err {med('model_ship') - med('meas_cend'):+.2f} | two-wave {med('two_wave'):+.2f} "
              f"(free switches {med('two_wave_free_switch'):+.2f}) | pool {med('pool'):+.2f} | both {med('two_wave+pool'):+.2f} | "
              f"helper tail {med('helper_tail'):.2f} (WSBL removes most of it) | barrier after helper end {med('bar_after_hend'):.2f}")


if __name__ == "__main__":
    main()

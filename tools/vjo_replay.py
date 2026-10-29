"""VJO step 0 ($0): VMS job-order lever replayed on measured per-CTA marks (R-vms-7 box_out/vms7/run.json, arm n8 =
globaltimer marks, no clock64 in the loop). Arms (per CTA static unit order; unit set per CTA = ship unless 'bal'):

  ship      ship shares (contiguous A slice then contiguous B slice), A then B
  lit_i     ship shares, per-job order A(j0) B(j0) A(j1) B(j1) ...            (brief arm i, literal)
  lit_ii    ship shares, A(first half of jobs) A(second half) B(first) B(second) (brief arm ii, literal)
  bal_ii    per-wave shares over all G CTAs (2 waves = job halves), A1 A2 B1 B2  (= R-vms-7 'two-wave')
  bal_iie   bal_ii with the ship's exact per-CTA A and B counts (wave 2 = remainder; no quantization cost)
  bal_iix   per-wave shares, A1 B1 A2 B2                                         (wonder_new_levers.md idea 2)
  bal_i     per-job shares over all G CTAs, A(j0) B(j0) A(j1) B(j1) ...          (ext_muse_vms.md #1, needs new shares)
  bal_ilag  per-job shares, A(j0) A(j1) B(j0) A(j2) B(j1) ... B(jn-1)            (lag-1 software pipeline)
  oracle    ship shares, every doneA wait removed for free (R-slc-0 ceiling: max_c(end_c - wait_c))

Model: per CTA the measured first-landed time and unit pace u_c = (cend - land - bw - sw * switches_ship) / (N - 1) are
kept, so the ship arm reproduces each CTA up to the doneA model; a B segment of job j starts at max(t, doneA[j] + dsig),
doneA[j] = the last A unit of j on any CTA. dsig is fitted per routing so the ship model reproduces the measured last
consumer end. Every segment switch (phase or job change) costs sw (x / inter reload; the B switch's relaxed poll + acquire
is in dsig). Deadlock = a B unit whose job's A units cannot all complete (reported, never seen for job-sorted orders).

    python -I tools/vjo_replay.py [box_out/vms7/run.json] [--arm=n8] [--sw=0,0.35,0.7]
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


def shares(nj, c):
    NA, NN = nj * RBA, nj * (RBA + RBB)
    a0, a1 = c * NA // G, (c + 1) * NA // G
    t0, t1 = c * NN // G, (c + 1) * NN // G
    return a0, a1, t0 - a0, t1 - a1


def ship_units(nj, c, off=0):
    a0, a1, b0, b1 = shares(nj, c)
    return [("A", off + v // RBA) for v in range(a0, a1)], [("B", off + v // RBB) for v in range(b0, b1)]


def seqs_for(arm, nj):
    out = []
    half = max(1, nj // 2)
    for c in range(G):
        if arm in ("ship", "oracle"):
            A, B = ship_units(nj, c)
            out.append(A + B)
        elif arm == "lit_i":
            A, B = ship_units(nj, c)
            out.append([u for j in range(nj) for u in [x for x in A if x[1] == j] + [x for x in B if x[1] == j]])
        elif arm == "lit_ii":
            A, B = ship_units(nj, c)
            lo = lambda L, f: [x for x in L if f(x[1])]  # noqa: E731
            out.append(lo(A, lambda j: j < half) + lo(A, lambda j: j >= half) + lo(B, lambda j: j < half) + lo(B, lambda j: j >= half))
        elif arm in ("bal_ii", "bal_iix"):
            waves = [(0, half), (half, nj)] if nj >= 2 else [(0, nj)]
            segs = [ship_units(w1 - w0, c, w0) for w0, w1 in waves]
            if arm == "bal_ii":
                out.append([u for A, _ in segs for u in A] + [u for _, B in segs for u in B])
            else:
                out.append([u for A, B in segs for u in A + B])
        elif arm == "bal_iie":   # A1 A2 B1 B2 with the ship's exact per-CTA A / B counts (wave 2 takes the remainder)
            if nj < 2:
                A, B = ship_units(nj, c)
                out.append(A + B)
                continue
            NA1, NB1, NA, NB = half * RBA, half * RBB, nj * RBA, nj * RBB
            S = lambda n, k: k * n // G  # noqa: E731
            sA = lambda k: S(NA, k)  # noqa: E731
            sB = lambda k: S(nj * (RBA + RBB), k) - S(NA, k)  # noqa: E731
            A1 = range(S(NA1, c), S(NA1, c + 1))
            A2 = range(NA1 + sA(c) - S(NA1, c), NA1 + sA(c + 1) - S(NA1, c + 1))
            B1 = range(S(NB1, c), S(NB1, c + 1))
            B2 = range(NB1 + sB(c) - S(NB1, c), NB1 + sB(c + 1) - S(NB1, c + 1))
            out.append([("A", v // RBA) for v in list(A1) + list(A2)] + [("B", v // RBB) for v in list(B1) + list(B2)])
        elif arm in ("bal_i", "bal_ilag"):
            segs = [ship_units(1, c, j) for j in range(nj)]
            if arm == "bal_i":
                out.append([u for A, B in segs for u in A + B])
            else:
                s = list(segs[0][0])
                for j in range(1, nj):
                    s += segs[j][0] + segs[j - 1][1]
                s += segs[nj - 1][1]
                out.append(s)
        else:
            raise ValueError(arm)
    return out


def switches(seq):
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def simulate(seqs, start, unit, sw, dsig, nowait=False):
    """returns (max consumer end, per-CTA ends, deadlock flag)"""
    nA = {}
    for s in seqs:
        for ph, j in s:
            if ph == "A":
                nA[j] = nA.get(j, 0) + 1
    doneA, cntA = {}, {}
    t = list(start)
    pos = [0] * G
    prev = [None] * G
    progress = True
    while progress:
        progress = False
        for c in range(G):
            s = seqs[c]
            while pos[c] < len(s):
                ph, j = s[pos[c]]
                if ph == "B" and not nowait and cntA.get(j, 0) < nA.get(j, 0):
                    break
                if prev[c] is None or s[pos[c]] != prev[c]:   # new segment (phase or job change)
                    if prev[c] is not None:
                        t[c] += sw
                    if ph == "B" and not nowait:
                        t[c] = max(t[c], doneA.get(j, 0.0) + dsig)
                prev[c] = s[pos[c]]
                t[c] += unit[c]
                if ph == "A":
                    cntA[j] = cntA.get(j, 0) + 1
                    doneA[j] = max(doneA.get(j, 0.0), t[c])
                pos[c] += 1
                progress = True
    dead = any(pos[c] < len(seqs[c]) for c in range(G))
    return max(t), t, dead


def main():
    path = next((a for a in sys.argv[1:] if not a.startswith("--")), "box_out/vms7/run.json")
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    arm_src = args.get("arm", "n8")
    sws = [float(x) for x in args.get("sw", "0,0.35,0.7").split(",")]
    recs = json.load(open(path))
    recs = recs["routings"] if isinstance(recs, dict) else recs
    arms = ["lit_i", "lit_ii", "bal_ii", "bal_iie", "bal_iix", "bal_i", "bal_ilag"]
    rows = []
    for r in recs:
        raw = (r.get("raw") or {}).get(arm_src)
        if not raw:
            continue
        mk = raw[0]
        t0 = min(x[I_ENTRY] for x in mk if x[I_ENTRY] > 0)
        us = lambda x: (x - t0) / 1e3  # noqa: E731
        nj = len(jobs_of(r["cnt_by_id"]))
        start = [us(x[I_LAND]) for x in mk]
        bw = [us(x[I_BW1]) - us(x[I_BW0]) if x[I_BW0] > 0 and x[I_BW1] > 0 else 0.0 for x in mk]
        cend = [us(x[I_CEND]) for x in mk]
        meas = max(cend)
        ship = seqs_for("ship", nj)
        assert all(len(ship[c]) == mk[c][I_N] for c in range(G)), "ship shares != marked N"
        lit_ii = seqs_for("lit_ii", nj)
        same_ii = sum(1 for c in range(G) if lit_ii[c] == ship[c])
        lit_i = seqs_for("lit_i", nj)
        diff_i = sum(1 for c in range(G) if lit_i[c] != ship[c])
        oracle = meas - max(c_ - w for c_, w in zip(cend, bw))
        rec = {"T": r["T"], "layer": r["layer"], "rs": r["rs"] % 1000, "nj": nj, "meas": round(meas, 2),
               "lit_ii_same_ctas": same_ii, "lit_i_changed_ctas": diff_i, "oracle": round(-oracle, 2)}
        for sw in sws:
            nsw = [switches(s) for s in ship]
            unit = [max(0.3, (cend[c] - start[c] - bw[c] - sw * nsw[c]) / max(1, len(ship[c]) - 1)) for c in range(G)]
            best = None
            for k in range(0, 61):
                dsig = 0.05 * k
                e, _, _ = simulate(ship, start, unit, sw, dsig)
                if best is None or abs(e - meas) < best[0]:
                    best = (abs(e - meas), dsig, e)
            err, dsig, e_ship = best
            rec[f"sw{sw}_dsig"] = round(dsig, 2)
            rec[f"sw{sw}_err"] = round(e_ship - meas, 2)
            for a in arms:
                sq = seqs_for(a, nj)
                e, _, dead = simulate(sq, start, unit, sw, dsig)
                rec[f"sw{sw}_{a}"] = None if dead else round(e - e_ship, 2)
                if a.startswith("bal"):
                    rec[f"maxN_{a}"] = max(len(s) for s in sq) - max(len(s) for s in ship)
                    rec[f"xsw_{a}"] = round(statistics.mean(switches(s) for s in sq) - statistics.mean(nsw), 2)
        rows.append(rec)
        print(" ".join(f"{k}={v}" for k, v in rec.items()))
    print("\nSUMMARY: median over routings of (arm max consumer end - ship model), us; negative = faster")
    for T in sorted({x["T"] for x in rows}):
        rr = [x for x in rows if x["T"] == T]
        med = lambda k: statistics.median(x[k] for x in rr if x[k] is not None)  # noqa: E731
        print(f" T={T:2d} n={len(rr)} oracle(all waits free) {med('oracle'):+.2f} [{min(x['oracle'] for x in rr):+.2f}"
              f"..{max(x['oracle'] for x in rr):+.2f}]  lit_ii identical CTAs {min(x['lit_ii_same_ctas'] for x in rr)}/{G}"
              f"  lit_i changed CTAs {statistics.median(x['lit_i_changed_ctas'] for x in rr)}")
        for a in arms:
            extra = "" if not a.startswith("bal") else (f"  (+switches/CTA {med('xsw_' + a):+.2f}, max N "
                                                          f"{med('maxN_' + a):+.0f})")
            print(f"   {a:9s} " + "  ".join(f"sw{sw}: {med(f'sw{sw}_{a}'):+.2f}" for sw in sws) + extra)
        print("   fit: " + "  ".join(f"sw{sw} dsig {med(f'sw{sw}_dsig'):.2f} err {med(f'sw{sw}_err'):+.2f}" for sw in sws))


if __name__ == "__main__":
    main()

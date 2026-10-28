"""m32e step 1 ($0, local): where VML's time above the leader model goes at T = 32 / 128 / 256 (layer routings).

  python -I tools/m32e_breakdown.py [box_out/vml12/cta.json.gz:f193s box_out/vml14/cta.json.gz:f193s8 ...] [--json=PATH]

Inputs: per-CTA marks dumps of tools/box/vml_run4.py --ctadump (lab v9 / v11 columns: 0 start, 7 route done, 5 consumers'
first stage (gather / self-fill done), 6 phase-A end (WG 0), 1-4 WG ends, 8-15 / 16-23 A / B stages by n-tiles, 24 B doneA
wait ns, 26 A xready wait ns) plus the run.json next to each dump (graph-replay us per arm, routing counts).
Kernel time per routing = replay us of the arm (vm_main + vm_fin, replay overhead removed), split exactly into
  skew   mean CTA start offset               route  mean(route done - start)       gather mean(first stage - route done)
  work   mean(CTA end - first stage)         spread max CTA end - mean CTA end     tail   replay us - max CTA end (vm_fin + launch)
and work into  stream1 = (all stages at the nt = 1 cost) / G,  hot = (stage cost - nt1 cost) / G,  resid = the rest
(waits, epilogues, fixups), with stage costs from a pooled per-T regression of CTA work on its per-nt stage counts (and the
R-vml-14 probe F table as a cross-check). stream1 = unique (distinct experts x 768 stages) + reread (extra jobs of experts
split at 64 tokens). Leader model M = distinct x 28.31 MB / 3.6 TB/s + 5 us (fresh_view_oct28 s3).
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path

G = 132
STAGES_PER_JOB = 768                    # 512 W13 + 256 W2 stages of 256 rows x 256 K (36,864 B)
MODEL_US_PER_EXPERT = 28.31e6 / 3.6e12 * 1e6
# R-vml-14 probe arm F (1 CTA / SM, all SMs streaming) us per stage by N (interpolated at odd n-tiles); bit 7 mma.sync at
# nt <= 2 (R-vml-8 ratios: 1.124 / 1.312 us)
PROBE_F = {1: 1.124, 2: 1.312, 3: 1.50, 4: 1.626, 5: 1.706, 6: 1.786, 7: 1.916, 8: 2.046}


def jobs_of(counts):
    out = []
    for tot in counts:
        if tot <= 0:
            continue
        nj = (tot + 63) // 64
        base = 64
        while base > 1 and (base - 1) * nj >= tot:
            base -= 1
        for k in range(nj):
            sz = min(base, tot - k * base)
            out.append((sz, (sz + 7) // 8))
    return out


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)


def lstsq(X, y):
    """least squares with intercept column already in X (normal equations, Gauss-Jordan)."""
    m = len(X[0])
    ata = [[sum(r[a] * r[b] for r in X) for b in range(m)] for a in range(m)]
    aty = [sum(r[a] * v for r, v in zip(X, y)) for a in range(m)]
    keep = [a for a in range(m) if ata[a][a] > 1e-9]
    A = [[ata[a][b] for b in keep] + [aty[a]] for a in keep]
    n = len(keep)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        A[col] = [v / pv for v in A[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [a - f * b for a, b in zip(A[r], A[col])]
    beta = [0.0] * m
    for i, a in enumerate(keep):
        beta[a] = A[i][n]
    return beta


def ctas(rec):
    reps = rec["reps"]
    n = len(reps)
    rows = []
    for i in range(len(reps[0])):
        r = [sum(rp[i][c] for rp in reps) / n for c in range(32)]
        rows.append({"st": r[0] / 1e3, "rd": r[7] / 1e3, "gd": r[5] / 1e3, "ae": r[6] / 1e3, "en": max(r[1:5]) / 1e3,
                     "sA": r[8:16], "sB": r[16:24], "wB": r[24] / 1e3, "wA": r[26] / 1e3})
    return rows


def main(argv):
    specs = [a for a in argv if not a.startswith("--")] or ["box_out/vml12/cta.json.gz:f193s", "box_out/vml14/cta.json.gz:f193s8"]
    outp = next((a.split("=", 1)[1] for a in argv if a.startswith("--json=")), "")
    recs = []
    for sp in specs:
        path, arm = sp.rsplit(":", 1)
        run = load(str(Path(path).with_name("run.json")))
        us = {(r["rs"], r["T"], str(r["seed"])): r for r in run["routings"]}
        for rec in load(path):
            if rec["arm"] != arm:
                continue
            rr = us[(rec["rs"], rec["T"], str(rec["seed"]))]
            recs.append({"src": path, "arm": arm, "T": rec["T"], "rs": rec["rs"], "seed": rec["seed"], "counts": rr["counts"],
                         "us": rr["us"][arm], "rows": ctas(rec)})
    res = {}
    for T in sorted({r["T"] for r in recs}, reverse=True):
        rr = [r for r in recs if r["T"] == T]
        # pooled regression: CTA work (end - first stage) on per-nt stage counts (A and B pooled), routing intercepts
        X, y = [], []
        for k, r in enumerate(rr):
            for c in r["rows"]:
                n = [a + b for a, b in zip(c["sA"], c["sB"])]
                if sum(n) == 0:
                    continue
                X.append([1.0 if j == k else 0.0 for j in range(len(rr))] + n)
                y.append(c["en"] - c["gd"])
        beta = lstsq(X, y)
        icpt = beta[:len(rr)]
        fitr = {nt + 1: beta[len(rr) + nt] for nt in range(8)}
        # the per-CTA regression is collinear with the cost-weighted split at T = 32 / 256 (every CTA holds ~equal cost):
        # attribute with the measured probe table, report the regression only as a diagnostic
        fitc = dict(PROBE_F) if "--fit" not in argv else fitr
        f1 = fitc[1]
        lines = []
        agg = {}
        for k, r in enumerate(rr):
            rows = [c for c in r["rows"]]
            jobs = jobs_of(r["counts"])
            nj, nd = len(jobs), len([c for c in r["counts"] if c > 0])
            ends = [c["en"] for c in rows]
            emax, emean = max(ends), statistics.mean(ends)
            comp = {"skew": statistics.mean(c["st"] for c in rows), "route": statistics.mean(c["rd"] - c["st"] for c in rows),
                    "gather": statistics.mean(c["gd"] - c["rd"] for c in rows), "work": statistics.mean(c["en"] - c["gd"] for c in rows),
                    "spread": emax - emean, "tail": r["us"] - emax}
            nst = [sum((c["sA"][t] + c["sB"][t]) for c in rows) for t in range(8)]
            assert sum(nst) == STAGES_PER_JOB * nj, (sum(nst), nj)
            comp["stream1"] = sum(nst) * f1 / G
            comp["hot"] = sum(nst[t] * (fitc[t + 1] - f1) for t in range(8) if nst[t]) / G
            comp["resid"] = comp["work"] - comp["stream1"] - comp["hot"]
            comp["unique"] = STAGES_PER_JOB * nd * f1 / G
            comp["reread"] = STAGES_PER_JOB * (nj - nd) * f1 / G
            comp["probe_hot"] = sum(nst[t] * (PROBE_F[t + 1] - PROBE_F[1]) for t in range(8)) / G
            comp["probe_stream1"] = sum(nst) * PROBE_F[1] / G
            model = nd * MODEL_US_PER_EXPERT + 5.0
            comp["model"] = model
            comp["total"] = r["us"]
            comp["excess"] = r["us"] - model
            # wait marks (diagnostic, inside work): B doneA wait and A xready wait per CTA
            comp["wB"] = statistics.mean(c["wB"] for c in rows)
            comp["wA"] = statistics.mean(c["wA"] for c in rows)
            comp["nj"], comp["nd"] = nj, nd
            comp["merge_save"] = 0.0
            # what one weight pass per expert would save on the probe slope (hot expert as one job, N = tokens, F(N) linear in N)
            for tot in r["counts"]:
                if tot > 64:
                    parts = [j for j in jobs_of([tot])]
                    split_cost = sum(PROBE_F[nt] for _, nt in parts)
                    npad = ((tot + 7) // 8) * 8
                    merged = PROBE_F[8] + (npad - 64) * (PROBE_F[8] - PROBE_F[4]) / 32.0
                    comp["merge_save"] += STAGES_PER_JOB * (split_cost - merged) / G
            for kk, v in comp.items():
                agg.setdefault(kk, []).append(v)
            lines.append(f"  rs={r['rs']:>5} in={r['seed']} n={nd} jobs={nj} top={r['counts'][:4]} total {r['us']:.1f} model {model:.1f} "
                         f"excess {r['us'] - model:+.1f} | skew {comp['skew']:.1f} route {comp['route']:.1f} gather {comp['gather']:.1f} "
                         f"work {comp['work']:.1f} (stream1 {comp['stream1']:.1f} = uniq {comp['unique']:.1f} + reread {comp['reread']:.1f}; "
                         f"hot {comp['hot']:.1f}; resid {comp['resid']:+.1f}) spread {comp['spread']:.1f} tail {comp['tail']:.1f} | "
                         f"wB {comp['wB']:.1f} wA {comp['wA']:.1f} merge-save {comp['merge_save']:.1f}")
        med = {kk: statistics.median(v) for kk, v in agg.items()}
        print(f"T={T}: {len(rr)} routings; regression stage us by nt " + " ".join(f"{t}:{v:.3f}" for t, v in fitr.items())
              + f" | routing intercepts median {statistics.median(icpt):.1f} us")
        for ln in lines:
            print(ln)
        print(f"  MEDIAN total {med['total']:.1f} model {med['model']:.1f} excess {med['excess']:+.1f} | prologue (skew+route+gather) "
              f"{med['skew'] + med['route'] + med['gather']:.1f} [skew {med['skew']:.1f} route {med['route']:.1f} gather {med['gather']:.1f}] | "
              f"stream1 {med['stream1']:.1f} (uniq {med['unique']:.1f} vs model-5 {med['model'] - 5:.1f} -> rate {med['unique'] - med['model'] + 5:+.1f}; "
              f"reread {med['reread']:.1f}) | hot {med['hot']:.1f} (probe {med['probe_hot']:.1f}) | resid {med['resid']:+.1f} | spread {med['spread']:.1f} | "
              f"tail {med['tail']:.1f} | merge-save {med['merge_save']:.1f}")
        res[T] = {"fit": fitc, "median": med, "per": agg}
    if outp:
        Path(outp).write_text(json.dumps(res, indent=1))
        print(f"[m32e] wrote {outp}")


if __name__ == "__main__":
    main(sys.argv[1:])

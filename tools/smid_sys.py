# Systematic per-SM effect: split-half (even/odd launches) covariance of per-SM mean residuals.
import gzip, json, statistics as S, sys, collections, random
def sysfx(launches, label):
    # launches: list of dict smid -> residual (us)
    ev, od = collections.defaultdict(list), collections.defaultdict(list)
    for i, L in enumerate(launches):
        g = L.pop("_g", i)
        L.pop("_g", None)
        for sm, y in L.items(): (ev if g % 2 == 0 else od)[sm].append(y)
    sms = [s for s in ev if s in od]
    a = [S.mean(ev[s]) for s in sms]; b = [S.mean(od[s]) for s in sms]
    ma, mb = S.mean(a), S.mean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(sms) - 1)
    r = cov / (S.pstdev(a) * S.pstdev(b) + 1e-12)
    allm = sorted((S.mean(ev[s] + od[s]), s) for s in sms)
    sd = max(cov, 0) ** 0.5
    rawsd = S.pstdev([y for L in launches for y in L.values()])
    spread = S.median([max(L.values()) - S.median(L.values()) for L in launches])
    print(f"{label}: groups {len(set(L.get('_g',0) for L in launches))} n_launch {len(launches)} SMs {len(sms)} | split-half r {r:+.2f} | systematic SD {sd:.3f} us (raw SD {rawsd:.2f}) "
          f"| est. slowest-SM systematic excess ~{2.5*sd:.2f} us | max-p50 spread median {spread:.2f} | slowest SMs {[s for _, s in allm[-4:]]}")
recs = json.load(gzip.open("box_out/vml14/cta.json.gz", "rt"))
for T in (32, 128, 256):
    for col, nm in ((6, "A end"), (None, "kernel end")):
        Ls = []; G = {}
        for ri, r in enumerate([q for q in recs if q["T"] == T]):
            for rep in r["reps"]:
                v = [(row[6] if col else max(row[1:5])) / 1e3 for row in rep]; m = S.median(v)
                d = {int(row[29]): (row[6] if col else max(row[1:5])) / 1e3 - m for row in rep}; key = tuple(int(row[29]) for row in rep); G.setdefault(key, len(G)); d["_g"] = G[key]; Ls.append(d)
        sysfx(Ls, f"VML T={T} {nm}")
for k, ci, nm in (("k1", 10, "K2A-v0 K1 epi_exit"), ("k2", 10, "K2A-v0 K2 epi_exit"), ("k1", 9, "K2A-v0 K1 cons_end")):
    Ls = []
    for rs in ("12345", "777", "8677534258059892491"):
        for r in json.load(open(f"box_out/t1/raw_{rs}.json")):
            rows = [x for x in r[k] if x[ci] is not None]
            # per SM: max over its CTAs (2 slots); residual vs median; control for stages via per-launch per-stages median
            bys = collections.defaultdict(list)
            for x in rows: bys[x[2]].append(x[ci])
            med = {s: S.median(v) for s, v in bys.items()}
            per = collections.defaultdict(list)
            for x in rows: per[int(x[0])].append(x[ci] - med[x[2]])
            Ls.append({s: max(v) for s, v in per.items()})
    sysfx(Ls, nm)

"""slice_plan.py: $0 CPU replay of the VML static plan for L5 SLICE-READY (notes/study/lead_oct29.md s3; R-slc-0).

    python tools/slice_plan.py [--G=132] [--sig=0.025] [--draws=20]

Per routing (layer routings from box_out/m32e*/vml*/sk run.json, deduped; counts are sorted, so 3 expert orders are
replayed: sorted desc, asc, and one fixed shuffle): jobs (VM_JN 64 at T=32 bf2, 72 at T=128/256 w72, tile-aligned split),
cost-cut A / B ranges (VML_COST of k3_gc_19), A order with the split_point rotation (flags bit 0), a sequential per-CTA
timing model (A / B us per stage by nt; T=32 tier paces from the R-v32 marks, T>=128 cost-proportional x 1.19 us),
0.4 us per split share (half at A end, half at B end), split units ready after both shares + the resolve.
Arms (A and B work per CTA identical in every arm, no extra segments or split shares):
  base  identity: slice s = A units 2s, 2s+1; B in positional order
  pair  the brief's closed form: A position p -> logical unit p/2 (p even) or 16 + p/2 (p odd); B inside each segment
        visits its K-slices in modelled-readiness order (an oracle B order inside a segment)
  orac  oracle A map (positions sorted by modelled finish, slice k = the 2k-th and (2k+1)-th) + the same B order
  zero  every slice ready at t = 0 (upper bound of any readiness lever)
Gate (pre-registered, notes/memory/gpu_requests.md R-slc-0): GO if the pair or orac arm moves the median (over routings)
modelled max-CTA end >= 2 us earlier at T=32 and >= 3 us at T=128 / 256.
"""
import glob
import json
import random
import sys

VML_COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
            128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
            32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}
JN = {32: 64, 128: 72, 256: 72}
TA32 = [1.19, 1.60, 1.70, 1.81, 1.95, 2.05, 2.15, 2.25]
TB32 = [1.20, 1.36, 1.52, 1.69, 1.85, 2.0, 2.1, 2.2]
SPLIT_US, XLAT = 0.4, 0.6


def cost_ext(c, ntm):
    out = []
    for pb in (0, 1):
        h = c[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out += h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)]
    return out


def routings():
    seen, out = set(), {}
    for f in sorted(glob.glob("box_out/*/run.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in (d.get("routings") or []) if isinstance(d, dict) else []:
            T, c = r.get("T", r.get("bs")), r.get("counts")
            if T not in JN or not isinstance(c, list) or not all(isinstance(x, int) for x in c):
                continue
            key = (T, tuple(c))
            if key not in seen:
                seen.add(key)
                out.setdefault(T, []).append([x for x in c if x > 0])
    return out


def jobs_of(counts, T):
    js = []
    for n in counts:
        nj = -(-n // JN[T])
        base = -(-(-(-n // 8)) // nj) * 8
        for k in range(nj):
            sz = min(base, n - k * base)
            if sz > 0:
                js.append(-(-sz // 8))
    return js


def ranges(nts, G, ns, cost, off):
    cs, acc = [], 0
    for nt in nts:
        cs.append(acc)
        acc += ns * cost[off + nt - 1]
    b = []
    for c in range(G + 1):
        if c >= G:
            b.append(len(nts) * ns)
            continue
        x = acc * c // G
        lo = max(i for i in range(len(nts)) if cs[i] <= x)
        st = min(ns - 1, (x - cs[lo]) // cost[off + nts[lo] - 1])
        b.append(lo * ns + st)
    return b


def order_a(g0, g1):
    ue = ((g1 - 1) >> 4) << 4
    if g1 & 15 and ue > g0:
        return list(range(ue, g1)) + list(range(g0, ue))
    return list(range(g0, g1))


def nshares(g0, g1):
    if g1 <= g0:
        return 0
    return sum(1 for u in range(g0 >> 4, ((g1 - 1) >> 4) + 1) if min(g1, 16 * u + 16) - max(g0, 16 * u) < 16)


def sim(nts, T, G, arm, jit):
    cost = cost_ext(VML_COST[T], max(8, max(nts)))
    ntm = len(cost) // 2
    if T == 32:
        ta, tb = TA32, TB32
    else:
        ta = [1.19 * cost[k] / 128 for k in range(ntm)]
        tb = [1.19 * cost[ntm + k] / 128 for k in range(ntm)]
    bA, bB = ranges(nts, G, 512, cost, 0), ranges(nts, G, 256, cost, ntm)
    sh = [nshares(bA[c], bA[c + 1]) + nshares(bB[c], bB[c + 1]) for c in range(G)]
    # phase A: per CTA sequential; unit (j, p) finish = max over sharing CTAs (+ resolve if shared)
    aend, ufin, ucnt = [0.0] * G, {}, {}
    for c in range(G):
        t = 0.0
        for gs in order_a(bA[c], bA[c + 1]):
            j, p = gs >> 9, (gs >> 4) & 31
            t += ta[nts[j] - 1] * jit[c]
            ufin[(j, p)] = max(ufin.get((j, p), 0.0), t)
        for u in range(bA[c] >> 4, ((bA[c + 1] - 1) >> 4) + 1) if bA[c + 1] > bA[c] else ():
            ucnt[(u >> 5, u & 31)] = ucnt.get((u >> 5, u & 31), 0) + 1
        aend[c] = t + SPLIT_US * sh[c] / 2
    for k, n in ucnt.items():
        if n > 1:
            ufin[k] += SPLIT_US
    ready = {}
    for j in range(len(nts)):
        f = [ufin[(j, p)] for p in range(32)]
        if arm == "base":
            m = list(range(32))                                   # logical unit of position p
        elif arm == "pair":
            m = [p // 2 if p % 2 == 0 else 16 + p // 2 for p in range(32)]
        else:
            srt = sorted(range(32), key=lambda p: f[p])
            m = [0] * 32
            for k, p in enumerate(srt):
                m[p] = k
        lf = [0.0] * 32
        for p in range(32):
            lf[m[p]] = f[p]
        for s in range(16):
            ready[(j, s)] = 0.0 if arm == "zero" else max(lf[2 * s], lf[2 * s + 1])
    bend, wait = [0.0] * G, [0.0] * G
    for c in range(G):
        t = aend[c]
        g0, g1 = bB[c], bB[c + 1]
        segs, gs = [], g0
        while gs < g1:
            e = min(g1, ((gs >> 4) + 1) << 4)
            segs.append((gs, e))
            gs = e
        for a, b in segs:
            j = a >> 8
            ks = [g & 15 for g in range(a, b)]
            if arm != "base":
                ks.sort(key=lambda s: ready[(j, s)])
            for s in ks:
                r = ready[(j, s)] + XLAT
                if r > t:
                    wait[c] += r - t
                    t = r
                t += tb[nts[j] - 1] * jit[c]
        bend[c] = t + SPLIT_US * sh[c] / 2
    act = [c for c in range(G) if bA[c + 1] > bA[c] or bB[c + 1] > bB[c]]
    return max(bend), sorted(bend[c] for c in act)[len(act) // 2], sum(wait) / len(act), max(wait)


def main():
    G = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--G=")), 132))
    sig = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--sig=")), 0.025))
    nd = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--draws=")), 20))
    rs = routings()
    arms = ("base", "pair", "orac", "zero")
    verdict = {}
    for T in (32, 128, 256):
        rows = []
        for counts in rs.get(T, []):
            for oi, order in enumerate((sorted(counts, reverse=True), sorted(counts), random.Random(7).sample(counts, len(counts)))):
                nts = jobs_of(order, T)
                rng = random.Random(1000 * T + oi)
                acc = {a: [0.0, 0.0, 0.0, 0.0] for a in arms}
                for _ in range(nd):
                    jit = [max(0.8, rng.gauss(1.0, sig)) for _ in range(G)]
                    for a in arms:
                        r = sim(nts, T, G, a, jit)
                        for k in range(4):
                            acc[a][k] += r[k] / nd
                rows.append(acc)
        if not rows:
            continue
        med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
        print(f"T={T}: {len(rows) // 3} routings x 3 expert orders, {nd} jitter draws (sigma {sig})")
        for a in arms:
            d = [r[a][0] - r["base"][0] for r in rows]
            print(f"  {a:4s} bmax {med([r[a][0] for r in rows]):6.1f}  dbmax median {med(d):+5.2f} min {min(d):+5.2f} | "
                  f"bp50 {med([r[a][1] for r in rows]):6.1f} | B wait/CTA {med([r[a][2] for r in rows]):4.2f} max {med([r[a][3] for r in rows]):4.2f}")
        verdict[T] = min(med([r[a][0] - r["base"][0] for r in rows]) for a in ("pair", "orac"))
    ok = verdict.get(32, 0) <= -2.0 and verdict.get(128, 0) <= -3.0 and verdict.get(256, 0) <= -3.0
    print("gate:", {T: round(v, 2) for T, v in verdict.items()}, "->", "GO" if ok else "KILL")


if __name__ == "__main__":
    main()

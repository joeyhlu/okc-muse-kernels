"""skw_plan.py: $0 CPU replay of the VML static plan at bucket 32 (VM_RT, VM_JN 64, one job per expert), base vs SKEW
(notes/study/fresh_eyes_oct29c.md mechanism 1: A(J1) A(J2) B(J1) B(J2), snake-split halves by nt, J1 nt-desc / J2 nt-asc).

    python tools/skw_plan.py [--G=132]

Per routing (T=32 counts from box_out/*/run.json, deduped): split shares per CTA, a sequential per-CTA timing model
(A / B us per stage by nt from the R-v32 marks, h slice s of job j ready when its 2 A units are done, 0.4 us per split share)
-> modelled B end max, doneA idle, and the SKEW gate: min_c B(J1) length >= max(A(J2) end spread, 5.4 measured) + 2 us.
"""
import glob
import json
import sys

COST_A = [128, 167, 183, 229, 171, 182, 193, 204]
COST_B = [128, 162, 184, 215, 171, 182, 193, 204]
TA = [1.19, 1.60, 1.70, 1.81, 1.95, 2.05]      # us / A stage by nt (tier paces, R-v32 marks)
TB = [1.20, 1.36, 1.52, 1.69, 1.85, 2.0]       # us / B stage by nt (in-layer fit 128/145/162/180 x 1.2)
SPLIT_US = 0.4


def routings():
    seen, out = set(), []
    for f in sorted(glob.glob("box_out/*/run.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in (d.get("routings") or []) if isinstance(d, dict) else []:
            if r.get("T", r.get("bs")) == 32 and "counts" in r:
                c = tuple(int(x) for x in r["counts"] if int(x) > 0)
                if c not in seen:
                    seen.add(c)
                    out.append(list(c))
    return out


def ranges(jobs, G, ns, cost):
    """jobs = [(nt)], one half: cumulative cost cut at c / G -> per CTA [g0, g1) in the half's local stage space."""
    cs, acc = [], 0
    for nt in jobs:
        cs.append(acc)
        acc += ns * cost[nt - 1]
    b = []
    for c in range(G + 1):
        if c >= G or acc == 0:
            b.append(len(jobs) * ns)
            continue
        x = acc * c // G
        lo = max(i for i in range(len(jobs)) if cs[i] <= x)
        st = min(ns - 1, (x - cs[lo]) // cost[jobs[lo] - 1])
        b.append(lo * ns + st)
    return b


def order_a(g0, g1):
    ue = ((g1 - 1) >> 4) << 4
    if g1 & 15 and ue > g0:
        return list(range(ue, g1)) + list(range(g0, ue))
    return list(range(g0, g1))


def split_shares(g0, g1):
    if g1 <= g0:
        return 0
    return sum(1 for u in range(g0 >> 4, ((g1 - 1) >> 4) + 1) if min(g1, 16 * u + 16) - max(g0, 16 * u) < 16)


def sim(counts, G, skew, jit=None):
    nts = [(n + 7) // 8 for n in counts]
    if skew:   # snake split of the nt-desc rank: positions 0,3,4,7,.. -> J1 (nt desc), 1,2,5,6,.. -> J2 (nt asc)
        rk = sorted(range(len(nts)), key=lambda i: (-nts[i], i))
        h1 = [rk[p] for p in range(len(rk)) if p % 4 in (0, 3)]
        h2 = [rk[p] for p in range(len(rk)) if p % 4 in (1, 2)][::-1]
        halves = [h1, h2]
    else:
        halves = [list(range(len(nts)))]
    jobs = [e for h in halves for e in h]   # global job order
    off, j0 = [], 0
    for h in halves:
        off.append(j0)
        j0 += len(h)
    seqs = [[] for _ in range(G)]   # per CTA: list of (phase, job, unit, stage)
    shares = [0] * G
    for pb, ns, cost in ((0, 512, COST_A), (1, 256, COST_B)):
        for hi, h in enumerate(halves):
            b = ranges([nts[e] for e in h], G, ns, cost)
            for c in range(G):
                g0, g1 = b[c], b[c + 1]
                shares[c] += split_shares(g0, g1)
                for gs in (order_a(g0, g1) if pb == 0 else range(g0, g1)):
                    seqs[c].append((pb, off[hi] + gs // ns, (gs % ns) >> 4, gs & 15))
    # A pass: per CTA sequential, record unit completion (A unit = 16 stages; h slice s = units 2s, 2s+1)
    t = [0.0] * G
    udone = {}
    aend = [0.0] * G
    for c in range(G):
        prev = None
        for pb, j, u, s in seqs[c]:
            if pb:
                break
            if prev is not None and prev != (j, u):
                pass
            t[c] += TA[nts[jobs[j]] - 1] * (jit[c] if jit else 1.0)
            udone[(j, u)] = max(udone.get((j, u), 0.0), t[c])
            prev = (j, u)
        t[c] += SPLIT_US * shares[c] / 2
        aend[c] = t[c]
    ready = {(j, s): max(udone.get((j, 2 * s), 0.0), udone.get((j, 2 * s + 1), 0.0)) for j in range(len(jobs)) for s in range(16)}
    bend, idle, b1len = [0.0] * G, [0.0] * G, [0.0] * G
    nB1 = len(halves[0]) if skew else len(jobs)
    for c in range(G):
        tc, b1s = aend[c], None
        for pb, j, u, s in seqs[c]:
            if not pb:
                continue
            r = ready[(j, s)]
            if r > tc:
                idle[c] += r - tc
                tc = r
            if j < nB1 and b1s is None:
                b1s = tc
            tc += TB[nts[jobs[j]] - 1] * (jit[c] if jit else 1.0)
            if j < nB1:
                b1len[c] = tc - b1s
        bend[c] = tc + SPLIT_US * shares[c] / 2
    act = [c for c in range(G) if seqs[c]]
    # A(J2) end spread (skew): per CTA time its last A2 stage ends
    return dict(bmax=max(bend), bp50=sorted(bend[c] for c in act)[len(act) // 2], idle=sum(idle) / max(1, len(act)),
                idlemax=max(idle), sh=sum(shares) / max(1, len(act)), shmax=max(shares),
                spread=max(aend[c] for c in act) - min(aend[c] for c in act),
                b1min=min((b1len[c] for c in act if b1len[c] > 0), default=0.0), nj=len(jobs))


def main():
    G = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--G=")), 132))
    rs = routings()
    import random
    rng = random.Random(1)
    SIG = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--sig=")), 0.025))   # per-CTA pace jitter (end sigma ~1.3 us)
    ok, rows = 0, []
    for counts in rs:
        b, k = sim(counts, G, False), sim(counts, G, True)
        if SIG > 0:   # Monte Carlo: same per-CTA jitter draw for base and skew, 20 draws, mean of bmax / idle
            acc = {"b": [0.0, 0.0], "k": [0.0, 0.0]}
            for _ in range(20):
                jit = [max(0.8, rng.gauss(1.0, SIG)) for _ in range(G)]
                for key, sk in (("b", False), ("k", True)):
                    r = sim(counts, G, sk, jit)
                    acc[key][0] += r["bmax"] / 20
                    acc[key][1] += r["idle"] / 20
            b["bmax"], b["idle"], k["bmax"], k["idle"] = acc["b"][0], acc["b"][1], acc["k"][0], acc["k"][1]
        gate = k["b1min"] >= max(k["spread"], 5.4) + 2.0
        ok += gate
        rows.append((b, k, gate))
    print(f"{len(rs)} routings at T=32, G={G}; columns base -> skew")
    print("nj  bmax          dB    idle/CTA     share/CTA(max)   A-spread  B1min  gate")
    for b, k, g in rows[:14]:
        print(f"{b['nj']:2d}  {b['bmax']:5.1f}->{k['bmax']:5.1f} {k['bmax'] - b['bmax']:+5.1f}  {b['idle']:4.1f}->{k['idle']:4.1f}  "
              f"{b['sh']:.2f}({b['shmax']})->{k['sh']:.2f}({k['shmax']})  {k['spread']:5.1f}  {k['b1min']:5.1f}  {'ok' if g else 'NO'}")
    d = sorted(k["bmax"] - b["bmax"] for b, k, _ in rows)
    print(f"gate pass {ok}/{len(rs)} = {100 * ok / max(1, len(rs)):.0f} %; dbmax median {d[len(d) // 2]:+.2f} us, "
          f"mean {sum(d) / len(d):+.2f}; extra shares/CTA {sum(k['sh'] - b['sh'] for b, k, _ in rows) / len(rows):+.2f}")


if __name__ == "__main__":
    main()

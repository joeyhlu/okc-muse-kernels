"""vml32_deal.py: $0 CPU price of Muse's ext_muse_moe_field.md s4.2 "split-free whole-unit deal with expert-home lagged B" at T=32
(follow-up to tools/vml32_align.py; same calibrated bucket-32 timeline as tools/plan_search.py, same 60 board seeds + 54 layer profiles).

Proposal as modelled
  units   A unit = (expert, 256 rows, full K) = 16 stages, 32 per job; B unit = (expert, 256 output rows, full intermediate K)
          = 16 stages, 16 per job. No K split anywhere => zero split shares, zero resolvers, zero count atomics.
  list    expert-interleaved, id order (or nt-descending): A(e0) A(e1) [B(e0)] A(e2) [B(e1)] ..., B(e) enters L experts after A(e)
          (L 2 or 3); B(e) units go back to the CTAs that held A(e) ("home"); B waits only on that expert's doneA (counter gate: all
          32 A units of e, + HOP + XLAT; no global A barrier). Stage-level gate (ship's 2s / 2s+1 flags) is an extra arm.
  deal    static, computed in route() from the cost tables: greedy = each unit to the CTA with the least ESTIMATED load (stages x
          table pace + piece end), no knowledge of waits; rr = literal cyclic deal. Execution then runs on the true pace with waits.
  tail    the last Q list entries are cut into disjoint-row pieces of 128 or 64 rows (full K, so no shares / resolver: a piece costs
          rows/256 x 16 stages + a normal piece end, EPI + PF for A, BEND for B). Extra arm: pieces cost eff x their bytes.
Piece costs are the plan_search measured sub-steps (EPI 0.59, PF 0.68, BEND 0.62; 0.33 cheaper first stage of every later piece;
HOP 0.5 + XLAT 0.6); START 8.0 + 0.5 (the dealer in route() is not free); phi = 0 at T=32 (additive, as calibrated).
Metric: max CTA end vs the shipped static plan, simulated by plan_search.simulate on the same routing.
  python -I tools/vml32_deal.py [--n=60]
"""
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_search as ps  # noqa: E402
import vml32_align as va  # noqa: E402

G = ps.G
T = 32


def build_list(nts, L):
    n, out = len(nts), []
    for k in range(n + L):
        if k < n:
            out += [("A", k, u) for u in range(32)]
        if k - L >= 0:
            out += [("B", k - L, b) for b in range(16)]
    return out


def cut_tail(lst, Q, rows):
    """last Q entries -> disjoint-row pieces of `rows` rows each"""
    if Q <= 0 or rows >= 256:
        return [(k, j, i, 256) for k, j, i in lst]
    cut = max(0, len(lst) - Q)
    out = [(k, j, i, 256) for k, j, i in lst[:cut]]
    for k, j, i in lst[cut:]:
        out += [(k, j, i, rows)] * (256 // rows)
    return out


def paces_for(nts, P, a):
    ntm = max(8, max(nts))
    b8 = ps.MB_LAYER_COST[32]
    tabA, tabB = ps.cost_ext(list(b8[:8]) + list(b8[8:]), ntm)
    return ps.paces(T, P, a, ntm, tabA, tabB)


def run(counts, P, a, C0, arm):
    """one routing, one arm -> dict(end, ...). arm keys: L, order, dealer rr|greedy, home, gate counter|stage, Q, rows, eff, pro, err"""
    nts = ps.jobs_of(counts, 64, True, arm.get("order", "id"))
    if not nts:
        return None
    S = ps.SUB["ps"]
    pa_t, pb_t = paces_for(nts, dict(P, err=None), a)               # the table the dealer sees
    pa, pb = paces_for(nts, dict(P, err=arm.get("err")), a)          # the truth (pace error variant)
    f = 1.0                                                          # phi 0 at T=32
    lst = cut_tail(build_list(nts, arm.get("L", 2)), arm.get("Q", 0), arm.get("rows", 256))
    eff = arm.get("eff", 1.0)
    pe = arm.get("pe", 1.0)   # piece-end work scale for disjoint-row pieces (r < 256); 1.0 = a normal piece end
    prop = arm.get("prop", False)   # row-proportional end: A = EPI x rows/256 + PF (SwiGLU + stores scale, the release fence does not), B = BEND x (0.5 + 0.5 rows/256)

    def endA(r):
        if r >= 256:
            return S["EPI"] + S["PF"]
        return S["EPI"] * r / 256 + S["PF"] if prop else (S["EPI"] + S["PF"]) * pe

    def endB(r):
        if r >= 256:
            return S["BEND"]
        return S["BEND"] * (0.5 + 0.5 * r / 256) if prop else S["BEND"] * pe
    START = P["START"] + arm.get("pro", 0.5)
    FS, HOP, XLAT = P["FS"], P["HOP"], P["XLAT"]
    # ---- deal (estimated loads: table pace + piece end, no waits)
    load = [0.0] * G
    holders, assign, rr = {}, [], 0
    if arm.get("dealer") == "informed":   # clairvoyant bound: every piece to the CTA that finishes it first (true pace, waits, home)
        tt = [START] * G
        fst = [True] * G
        dj = {}
        for k, j, i, r in lst:
            nt = nts[j]
            e = eff if r < 256 else 1.0
            if k == "A":
                cand = range(G)
                dur = 16 * pa[nt - 1] * r / 256 * e + endA(r)
                c = min(cand, key=lambda x: tt[x] + dur - (0.0 if fst[x] else FS))
                tt[c] += dur - (0.0 if fst[c] else FS)
                dj[j] = max(dj.get(j, 0.0), tt[c])
                holders.setdefault(j, [])
                if c not in holders[j]:
                    holders[j].append(c)
            else:
                cand = holders[j] if arm.get("home", True) else range(G)
                dur = 16 * pb[nt - 1] * r / 256 * e + endB(r)
                rdy = dj[j] + HOP + XLAT
                c = min(cand, key=lambda x: max(tt[x], rdy) + dur - (0.0 if fst[x] else FS))
                tt[c] = max(tt[c], rdy) + dur - (0.0 if fst[c] else FS)
            fst[c] = False
            assign.append(c)
    for k, j, i, r in (lst if arm.get("dealer") != "informed" else []):
        nt = nts[j]
        if k == "A":
            est = 16 * pa_t[nt - 1] * r / 256 + S["EPI"] + S["PF"]
            if arm.get("dealer", "greedy") == "rr":
                c = rr % G
                rr += 1
            else:
                c = min(range(G), key=lambda x: load[x])
            holders.setdefault(j, [])
            if c not in holders[j]:
                holders[j].append(c)
        else:
            est = 16 * pb_t[nt - 1] * r / 256 + S["BEND"]
            cand = holders[j] if arm.get("home", True) else range(G)
            if arm.get("dealer", "greedy") == "rr":
                c = holders[j][i % len(holders[j])]
            else:
                c = min(cand, key=lambda x: load[x])
        load[c] += est
        assign.append(c)
    # ---- execute (true pace, waits)
    t = [START] * G
    first = [True] * G
    wait = [0.0] * G
    nA, nB = [0] * G, [0] * G
    doneU, doneJ = {}, {}
    for (k, j, i, r), c in zip(lst, assign):
        nt = nts[j]
        e = eff if r < 256 else 1.0
        fs = 0.0 if first[c] else FS
        if k == "A":
            t[c] += 16 * pa[nt - 1] * r / 256 * e - fs + endA(r)
            doneU[(j, i)] = max(doneU.get((j, i), 0.0), t[c])
            doneJ[j] = max(doneJ.get(j, 0.0), t[c])
            nA[c] += 1
        else:
            if arm.get("gate", "counter") == "counter":
                rdy = doneJ[j] + HOP + XLAT
                if rdy > t[c]:
                    wait[c] += rdy - t[c]
                    t[c] = rdy
                t[c] += 16 * pb[nt - 1] * r / 256 * e - fs + endB(r)
            else:
                t[c] -= fs
                for s in range(16):
                    rdy = max(doneU[(j, 2 * s)], doneU[(j, 2 * s + 1)]) + HOP + XLAT
                    if rdy > t[c]:
                        wait[c] += rdy - t[c]
                        t[c] = rdy
                    t[c] += pb[nt - 1] * r / 256 * e
                t[c] += endB(r)
            nB[c] += 1
        first[c] = False
    act = [c for c in range(G) if nA[c] + nB[c]]
    end = max(t) + C0
    ovh = sum((nA[c] * (S["EPI"] + S["PF"]) + nB[c] * S["BEND"] - FS * (nA[c] + nB[c] - 1)) for c in act) / len(act)   # piece-end work net of FS credit
    floor = START + sum(16 * pa[nts[j] - 1] * r / 256 if k == "A" else 16 * pb[nts[j] - 1] * r / 256 for k, j, i, r in lst) / G + C0
    return dict(end=end, A=sum(nA[c] for c in act) / len(act), B=sum(nB[c] for c in act) / len(act), nact=len(act),
                wait=sum(wait[c] for c in act) / len(act), wmax=max(wait), bal=max(t) - st.mean(t[c] for c in act),
                last=max(t) - sorted(t[c] for c in act)[len(act) // 2], pieces=len(lst), floor=floor, ovh=ovh)


def ship_info(counts, P, a, C0):
    d = ps.simulate({}, counts, T, P, a, C0, detail=True)
    nts = ps.jobs_of(counts, 64, True)
    b8 = ps.MB_LAYER_COST[32]
    _, tabB = ps.cost_ext(list(b8[:8]) + list(b8[8:]), max(8, max(nts)))
    bB = ps.cut(nts, 256, tabB, G)
    pB = [ps.pieces(bB[c], bB[c + 1], False, 8) for c in range(G)]
    act = [c for c in range(G) if pB[c]]
    d["B"] = sum(len(pB[c]) for c in act) / len(act)
    return d


ARMS = {
    "rr L2 (literal)": dict(dealer="rr", L=2),
    "deal L2": dict(L=2),
    "deal L3": dict(L=3),
    "deal L2 lpt": dict(L=2, order="nt_desc"),
    "deal L2 unhomed": dict(L=2, home=False),
    "deal L2 stage-gate": dict(L=2, gate="stage"),
}
for _Q in (66, 132, 264, 400):
    for _r in (128, 64):
        ARMS[f"deal L2 tail{_Q} r{_r}"] = dict(L=2, Q=_Q, rows=_r)
for _r in (128, 64):
    ARMS[f"deal L2 tailALL r{_r}"] = dict(L=2, Q=10 ** 6, rows=_r)
for _L in (2, 3, 4):
    ARMS[f"informed L{_L}"] = dict(dealer="informed", L=_L)
    for _Q in (132, 264):
        for _r in (128, 64):
            ARMS[f"informed L{_L} tail{_Q} r{_r}"] = dict(dealer="informed", L=_L, Q=_Q, rows=_r)
ARMS["informed L3 unhomed"] = dict(dealer="informed", L=3, home=False)
ARMS["informed L3 stage-gate"] = dict(dealer="informed", L=3, gate="stage")
ARMS["informed L3 tail132 r64 stage-gate"] = dict(dealer="informed", L=3, Q=132, rows=64, gate="stage")
ARMS["informed L3 tail132 r64 eff1.15"] = dict(dealer="informed", L=3, Q=132, rows=64, eff=1.15)
ARMS["informed L3 tail264 r64 prop-end"] = dict(dealer="informed", L=3, Q=264, rows=64, prop=True)
ARMS["informed L3 tail264 r128 prop-end"] = dict(dealer="informed", L=3, Q=264, rows=128, prop=True)
ARMS["informed L3 tail132 r64 prop-end"] = dict(dealer="informed", L=3, Q=132, rows=64, prop=True)
ARMS["deal L3 tail264 r64 prop-end"] = dict(L=3, Q=264, rows=64, prop=True)
ARMS["deal L3 tail132 r64 prop-end"] = dict(L=3, Q=132, rows=64, prop=True)
for _e in (1.15, 1.30):
    ARMS[f"informed L3 tail264 r64 prop-end eff{_e}"] = dict(dealer="informed", L=3, Q=264, rows=64, prop=True, eff=_e)
    ARMS[f"deal L3 tail264 r64 prop-end eff{_e}"] = dict(L=3, Q=264, rows=64, prop=True, eff=_e)
ARMS["informed L4 tail264 r64 prop-end"] = dict(dealer="informed", L=4, Q=264, rows=64, prop=True)
ARMS["deal L4 tail264 r64 prop-end"] = dict(L=4, Q=264, rows=64, prop=True)
ARMS["informed L3 tail264 r64 prop-end pro+1.5"] = dict(dealer="informed", L=3, Q=264, rows=64, prop=True, pro=2.0)
ARMS["informed L3 tail264 r64 prop-end stage-gate"] = dict(dealer="informed", L=3, Q=264, rows=64, prop=True, gate="stage")
ARMS["deal L3 tail264 r64 prop-end err5% s1"] = dict(L=3, Q=264, rows=64, prop=True, err=(0.05, 1))
ARMS["deal L3 tail264 r64 prop-end err5% s2"] = dict(L=3, Q=264, rows=64, prop=True, err=(0.05, 2))
for _pe in (0.5, 0.25, 0.0):
    ARMS[f"informed L3 tail264 r64 piece-end x{_pe}"] = dict(dealer="informed", L=3, Q=264, rows=64, pe=_pe)
ARMS["informed L3 tail132 r64 pro+1.5"] = dict(dealer="informed", L=3, Q=132, rows=64, pro=2.0)
ARMS["deal L3 tail132 r64"] = dict(L=3, Q=132, rows=64)
ARMS["fav: lpt+unhomed+stage L2 tail132 r64"] = dict(L=2, order="nt_desc", home=False, gate="stage", Q=132, rows=64)


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(f * len(v)))]


def evaluate(name, rs, P, a, C0, arms, ship):
    base = [s["end"] for s in ship]
    print(f"\n== {name}: {len(rs)} routings; ship end median {st.median(base):.1f}, ship segs/CTA A {st.mean(s['segA'] for s in ship):.2f}"
          f" B {st.mean(s['B'] for s in ship):.2f}, shares/CTA {st.mean(s['split'] for s in ship):.2f}, B wait/CTA {st.mean(s['wait'] for s in ship):.2f},"
          f" ship max-mean end {st.mean(s['bal'] for s in ship):.1f}, ship piece-end work net of FS {st.mean(s['ovm'] - 0.33 * (s['segA'] - 1) for s in ship):.2f} us/CTA")
    print(f"{'arm':40s} {'d end med':>9s} {'p10':>6s} {'p90':>6s} {'mean':>6s} | {'shares':>6s} {'A seg':>6s} {'B seg':>6s} {'wait/CTA':>8s} {'wait max':>8s} {'max-mean':>8s} {'ovh/CTA':>7s}")
    res = {}
    for nm, arm in arms.items():
        rr = [run(r, P, a, C0, arm) for r in rs]
        pd = [x["end"] - b for x, b in zip(rr, base)]
        res[nm] = pd
        d = sorted(pd)
        print(f"{nm:40s} {st.median(d):+9.1f} {q(d, .1):+6.1f} {q(d, .9):+6.1f} {st.mean(d):+6.1f} | {0.0:6.2f} {st.mean(x['A'] for x in rr):6.2f} "
              f"{st.mean(x['B'] for x in rr):6.2f} {st.mean(x['wait'] for x in rr):8.2f} {st.mean(x['wmax'] for x in rr):8.1f} {st.mean(x['bal'] for x in rr):8.1f} {st.mean(x['ovh'] for x in rr):7.2f}")
    return res


def main():
    n = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--n=")), 60))
    P = dict(ps.DEF)
    a, C0, rms = va.calib32(P)
    print(f"calibration bucket 32: pace a {a:.2f}, C0 {C0:.1f} us, rms {rms:.2f}")
    sets = {"board bs32": va.board60()[:n], "layer T=32 (54 saved)": ps.layer(32)}
    syn, za, _ = va.syn_layer(120)
    sets["layer synthetic 8.33"] = syn
    for nm, rs in sets.items():
        ship = [ship_info(r, P, a, C0) for r in rs]
        fl = [run(r, P, a, C0, dict(L=3))["floor"] - sh["end"] for r, sh in zip(rs, ship)]
        print(f"[{nm}] fluid floor (stages / 132 + start + C0, zero overhead) vs ship: median {st.median(fl):+.1f} us [{q(fl, .1):+.1f}, {q(fl, .9):+.1f}]")
        res = evaluate(nm, rs, P, a, C0, ARMS, ship)
        # oracle: per routing best of no-tail and every tail arm (paired; selection on model noise = unreachable bound)
        for lab, cands in (("greedy deals", [k for k in res if k.startswith("deal L") and "prop" not in k and "err" not in k]),
                           ("greedy deals prop-end", [k for k in res if k.startswith("deal L") and "prop" in k and "eff" not in k and "err" not in k]),
                           ("informed prop-end", [k for k in res if k.startswith("informed L") and "prop" in k and "eff" not in k and "pro+" not in k]),
                           ("informed deals", [k for k in res if k.startswith("informed L") and "prop" not in k and "piece-end" not in k and "eff" not in k and "pro+" not in k and "gate" not in k and "unhomed" not in k])):
            orc = sorted(min(res[k][i] for k in cands) for i in range(len(rs)))
            print(f"{'oracle best of ' + lab + ' (paired)':40s} {st.median(orc):+9.1f} {q(orc, .1):+6.1f} {q(orc, .9):+6.1f} {st.mean(orc):+6.1f}")


if __name__ == "__main__":
    main()

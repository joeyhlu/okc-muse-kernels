"""tq_model.py: $0 CPU gate R-tq-0 for a TICKET QUEUE (whole-unit work claiming) in the VML MoE at T >= 32 (lab R-tq-0).

    python tools/tq_model.py [--nd=3] [--seed=11] [--marks=box_out/lat32/m_ship/marks_raw.json.gz] [--only=substr]     (~40 s, prints 20 rows)
    python tools/tq_model.py --sdscale=0|0.5 | --iid=1      (noise sensitivities: no persistent per-CTA pace noise / TQ noise per ticket)
    python tools/tq_model.py --persist=1                     (is the ship's end tail CTA-intrinsic? cross-routing residual correlation, 3 rows)

Ship: static plan (cost tables), 132 CTAs, K-split segments balanced up front.  TQ: no cost tables, no K-splits.  The queue is
  A units (expert, 256-row block, full K = 16 stages) first (jobs in LPT order = descending token count, 32 A units per job), then the
  B units (16 per job, same job order).  Each CTA's first unit is its blockIdx (static, free); every later unit is claimed from a global
  counter (one dependent L2 atomic, cost L, claims are served in issue-time order).
Machinery = tools/e4_model.py (marks-fitted generative ship model, noise draws shared with the ship = CRN) + tools/iwg_model.py (T = 32 setup).
Charges (all of the lead's list):
  1. claim latency L (2.5 / 4.5): ahead 0 = claim issued when the unit's last stage ends (exposed; the epilogue overlaps it), a0h = claimed
     at stage 8 of the running unit (8 stages of commitment), ahead 1 / 2 = ticket for unit i + k claimed at the START of unit i (hidden,
     but the first k+1 rounds are then effectively static).  A CTA that finds the counter empty still waits for that last claim.
  2. B gating: B unit of job j starts no earlier than (latest doneA release over ALL 32 A units of j) + L + XL (one dependent round trip +
     x landing); idle wait is simulated (a CTA holding a not-ready B ticket waits, it cannot skip).  doneA release = ship's DEFER rule
     (next unit's start + kf x fa + phiF), A -> B transition = aend rule (t4 + phiA, B starts at aend + g0).  gate=stage: ship's per-stage
     gating (B stage m needs A units 2m, 2m+1) for sensitivity.
  3. cold first stage: data for a newly claimed unit lands at max(claim result, ring-limited prefetch start) + C0, C0 = max(2.5, XL(T)).
     The ring prefetch starts 2 stages before the previous unit ends, so a ticket known early hides it, a ticket claimed at the end does not.
  4. tail quantization is simulated as is (whole 16-stage units); tail variant: the last M queue units are cut into 4 pieces of 4 stages
     (partial end work each, the last piece to be claimed resolves: waits for all 4 arrivals + 4 round trips of L).
Null/validation: V0 = zero charges (L = 0, C0 = 0, stage gating, expert order, claim at free) must land near the R-iwg-0 'ship + dyn unit
queue' control (-9.3 / -20.6 / -25.0 us); only deltas vs ship on the same noise draws count (marks ran ~10 % slow).
"""
import argparse
import heapq
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iwg_model as iwg  # noqa: E402  (sets e4.COST[32] and e4.SIG[32])
import e4_model as e4  # noqa: E402
import ae_model as ae  # noqa: E402

ROOT, G = ae.ROOT, ae.G
NA, NB = 32, 16                     # units per job: A 512 stages / 16, B 256 / 16


class Cfg:
    """one arm. ahead: 0 = claim near the end of the running unit (lead stages before it ends), k >= 1 = ticket i + k claimed at the start of unit i.
    L: claim / release round trip; ss: TQ stage cost scale (ship unchanged); gate: job | stage; tail: last M queue units cut in 4 pieces."""

    def __init__(s, name, ahead=0, lead=0.0, L=3.5, ss=1.0, C0=None, lrel=None, gate="job", order="lpt", tail=0, chain=4.0, claim_at="te", c0min=2.5, ring=2.0, lag=None, iid=False):
        s.__dict__.update(locals())
        del s.__dict__["s"]


def build_queue(jobs, cfg, n=G):
    """ticket list [(type, job, unit, piece, queue index)]: A units first then B (lag None), or B(j) placed lag x n tickets after A(j)'s last ticket."""
    idx = list(range(len(jobs)))
    if cfg.order == "lpt":
        idx.sort(key=lambda j: (-jobs[j].N, j))
    q = [(0, j, u, 0) for j in idx for u in range(NA)]
    bq = [(1, j, u, 0) for j in idx for u in range(NB)]
    if cfg.lag is None:
        q = q + bq
    else:
        endA = {j: max(i for i, t in enumerate(q) if t[1] == j) + 1 for j in idx}
        keyed = [(i, 0, t) for i, t in enumerate(q)] + [(endA[t[1]] + cfg.lag * n, k, t) for k, t in enumerate(bq)]
        keyed.sort(key=lambda z: (z[0], z[1]))
        q = [z[2] for z in keyed]
    if cfg.tail:
        q = q[:-cfg.tail] + [(2, j, u, k) for (_t, j, u, _p) in q[-cfg.tail:] for k in range(4)]
    return [t + (i,) for i, t in enumerate(q)]


def tq_run(jobs, p, ctas, eps, cfg, um=None):
    """one run: ({c: end}, stats).  Event-driven: claims are served in issue-time order; every claim schedules its unit on the claiming CTA."""
    ea, eb = eps
    act = sorted(ctas)
    n = len(act)
    ss, L = cfg.ss, cfg.L
    lrel = cfg.lrel if cfg.lrel is not None else L
    C0 = cfg.C0 if cfg.C0 is not None else max(cfg.c0min, p.XL)
    g, g0, phiA, phiF, kf, tau, XL, d = p.g, p.g0, p.phiA, p.phiF, p.kf, p.tau, p.XL, p.d
    eAf = [p.eAf("full", jb.N) for jb in jobs]
    eBf = [p.eBf("full", jb.N) for jb in jobs]
    eBp = [p.eBf("part", jb.N) for jb in jobs]
    Q = build_queue(jobs, cfg, n)
    nQ = len(Q)
    if um is not None:
        um = um(Q)
    k_ahead = cfg.ahead
    ptr = 0
    # per job
    reluns = [NA] * len(jobs)
    relmax = [0.0] * len(jobs)
    relR = [[0.0] * NA for _ in jobs]
    blockedj = {}
    arr = {}
    # per CTA
    fc = [ctas[c]["fc"] for c in act]
    K = [0] * n
    nsch = [0] * n
    t4p = [0.0] * n
    tep = [0.0] * n
    pacep = [1.0] * n
    prevtyp = [-1] * n
    lastA = [None] * n          # (j, u, t4, free_prev) of the last scheduled A unit whose successor is not known yet
    pend = [deque() for _ in range(n)]
    blocked = [False] * n
    done = [False] * n
    noneav = [None] * n
    heap = []
    seq = [0]
    st = dict(wgate=0.0, wclaim=0.0, nclaim=0)

    def push(t, i):
        seq[0] += 1
        heapq.heappush(heap, (t, i, seq[0]))

    def resolve(i, s_next, fa_next):
        la = lastA[i]
        if la is None:
            return
        j, u, t4, free_prev = la
        if s_next is None:
            rel = t4 + phiA
        else:
            rel = min(s_next, free_prev) + kf * fa_next + phiF
        relR[j][u] = rel
        relmax[j] = max(relmax[j], rel)
        reluns[j] -= 1
        lastA[i] = None
        if reluns[j] == 0:
            for c2 in blockedj.pop(j, []):
                blocked[c2] = False
                advance(c2)

    def sched(i, tk, avail):
        typ, j, u, pc, qi = tk
        jb = jobs[j]
        e = (ea[act[i]] if typ == 0 else eb[act[i]]) if um is None else um[qi]
        pa, fa, pb = jb.pA * ss, jb.fA * ss, jb.pB * ss
        first = nsch[i] == 0
        if first:
            free = fc[i]
            s = free
        else:
            free = t4p[i] + ((phiA + g0) if (prevtyp[i] == 0 and typ != 0) else g)
            pref = max(avail, tep[i] - cfg.ring * pacep[i])
            s = max(free, pref + C0)
            st["wclaim"] += s - free
        if typ == 0:
            fcost = pa if first else fa
            te = max(s + (fcost + 15 * pa) * e, fc[i] + (K[i] + 15) * tau)
            K[i] += 16
            t4 = te + d + eAf[j]
            resolve(i, s, fa)
            lastA[i] = (j, u, t4, t4 + g)
            pace = pa * e
        else:
            resolve(i, None, 0.0)
            nst, base = (16, 0) if typ == 1 else (4, 4 * pc)
            rw = relmax[j] + lrel + XL
            t = s
            for m in range(nst):
                r = rw if cfg.gate == "job" else max(relR[j][2 * (base + m)], relR[j][2 * (base + m) + 1]) + lrel + XL
                K[i] += 1
                stt = max(t, r, fc[i] + (K[i] - 1) * tau)
                if m == 0 and prevtyp[i] == 0:
                    stt = max(stt, t4p[i] + phiA + p.xb0)          # ship: first B stage no earlier than aend + XL
                st["wgate"] += max(0.0, r - t)
                t = stt + pb * e
            te = t
            t4 = te + d + (eBf[j] if typ == 1 else eBp[j])
            if typ == 2:
                lst = arr.setdefault((j, u), [])
                lst.append(t4)
                if len(lst) == 4:
                    t4 = max(lst) + cfg.chain * L
            pace = pb * e
        nsch[i] += 1
        prevtyp[i], t4p[i], tep[i], pacep[i] = typ, t4, te, pace
        if done[i]:
            return
        if k_ahead == 0:
            push((t4 + g) if cfg.claim_at == "free" else max(s, te - cfg.lead * pace), i)
        else:
            for _ in range(k_ahead if nsch[i] == 1 else 1):
                push(s, i)

    def advance(i):
        while pend[i] and not blocked[i]:
            tk, avail = pend[i][0]
            typ, j = tk[0], tk[1]
            if typ != 0:
                resolve(i, None, 0.0)
                if reluns[j] > 0:
                    blocked[i] = True
                    blockedj.setdefault(j, []).append(i)
                    return
            pend[i].popleft()
            sched(i, tk, avail)

    def claim(t, i):
        nonlocal ptr
        avail = t + L
        st["nclaim"] += 1
        if ptr < nQ:
            tk = Q[ptr]
            ptr += 1
            if not done[i]:
                pend[i].append((tk, avail))
                advance(i)
        else:
            if not done[i]:
                done[i] = True
                noneav[i] = avail
                if not blocked[i]:
                    resolve(i, None, 0.0)

    for i in range(n):
        tk = Q[i]
        ptr = i + 1
        pend[i].append((tk, fc[i]))
        advance(i)
    while heap:
        t, i, _s = heapq.heappop(heap)
        claim(t, i)
    if ptr < nQ or any(b for b in blocked):
        raise RuntimeError("tq deadlock: ptr %d/%d blocked %d" % (ptr, nQ, sum(blocked)))
    end = {act[i]: max(t4p[i] + g, noneav[i] if noneav[i] is not None else 0.0) for i in range(n)}
    assert sum(nsch) == nQ, (sum(nsch), nQ)
    st["nunit"] = nQ
    return end, st


def arms():
    A = [Cfg("V0 zero charges (L0 C0 0, stage gate, expert order)", L=0.0, C0=0.0, lrel=0.0, gate="stage", order="expert", claim_at="free")]
    A.append(Cfg("a0  exposed (claim at unit end) L2.5", ahead=0, L=2.5))
    A.append(Cfg("a0  exposed L4.5", ahead=0, L=4.5))
    A.append(Cfg("a0h claim at stage 8 of the unit L2.5", ahead=0, lead=8.0, L=2.5))
    A.append(Cfg("a0h claim at stage 8 L4.5", ahead=0, lead=8.0, L=4.5))
    A.append(Cfg("a1  1 unit ahead L4.5", ahead=1, L=4.5))
    A.append(Cfg("a2  2 units ahead L2.5", ahead=2, L=2.5))
    A.append(Cfg("a2  2 units ahead L4.5", ahead=2, L=4.5))
    for nm, ah, kw in (("a0 ", 0, {}), ("a0h", 0, dict(lead=8.0)), ("a1 ", 1, {}), ("a2 ", 2, {})):
        A.append(Cfg(f"CORNER {nm} L4.5 ss1.1", ahead=ah, L=4.5, ss=1.1, **kw))
    A.append(Cfg("a0h L4.5 + tail24 (4-stage pieces, 4 RT resolve)", ahead=0, lead=8.0, L=4.5, tail=24))
    A.append(Cfg("a0h L4.5 + tail24, resolve 1 RT (break-even probe)", ahead=0, lead=8.0, L=4.5, tail=24, chain=1.0))
    A.append(Cfg("a0h L4.5 ss1.04 (break-even probe)", ahead=0, lead=8.0, L=4.5, ss=1.04))
    A.append(Cfg("IID a0h L4.5 (noise per ticket, not per CTA)", ahead=0, lead=8.0, L=4.5, iid=True))
    A.append(Cfg("IID CORNER a0h L4.5 ss1.1", ahead=0, lead=8.0, L=4.5, ss=1.1, iid=True))
    return A


def evaluate(AR, nd=3, seed=11, marks=None, sdscale=1.0, iid=False, quiet=True):
    """run all arms on the 3 router seeds x 2 layers x 5 reps x nd noise draws per T; returns (acc, info, diag).  iid: TQ draws one noise factor per
    ticket (sd = persistent sd x sqrt(units per CTA), same spread per CTA) instead of the per-CTA factor the ship keeps (rebalancing cannot help)."""
    t0 = time.time()
    runs = iwg.load_all(marks or str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    Ts = (32, 128, 256)
    P = {T: e4.setup(T, runs) for T in Ts}
    ship1 = iwg.Cfg("ship+L", nwg=1, add=0.0, lrel=3.5, xm="free", tma=0.0)
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    acc = {T: {} for T in Ts}
    info = {T: [] for T in Ts}
    diag = {T: {} for T in Ts}
    for key in sorted(runs):
        T, rs, ly = key
        v = runs[key]
        if not v["ok"]:
            continue
        p = P[T]
        sp_, sb_ = e4.SIG[T]
        sp_, sb_ = sp_ * sdscale, sb_ * sdscale
        sj = e4.ship_jobs(v["jobs"], p)
        ship_cost = e4.ext_cost(e4.COST[T])
        info[T].append((len(sj), len(build_queue(sj, Cfg("x")))))
        for ctas in v["recs"]:
            cs = sorted(ctas)
            s1 = iwg.Sim(sj, ship_cost, ctas, p, ship1)
            for _ in range(nd):
                ea = {c: float(np.exp(rng.normal(0, sp_))) for c in cs}
                eb = {c: ea[c] * float(np.exp(rng.normal(0, sb_))) for c in cs}
                r0, _i = e4.simulate(sj, ship_cost, ctas, p, eps=(ea, eb))
                b0 = np.array([b for _a, b in r0.values()])
                row = {"ship": (b0.max(), np.median(b0))}
                r1, _x = s1.run((ea, eb))
                b1 = np.array(list(r1.values()))
                row["ship+L3.5"] = (b1.max(), np.median(b1))
                um = None
                if iid or any(c.iid for c in AR):
                    z = rng2.normal(0, 1, (3, len(sj), NA, 4))
                    U = len(build_queue(sj, Cfg("x"))) / len(cs)

                    def um(Q, z=z, U=U, sp_=sp_, sb_=sb_):
                        return np.array([float(np.exp(z[t[0], t[1], t[2], t[3]] * (sp_ if t[0] == 0 else float(np.hypot(sp_, sb_))) * np.sqrt(U))) for t in Q])
                for c in AR:
                    end, sd = tq_run(sj, p, ctas, (ea, eb), c, um if (iid or c.iid) else None)
                    bb = np.array(list(end.values()))
                    row[c.name] = (bb.max(), np.median(bb))
                    diag[T].setdefault(c.name, []).append((sd["wgate"] / len(cs), sd["wclaim"] / len(cs)))
                for nme, mv in row.items():
                    acc[T].setdefault(nme, {}).setdefault(rs, []).append(mv)
        if not quiet:
            print(f"  [{time.time() - t0:5.0f}s] T{T} {rs % 100000} {ly}", file=sys.stderr, flush=True)
    return acc, info, diag


def agg(acc, T, nme, k):
    """median over router seeds of the seed-mean (max or p50) delta vs ship (CRN pairs)."""
    out = []
    for rs, lst in acc[T][nme].items():
        sh = acc[T]["ship"][rs]
        out.append(np.mean([x[k] - y[k] for x, y in zip(lst, sh)]))
    return float(np.median(out))


def lvl(acc, T, nme, k):
    return float(np.median([np.mean([x[k] for x in lst]) for lst in acc[T][nme].values()]))


def persist(marks=None):
    """fragile-assumption probe: per-CTA residual (measured end - noiseless generative ship end), mean over the 5 reps of a routing.  If the ship's end
    tail were CTA-intrinsic slowness (what a ticket queue can shed), residuals of the same CTA would correlate across DIFFERENT routings."""
    runs = iwg.load_all(marks or str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    for T in (32, 128, 256):
        p = e4.setup(T, runs)
        cost = e4.ext_cost(e4.COST[T])
        R, W = {}, []
        for key, v in sorted(runs.items()):
            if key[0] != T or not v["ok"]:
                continue
            sj = e4.ship_jobs(v["jobs"], p)
            rr = []
            for ctas in v["recs"]:
                r0, _i = e4.simulate(sj, cost, ctas, p)
                rr.append(np.array([ctas[c]["bend"] - r0[c][1] for c in sorted(ctas)]))
            rr = np.array(rr)
            R[key] = rr.mean(0)
            W.append(np.corrcoef(rr[[0, 2, 4]].mean(0), rr[[1, 3]].mean(0))[0, 1])
        C = np.corrcoef(np.array(list(R.values())))
        off = C[np.triu_indices(len(R), 1)]
        print(f"T{T}: residual sd {np.mean([r.std() for r in R.values()]):.1f} us | same-routing split-half corr {np.mean(W):.2f} | "
              f"cross-routing corr {off.mean():.2f} (min {off.min():.2f}, max {off.max():.2f}, {len(off)} pairs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default=str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    ap.add_argument("--nd", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default="")
    ap.add_argument("--persist", type=int, default=0, help="1: print the CTA-residual persistence probe (3 rows) and exit")
    ap.add_argument("--sdscale", type=float, default=1.0, help="scale on the persistent per-CTA pace noise (0 = noiseless ship and TQ)")
    ap.add_argument("--iid", type=int, default=0, help="1: TQ pace noise drawn per ticket (not persistent per CTA); ship keeps the calibrated per-CTA factor")
    a = ap.parse_args()
    if a.persist:
        return persist(a.marks)
    t0 = time.time()
    Ts = (32, 128, 256)
    AR = [x for x in arms() if a.only in x.name]
    acc, info, diag = evaluate(AR, a.nd, a.seed, a.marks, a.sdscale, bool(a.iid), quiet=False)
    n_ = 0

    def out_(x):
        nonlocal n_
        n_ += 1
        print(x)

    tag = f" [sdscale {a.sdscale} iid {a.iid}]" if (a.sdscale != 1.0 or a.iid) else ""
    out_("R-tq-0 ($0 CPU): ticket queue vs ship, max / p50 end over 132 CTAs, us (d vs ship on the same noise draws, median over 3 router seeds; marks ~10 % slow)" + tag + ". "
         "ship max/p50 " + "  ".join(f"T{T} {lvl(acc, T, 'ship', 0):.1f}/{lvl(acc, T, 'ship', 1):.1f}" for T in Ts)
         + " | jobs/units T32 {:.0f}/{:.0f} T128 {:.0f}/{:.0f} T256 {:.0f}/{:.0f}".format(*[x for T in Ts for x in np.mean(info[T], axis=0)])
         + " | ship+L3.5 d max " + " ".join(f"{agg(acc, T, 'ship+L3.5', 0):+.1f}" for T in Ts))
    out_("arm                                                  | d max T32  T128  T256 | TQ max level T32  T128  T256 | d p50 T32  T128  T256 | idle gate/claim us per CTA (T32 T128 T256) | GO(<=-5 at T32 and T128|T256)")
    for c in AR:
        nm = c.name
        dm = [agg(acc, T, nm, 0) for T in Ts]
        dp = [agg(acc, T, nm, 1) for T in Ts]
        lv = [lvl(acc, T, nm, 0) for T in Ts]
        dg = [np.mean(diag[T][nm], axis=0) for T in Ts]
        go = dm[0] <= -5.0 and min(dm[1], dm[2]) <= -5.0
        out_(f"{nm:52s} | {dm[0]:+6.1f} {dm[1]:+6.1f} {dm[2]:+6.1f} | {lv[0]:6.1f} {lv[1]:6.1f} {lv[2]:6.1f} | {dp[0]:+6.1f} {dp[1]:+6.1f} {dp[2]:+6.1f} | "
             + " ".join(f"{x[0]:.1f}/{x[1]:.1f}" for x in dg) + f" | {'GO' if go else 'no'}")
    corner = [x for x in AR if x.name.startswith("CORNER") and x.ahead <= 2 and not x.tail]
    gos = [(x.name, agg(acc, 32, x.name, 0), min(agg(acc, 128, x.name, 0), agg(acc, 256, x.name, 0))) for x in corner]
    best = min(gos, key=lambda z: max(z[1], z[2])) if gos else None
    out_("VERDICT (corner = L4.5, ss+10 %, ahead <= 2): " + ("GO" if any(z[1] <= -5 and z[2] <= -5 for z in gos) else "KILL") + "; best corner arm "
         + (f"{best[0]}: d max T32 {best[1]:+.1f}, best of T128/T256 {best[2]:+.1f}" if best else "-") + f" | rows printed {n_ + 1}")
    print(f"total {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

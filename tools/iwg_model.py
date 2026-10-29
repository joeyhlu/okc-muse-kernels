"""iwg_model.py: $0 CPU gate for INDEPENDENT WARPGROUPS (IWG) in the VML MoE at T >= 32 (lab R-iwg-0).

    python tools/iwg_model.py [--nd=2] [--check=1] [--marks=box_out/lat32/m_ship/marks_raw.json.gz]     (~6-10 min, prints <= 20 rows)

Ship: 4 consumer WGs share every 36 KB stage (256 rows), all reach a job end together; the planner balances 132 CTAs with K-splits.
IWG: every WG runs its own stream of 64-row x full-K jobs (16 stages of 9 KB); 528 bins; segment ends no longer coincide.

Machinery = tools/e4_model.py (marks-fitted generative ship model, planner closed form, CRN noise) generalized to nwg WG streams per
CTA (`sim`). nwg = 1 with zero charges reproduces e4_model.simulate (null check printed). IWG charges (all on top of the ship fits):
  * stage: today's tier pace + 0.045 us per WG stage (R-vml-14), x ss; first stage + 0.045; per-piece end work, first stage, drain, g as ship
  * 528 WG pieces: A block = 16 steps (unit key = step >> 4), B stage s of a 64-row block needs the 8 A blocks 8s..8s+7 (doneA per block);
    one doneA release per 64-row block = ship's DEFER release time + L_rel (1 dependent L2 round trip, 2.5 / 3.5 / 4.5 us)
  * K-splits that remain (planner quantum q stages, q = 16 means whole blocks) keep the ship split-share / resolver end work (fitted)
  * x supply (xm): 'free' (WGs share x for free: upper bound), 'smem' (per-WG x ring in the smem left beside the weight rings: Little's law
    depth, latency XL(T), x re-read 4x through L2->SM at bw TB/s), 'l2' (same, latency 4.5 us, bw 6.5)
  * dyn planner = ticket queue of whole 64-row blocks (no K-splits), the next ticket is taken at block start
Only max / p50 deltas against the same noise draws count (marks ran ~10 % slow).
"""
import argparse
import heapq
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_model as ae  # noqa: E402
import e4_model as e4  # noqa: E402

ROOT, G = ae.ROOT, ae.G
COST32 = [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]   # VML_COST[32], k3_gc_21.py l.8990
e4.COST[32] = COST32
e4.SIG[32] = (0.065, 0.02)          # calibrated here: gen max 107.4 / B spread 11.8 vs measured 107.5 / 13.5 (T 128 / 256 keep e4's)
SMEM_MAX, SMEM_OVH = 232448, 4300   # 227 KB per block; barriers + Tab (w72 free ~7 KB beside 3 x 36 KB + 3 x 36 KB)
WSTAGE = 9216                       # 64 rows x 256 K weights stage (36864 / 4)


class Cfg:
    """one arm. nwg 1 = ship machinery; 4 = IWG. add / ss: per-stage cost; lrel: doneA release latency; xm: x supply; q: planner quantum (stages)."""

    def __init__(s, name, nwg=4, q=1, plan="seq", add=0.045, ss=1.0, lrel=3.5, xm="smem", bw=8.0e12, lat=None, dw=2, tma=0.15, rho=0.0, refit=False, ew=1.0):
        s.__dict__.update(locals())
        del s.__dict__["s"]


def eff_jobs(jobs, p, cfg):
    """per job (pA, fA, pB) for this arm: ship values for nwg 1; IWG: (base / ship x-factor + add) x ss, then the x-supply limits."""
    out = []
    for jb in jobs:
        if cfg.nwg == 1 and cfg.add == 0 and cfg.ss == 1.0 and cfg.xm == "free":
            out.append((jb.pA, jb.fA, jb.pB))
            continue
        n = jb.N
        pcs = []
        for base in (jb.pA, jb.pB):
            lat = cfg.lat if cfg.lat is not None else p.XL
            fs = max(1.0, (lat + base) / (3.0 * base)) if cfg.xm != "free" else 1.0   # ship's own 3-deep shared x ring is inside the measured pace
            pc = (base / fs + cfg.add) * cfg.ss
            if cfg.xm != "free":
                dx = max(0.25, (SMEM_MAX - SMEM_OVH - 4 * cfg.dw * WSTAGE) / 4.0 / (512.0 * n))     # per-WG x ring depth (stage equivalents)
                pc = max(pc, (lat + pc) / dx)
                pc = max(pc, G * (36864 + 4 * 512 * n) / cfg.bw * 1e6)                              # x re-read by 4 WGs through L2 -> SM
            pc = max(pc, 4 * cfg.tma)
            pcs.append(pc)
        out.append((pcs[0], jb.fA + cfg.add, pcs[1]))
    return out


def refit_cost(p, cost):
    """VCT proxy: planner cost per step proportional to the fitted tier pace (class 0 = 128); classes without data keep the shipped entry."""
    out = []
    for pb, pace in ((0, p.pA), (1, p.pB)):
        h = list(cost[pb])
        for k in range(8):
            if pace[k] > 0 and pace[0] > 0:
                h[k] = int(round(128 * pace[k] / pace[0]))
        out.append(h)
    return out


def plan2(nts, cost, g, nw, q):
    """e4.plan generalized: g bins, nw x more steps per job (same cost per step), boundaries rounded to a multiple of q steps."""
    res = []
    for pb in (0, 1):
        sc = (512 >> pb) * nw
        cs = [0]
        for nt in nts:
            cs.append(cs[-1] + sc * cost[pb][nt])
        tot = cs[-1]
        b, j = [], 0
        for c in range(g + 1):
            x = (tot * c) // g
            if c >= g:
                b.append(len(nts) * sc)
                continue
            while j + 1 < len(nts) and cs[j + 1] <= x:
                j += 1
            b.append(j * sc + min((x - cs[j]) // cost[pb][nts[j]], sc - 1))
        if q > 1:
            b = [min(len(nts) * sc, int(round(v / q)) * q) for v in b]
            for k in range(1, len(b)):
                b[k] = max(b[k], b[k - 1])
        res.append(b)
    return res


class Sim:
    """one (routing, rep): static geometry that does not depend on the noise draw."""

    def __init__(s, jobs, cost, ctas, p, cfg):
        if cfg.refit:
            cost = refit_cost(p, cost)
        s.jobs, s.ctas, s.p, s.cfg, s.cost = jobs, ctas, p, cfg, cost
        nw = cfg.nwg
        s.nw, s.act = nw, sorted(ctas)
        s.scA, s.scB = 512 * nw, 256 * nw
        s.EJ = eff_jobs(jobs, p, cfg)
        ew = cfg.ew
        s.eAfull = [ew * p.eAf("full", jb.N) for jb in jobs]
        s.eApart = [ew * p.eAf("part", jb.N) for jb in jobs]
        s.eAres = [ew * p.eAf("res", jb.N) for jb in jobs]
        s.eBfull = [ew * p.eBf("full", jb.N) for jb in jobs]
        s.eBpart = [ew * p.eBf("part", jb.N) for jb in jobs]
        s.wgs = [c * nw + i for c in s.act for i in range(nw)]
        s.dyn = cfg.plan == "dyn"
        if not s.dyn:
            bA, bB = plan2([jb.nt for jb in jobs], cost, G * nw, nw, cfg.q)
            s.segA = {w: ae.segs_a(bA[w], bA[w + 1]) for w in s.wgs}
            s.segB = {w: ae.segs_b(bB[w], bB[w + 1]) for w in s.wgs}
            s.sh = {}
            for w in s.wgs:
                for i, (a, b) in enumerate(s.segA[w]):
                    s.sh.setdefault(a >> 4, []).append((w, i, b - a == 16))
            s.res0 = {}
            for lst in s.sh.values():
                if len(lst) > 1:
                    w, i, _ = max(lst, key=lambda x: (x[1], x[0]))
                    s.res0[(w, i)] = True

    # ---- phase A
    def fwd(s, w, res, epA, scl=None):
        p = s.p
        c = w // s.nw
        fc = s.ctas[c]["fc"]
        e = epA.get(c, 1.0)
        t, K, out = fc, 0, []
        for i, (a, b) in enumerate(s.segA[w]):
            j, n = a // s.scA, b - a
            pa, fa, _pb = s.EJ[j]
            K += n
            sc = scl.get((w, i), 1.0) if scl else 1.0
            te = max(t + ((pa if i == 0 else fa) + (n - 1) * pa) * e * sc, fc + (K - 1) * p.tau)
            if n == 16:
                pre, rsv = s.eAfull[j], 0.0
            else:
                pre, rsv = s.eApart[j], (s.eAres[j] if res.get((w, i)) else 0.0)
            arr = te + p.d + pre
            t4 = arr + rsv
            out.append((t, te, arr, t4, fa))
            t = t4 + p.g
        return out

    def phaseA(s, epA):
        scl = None
        for rep in range(1 + (2 if s.cfg.rho > 0 else 0)):
            res = dict(s.res0)
            for _ in range(4):
                tl = {w: s.fwd(w, res, epA, scl) for w in s.wgs}
                new = {}
                for lst in s.sh.values():
                    if len(lst) > 1:
                        w, i, _ = max(lst, key=lambda x: (tl[x[0]][x[1]][2], x[1], x[0]))
                        new[(w, i)] = True
                if new == res:
                    break
                res = new
            if s.cfg.rho > 0:
                scl = s.relief(tl)
        return tl, res

    def relief(s, tl):
        """contention relief (upper-bound sensitivity): while m siblings of a WG are in drain + end work, its compute runs faster by rho x m / 4."""
        nw, rho = s.nw, s.cfg.rho
        scl = {}
        for c in s.act:
            ws = [c * nw + i for i in range(nw)]
            stalls = {w: [(x[1], x[3]) for x in tl[w]] for w in ws}
            for w in ws:
                for i, x in enumerate(tl[w]):
                    a, b = x[0], x[1]
                    if b <= a:
                        continue
                    ov = 0.0
                    for w2 in ws:
                        if w2 != w:
                            for (u, v) in stalls[w2]:
                                if v > a and u < b:
                                    ov += min(b, v) - max(a, u)
                    scl[(w, i)] = max(0.3, 1.0 - rho / nw * ov / (b - a))
        return scl

    def releases(s, tl, res, fcs):
        p = s.p
        aend, rel = {}, {}
        for w in s.wgs:
            s_ = tl[w]
            aend[w] = s_[-1][3] + p.phiA if s_ else fcs[w]
            rel[w] = [(s_[i + 1][0] + p.kf * s_[i + 1][4] + p.phiF) if i < len(s_) - 1 else aend[w] for i in range(len(s_))]
        R = {}
        for u, lst in s.sh.items():
            r = [(w, i) for w, i, f in lst if res.get((w, i)) or f] or [(w, i) for w, i, f in lst]
            R[u] = max(rel[w][i] for w, i in r) + s.cfg.lrel
        return aend, R

    # ---- dynamic ticket queue (whole 64-row blocks)
    def dynA(s, epA):
        p, nw = s.p, s.nw
        nj = len(s.jobs)
        nb = nj * 32 * nw
        counter = 0
        heap = []
        for w in s.wgs:
            if counter < nb:
                heap.append((s.ctas[w // nw]["fc"], w, counter))
                counter += 1
        heapq.heapify(heap)
        pcs = {w: [] for w in s.wgs}
        Kc = {w: 0 for w in s.wgs}
        while heap:
            ts, w, blk = heapq.heappop(heap)
            c = w // nw
            fc = s.ctas[c]["fc"]
            e = epA.get(c, 1.0)
            j = blk // (32 * nw)
            pa, fa, _pb = s.EJ[j]
            Kc[w] += 16
            first = not pcs[w]
            te = max(ts + (pa if first else fa) * e + 15 * pa * e, fc + (Kc[w] - 1) * p.tau)
            t4 = te + p.d + s.eAfull[j]
            pcs[w].append((ts, te, t4, t4, fa, blk))
            if counter < nb:
                heapq.heappush(heap, (t4 + p.g, w, counter))
                counter += 1
        s.segA = {w: [(blk * 16, blk * 16 + 16) for *_x, blk in pcs[w]] for w in s.wgs}
        tl = {w: [x[:5] for x in pcs[w]] for w in s.wgs}
        s.sh = {}
        for w in s.wgs:
            for i, (a, b) in enumerate(s.segA[w]):
                s.sh.setdefault(a >> 4, []).append((w, i, True))
        return tl, {}

    def bpiece(s, w, t, K, a, b, aend, R, epB, first):
        """run B steps a..b (one 16-aligned piece) of WG w from time t; returns (t after end work, K, wait)."""
        p, nw = s.p, s.nw
        c = w // nw
        fc = s.ctas[c]["fc"]
        e = epB.get(c, 1.0)
        ws = 0.0
        j0 = a // s.scB
        for gi in range(a, b):
            j, sl = gi // s.scB, gi & 15
            base = 32 * nw * j + 2 * nw * sl
            r = max(R.get(base + x, 0.0) for x in range(2 * nw)) + p.XL
            K += 1
            stt = max(t, r, fc + (K - 1) * p.tau, aend + p.xb0 if (first and gi == a) else 0.0)
            ws += max(0.0, r - t)
            t = stt + s.EJ[j][2] * e
        t += p.d + (s.eBfull[j0] if b - a == 16 else s.eBpart[j0]) + p.g
        return t, K, ws

    def dynB(s, aend, R, epB, na):
        nj = len(s.jobs)
        nb = nj * 16 * s.nw
        heap = [(aend[w] + s.p.g0, w) for w in s.wgs]
        heapq.heapify(heap)
        pcs = {w: [] for w in s.wgs}
        K = dict(na)
        counter = 0
        free = {}
        while counter < nb and heap:
            ts, w = heapq.heappop(heap)
            blk = counter
            counter += 1
            t, K[w], _ws = s.bpiece(w, ts, K[w], blk * 16, blk * 16 + 16, aend[w], R, epB, not pcs[w])
            pcs[w].append((blk * 16, blk * 16 + 16))
            heapq.heappush(heap, (t, w))
        for t, w in heap:
            free[w] = t
        s.segB = pcs
        return free

    def run(s, eps):
        epA, epB = eps
        p, nw = s.p, s.nw
        fcs = {w: s.ctas[w // nw]["fc"] for w in s.wgs}
        tl, res = s.dynA(epA) if s.dyn else s.phaseA(epA)
        aend, R = s.releases(tl, res, fcs)
        na = {w: sum(b - a for a, b in s.segA[w]) for w in s.wgs}
        if s.dyn:
            s.dynB(aend, R, epB, na)
        end = {}
        for c in s.act:
            ends = []
            for i in range(nw):
                w = c * nw + i
                t, K = aend[w] + p.g0, na[w]
                for k, (a, b) in enumerate(s.segB[w]):
                    t, K, _ws = s.bpiece(w, t, K, a, b, aend[w], R, epB, k == 0)
                ends.append(t)
            end[c] = max(ends)
        return end, aend


def split_counts(jobs, cost, cfg, act_n=G):
    """partial (n < 16) A and B pieces per CTA for a planner (no noise)."""
    nts = [jb.nt for jb in jobs]
    bA, bB = plan2(nts, cost, G * cfg.nwg, cfg.nwg, cfg.q)
    n = 0
    for w in range(G * cfg.nwg):
        n += sum(1 for a, b in ae.segs_a(bA[w], bA[w + 1]) if b - a < 16) + sum(1 for a, b in ae.segs_b(bB[w], bB[w + 1]) if b - a < 16)
    return n / G


def load_all(path):
    """e4.load without the T = 32 skip: {(T, rs, layer): dict(recs, jobs, ok, ld)}."""
    import gzip
    import json
    raw = json.load(gzip.open(path, "rt"))
    out = {}
    for key, reps in raw.items():
        rs, ld, ly = key.split("|")
        T = dict(ae.LOADS)[ld]
        ja, jb = e4.decode_jobs(reps[0], 512, 8), e4.decode_jobs(reps[0], 256, 16)
        out[(T, int(rs), ly)] = dict(recs=[ae.parse(r) for r in reps], jobs=ja, ok=ja is not None and ja == jb, ld=ld)
    return out


def arms_list():
    A = []
    A.append(Cfg("I0 q1 no charge, x free", add=0.0, lrel=0.0, xm="free", tma=0.0))
    A.append(Cfg("I2 q1 +0.045/stage +release 3.5, x free", xm="free", tma=0.0))
    A.append(Cfg("C  q1 per-WG x ring (spec)"))
    A.append(Cfg("C  q16 whole blocks, seq", q=16))
    A.append(Cfg("C  dyn ticket queue (IWG central)", plan="dyn"))
    A.append(Cfg("FX dyn, x free (shared window)", xm="free", plan="dyn", tma=0.0))
    A.append(Cfg("CTRL ship + dyn unit queue (no IWG)", nwg=1, plan="dyn", add=0.0, lrel=0.0, xm="free", tma=0.0))
    A.append(Cfg("CTRL ship + refit cost table (VCT proxy)", nwg=1, add=0.0, lrel=0.0, xm="free", tma=0.0, refit=True))
    A.append(Cfg("K  rel 2.5 (C dyn)", plan="dyn", lrel=2.5))
    A.append(Cfg("K  rel 4.5 (C dyn)", plan="dyn", lrel=4.5))
    A.append(Cfg("K  stage -10 % (C dyn)", plan="dyn", ss=0.9))
    A.append(Cfg("K  stage +10 % (C dyn)", plan="dyn", ss=1.1))
    A.append(Cfg("K  x l2 lat 4.5 bw 6.5 (C dyn)", plan="dyn", xm="l2", lat=4.5, bw=6.5e12))
    A.append(Cfg("PES (4.5, +10 %, x l2) dyn", plan="dyn", lrel=4.5, ss=1.1, xm="l2", lat=4.5, bw=6.5e12))
    A.append(Cfg("PES-FX (4.5, +10 %, x free) dyn", plan="dyn", lrel=4.5, ss=1.1, xm="free", tma=0.0))
    A.append(Cfg("RHO 1.0 on I0 (sibling relief bound)", add=0.0, lrel=0.0, xm="free", tma=0.0, rho=1.0))
    A.append(Cfg("BOUND I2 + end work x0.5 + rho 1.0, x free", xm="free", tma=0.0, rho=1.0, ew=0.5))
    A.append(Cfg("BOUND I0 + end work x0.5 + rho 1.0, no stage/rel charge", add=0.0, lrel=0.0, xm="free", tma=0.0, rho=1.0, ew=0.5))
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default=str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    ap.add_argument("--nd", type=int, default=2, help="noise draws per rep")
    ap.add_argument("--check", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--only", default="", help="substring filter of arm names (debug)")
    a = ap.parse_args()
    t0 = time.time()
    runs = load_all(a.marks)
    Ts = (32, 128, 256)
    P = {T: e4.setup(T, runs) for T in Ts}
    arms = [x for x in arms_list() if a.only in x.name]
    ship1 = Cfg("ship+L (nwg1, release charged to ship)", nwg=1, add=0.0, lrel=3.5, xm="free", tma=0.0)
    null = Cfg("null nwg1", nwg=1, add=0.0, lrel=0.0, xm="free", tma=0.0)
    rng = np.random.default_rng(a.seed)
    # acc[T][arm] -> {rs: [ (max, p50) per sample ]}; ship measured as well
    acc = {T: {} for T in Ts}
    nullerr = 0.0
    spl = {T: [] for T in Ts}
    for key in sorted(runs):
        T, rs, ly = key
        v = runs[key]
        if not v["ok"]:
            continue
        p = P[T]
        sp_, sb_ = e4.SIG[T]
        sj = e4.ship_jobs(v["jobs"], p)
        ship_cost = e4.ext_cost(e4.COST[T])
        spl[T].append((split_counts(sj, ship_cost, Cfg("s", nwg=1)), split_counts(sj, ship_cost, Cfg("i", nwg=4, q=1)), split_counts(sj, ship_cost, Cfg("i", nwg=4, q=4))))
        for ri, ctas in enumerate(v["recs"]):
            cs = sorted(ctas)
            sims = [(c, Sim(sj, ship_cost, ctas, p, c)) for c in arms + [ship1]]
            for _ in range(a.nd):
                ea = {c: float(np.exp(rng.normal(0, sp_))) for c in cs}
                eb = {c: ea[c] * float(np.exp(rng.normal(0, sb_))) for c in cs}
                r0, _i = e4.simulate(sj, ship_cost, ctas, p, eps=(ea, eb))
                b0 = np.array([b for _a, b in r0.values()])
                row = {"ship": (b0.max(), np.median(b0))}
                if a.check and ri == 0:
                    rn, _x = Sim(sj, ship_cost, ctas, p, null).run((ea, eb))
                    nullerr = max(nullerr, max(abs(rn[c] - r0[c][1]) for c in cs))
                for c, sm in sims:
                    r1, _x = sm.run((ea, eb))
                    b1 = np.array(list(r1.values()))
                    row[c.name] = (b1.max(), np.median(b1))
                for nme, mv in row.items():
                    acc[T].setdefault(nme, {}).setdefault(rs, []).append(mv)
        print(f"  [{time.time() - t0:5.0f}s] T{T} {rs % 100000} {ly}", file=sys.stderr, flush=True)

    def agg(T, nme, k):
        """median over seeds of the seed-mean (max or p50) delta vs ship (CRN pairs)."""
        out = []
        for rs, lst in acc[T][nme].items():
            sh = acc[T]["ship"][rs]
            out.append(np.mean([x[k] - y[k] for x, y in zip(lst, sh)]))
        return float(np.median(out))

    def lvl(T, nme, k):
        return float(np.median([np.mean([x[k] for x in lst]) for lst in acc[T][nme].values()]))

    LINES = []

    def out_(x):
        LINES.append(x)
        print(x)

    spm = {T: np.mean(spl[T], axis=0) for T in Ts}
    out_("R-iwg-0 ($0 CPU): IWG vs ship VML, max / p50 end over 132 CTAs, us (marks ~10 % slow; deltas vs the same noise draws, median over 3 seeds); "
         f"null check (nwg 1 == e4.simulate) max |diff| {nullerr:.2e}")
    out_("ship gen max / p50 T32 {:.1f}/{:.1f}  T128 {:.1f}/{:.1f}  T256 {:.1f}/{:.1f} | partial pieces (A+B) per CTA ship / IWG q1 / IWG q4: T32 {:.1f}/{:.1f}/{:.1f} T128 {:.1f}/{:.1f}/{:.1f} T256 {:.1f}/{:.1f}/{:.1f}".format(
        *[lvl(T, "ship", k) for T in Ts for k in (0, 1)], *[x for T in Ts for x in spm[T]]))
    out_("arm (d vs ship; negative = IWG better)                | d max T32  T128  T256 | d p50 T32  T128  T256 | GO(>=4 at T32 and at T128|T256)")
    comp = {"K  rel 2.5 (C dyn)": "K  rel 4.5 (C dyn)", "K  stage -10 % (C dyn)": "K  stage +10 % (C dyn)"}
    skip = set(comp.values()) | {"PES-FX (4.5, +10 %, x free) dyn"}
    for c in arms:
        nm = c.name
        if nm in skip:
            continue
        dm = [agg(T, nm, 0) for T in Ts]
        dp = [agg(T, nm, 1) for T in Ts]
        go = dm[0] <= -4.0 and min(dm[1], dm[2]) <= -4.0
        if nm in comp:
            d2 = [agg(T, comp[nm], 0) for T in Ts]
            out_(f"{nm[:3]}{nm[3:].split(' (')[0]:20s} (C dyn) -> {comp[nm][3:].split(' (')[0]:12s} d max: " + " ".join(f"{x:+6.1f}" for x in dm) + "  ->  " + " ".join(f"{x:+6.1f}" for x in d2))
            continue
        out_(f"{nm:46s} | {dm[0]:+6.1f} {dm[1]:+6.1f} {dm[2]:+6.1f} | {dp[0]:+6.1f} {dp[1]:+6.1f} {dp[2]:+6.1f} | {'GO' if go else 'no'}")
    c_ = "C  dyn ticket queue (IWG central)"
    out_("IWG central (C dyn) max / p50 levels: " + "  ".join(f"T{T} {lvl(T, 'ship', 0):.1f}/{lvl(T, 'ship', 1):.1f} -> {lvl(T, c_, 0):.1f}/{lvl(T, c_, 1):.1f}" for T in Ts)
         + " | PES-FX dyn d max " + " ".join(f"{agg(T, 'PES-FX (4.5, +10 %, x free) dyn', 0):+.1f}" for T in Ts)
         + " | ship+L(3.5) d max " + " ".join(f"{agg(T, ship1.name, 0):+.1f}" for T in Ts))
    # smem verdict
    n32, n72 = 32, 72
    wts = 4 * 3 * WSTAGE
    row = []
    for n in (8, 16, 32, 72):
        need = 4 * 3 * 512 * n
        fit = (SMEM_MAX - SMEM_OVH - 4 * 2 * WSTAGE) / 4.0 / (512.0 * n)
        row.append(f"N{n}: x 3-deep x4 WG = {need / 1024:.0f} KB (fits depth {fit:.1f} with 2-deep weights)")
    out_(f"smem (227 KB, ovh {SMEM_OVH / 1024:.1f} KB): weights 4 WG x 3 x 9 KB = {wts / 1024:.0f} KB (= ship 3 x 36 KB); " + "; ".join(row))
    print(f"total {time.time() - t0:.0f}s", file=sys.stderr)
    return LINES


if __name__ == "__main__":
    main()

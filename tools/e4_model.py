"""e4_model.py: $0 CPU gate for lever E4 (lab R-e4-0): hot experts as ONE wide job (<= 128 tokens) in the decode_layer VML MoE.

    python tools/e4_model.py [--nd=8] [--seed=11] [--marks=box_out/lat32/m_ship/marks_raw.json.gz] [--calib=1]    (~50 s, prints 18 rows)

Question: at T = 256 (and T = 128 whenever an expert has > 72 tokens) does running each hot expert as one job of <= 128 tokens instead
of the shipped <= 72-token passes (w72) move the MAX over CTAs of the vm_main end by >= 5 us after every charge, and does a register
plan close for N = 128 (64 fp32 accumulators per thread)?  GO needs both.

Method (every number is a difference against the same generative model, so the +9-11 % marks slowdown and the model bias cancel):
  1. Ship plan.  The per-CTA marks of tools/box/lat32_marks.py (box_out/lat32/m_ship: 3 router seeds x 2 layers x 5 reps) give the nt class
     of every job (decoded from the per-CTA class counts; the A and B decodes agree).  The planner's closed form (k3_gc_21 route() /
     stage_at: equal COST split of 512 x cost[nt] (A) / 256 x cost[nt] (B) per job over 132 CTAs, VML_COST[T]) reproduces every
     measured per-CTA stage range exactly (printed plan check).
  2. Generative per-CTA timeline (forward; ae_model semantics: 16-aligned unit segments, split_point rotation, share / resolver end work,
     DEFER doneA releases, B stage = max(free, release + XL, first-slice floor, stream floor)).  Parameters are regressed per T on the marks:
     steady and first stage per nt class, end work per segment kind x N, B stage per class, B end work.  The measured tails come from
     per-CTA pace noise (persistent sd + extra B sd, SIG, calibrated to the measured A / B end spread with --calib=1); arms share the
     noise draws (common random numbers).  Fidelity (generative ship vs measured) is printed.
  3. E4 = the same model with every consistent hot pair merged into one wide job.  A pair is two adjacent jobs (class a, b) that the
     route() TA split produces for 73..128 tokens (PAIRS); tot = the middle of the class range.  Planner re-run with a cost entry for the
     wide class (lam x cost / pace of the widest shipped class x wide pace).  Wide stage = ship stages x PP(N) / (w72(b0) + w72(b1))
     (R-xc-0 cold table, by N; pp is measured at N 8..112, N <= 128 is an extrapolation), end work at the wide N (split shares move
     2x the bytes), first-stage extra, one boundary charge per engine switch, HBM relief of the removed bytes (upper bound).
     Arms add the charges one by one (g, hbm, end, fst, e4), plus envelope arms (cost table, doubled charges, stage ratio, read-back
     round trip, null merge) and the engines that could close the registers (3-WG x 160, PP for every job).
  Limits: the hot pairs are decoded from class counts (11 pairs in 6 routings = 1.83 per routing vs 1.67 in the saved-routing census);
  the oracle E4 puts the measured PP rate for wide jobs inside the unchanged w72 consumer, which the register check says cannot exist.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ae_model as ae  # noqa: E402

ROOT = ae.ROOT
G = ae.G
NM = ae.NMEAN                      # N of nt class 0..7 (class 7 = 57..72 -> 68)
# VML_COST[T] (k3_gc_21.py line 8990): 8 A entries then 8 B entries, nt 0..7; w72 extends linearly (_vml_cost_ext)
COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
        128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267]}
# R-xc-0 (notes/study/xc_moe.md s5, cold = HBM, stage = 256 rows x 256 K, probe clocks 1.65-1.78 GHz): us per stage by N
INC = {8: 1.155, 16: 1.307, 32: 1.617, 56: 2.040, 64: 2.167}
W72 = {32: 1.537, 56: 1.955, 64: 2.075}
PP = {8: 1.302, 16: 1.436, 32: 1.726, 56: 2.216, 64: 2.372, 112: 3.208}
W72[8], W72[16] = INC[8] * W72[32] / INC[32], INC[16] * W72[32] / INC[32]     # w72 not listed below N 32: inc x (w72 / inc at N 32)
NMAX = 15                          # nt classes 0..15 (N <= 128)


def interp(tab, n):
    """piecewise-linear in N, linear extrapolation with the last / first slope."""
    ks = sorted(tab)
    if n <= ks[0]:
        a, b = ks[0], ks[1]
    elif n >= ks[-1]:
        a, b = ks[-2], ks[-1]
    else:
        a = max(k for k in ks if k <= n)
        b = min(k for k in ks if k > n)
    return tab[a] + (tab[b] - tab[a]) * (n - a) / (b - a)


def ext_cost(c16, nmax=NMAX):
    """_vml_cost_ext: each phase's 8 entries, then h[7] + (h[7] - h[6]) x (nt - 7)."""
    out = []
    for pb in (0, 1):
        h = c16[8 * pb:8 * pb + 8]
        step = max(1, h[7] - h[6])
        out.append(h + [h[7] + step * (k - 7) for k in range(8, nmax + 1)])
    return out


def plan(nts, cost, g=G):
    """closed form of k3_gc_21 route() / stage_at: bA, bB = global step of each CTA start (+ end). nts: nt index per job in expert order."""
    res = []
    for pb in (0, 1):
        sc = 512 >> pb
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
        res.append(b)
    return res


def decode_jobs(rows, span=512, col=8):
    """nt class per job from the per-CTA class counts (r[8:16] A / r[16:24] B) walked in CTA order; None if inconsistent."""
    jobs, pos, cur = [], 0, None
    for c in range(G):
        n = rows[G + 1 + c][9 if col == 8 else 10]
        if n == 0:
            continue
        left = list(rows[c][col:col + 8])
        if sum(left) != n:
            return None
        while n > 0:
            if cur is None:
                cur = next(k for k in range(8) if left[k] > 0)
                jobs.append(cur)
                pos = 0
            take = min(n, span - pos, left[cur])
            if take <= 0:
                return None
            left[cur] -= take
            n -= take
            pos += take
            if pos == span:
                cur = None
    return jobs


def tsplit(tot, cap):
    """route() TA split: (sizes of the jobs of an expert with tot tokens)."""
    nj = (tot + cap - 1) // cap
    base = ((((tot + 7) >> 3) + nj - 1) // nj) << 3
    return [min(base, tot - k * base) for k in range(nj)]


def cls_of(sz):
    return min(7, (sz + 7) // 8 - 1)


PAIRS = {}
for _t in range(73, 145):
    _s = tsplit(_t, 72)
    if len(_s) == 2:
        PAIRS.setdefault((cls_of(_s[0]), cls_of(_s[1])), []).append(_t)


def nnls(X, y, iters=60):
    """Lawson-Hanson style active set on the normal equations (numpy only); columns with no data stay 0."""
    n = X.shape[1]
    keep = np.where(np.abs(X).sum(0) > 0)[0]
    A = X[:, keep]
    w = np.ones(len(keep), bool)
    for _ in range(iters):
        sol = np.zeros(len(keep))
        sol[w] = np.linalg.lstsq(A[:, w], y, rcond=None)[0]
        bad = w & (sol < 0)
        if not bad.any():
            break
        w &= ~bad
    out = np.zeros(n)
    out[keep] = np.maximum(sol, 0.0)
    return out


class Prm:
    """generative-model parameters of one T (us): regressed on the measured marks unless noted."""
    d = 0.0          # drain per segment (D_A / segment ~ 0.01)
    g, g0 = 0.73, 0.6                    # ae_model: per-segment gap of the marked kernel, B start overhead
    phiA, phiF, kf = 0.5, 0.5, 0.7       # ae_model: end-of-A flush, flush at the next first stage (+ kf x first stage)
    tau = 1.09                           # HBM share per 36 KB stage (stream floor)
    XL = 1.0
    xb0 = 0.0                            # first B stage no earlier than aend + xb0
    rbx = 0.0                            # extra read-back time of a wide resolver (a second dependent L2 round trip inside the stream)

    def __init__(s):
        s.pA, s.fA, s.pB = [0.0] * 8, [0.0] * 8, [0.0] * 8
        s.eA = {"full": (0.0, 0.0), "part": (0.0, 0.0), "res": (0.0, 0.0)}
        s.eB = {"full": (0.0, 0.0), "part": (0.0, 0.0)}
        s.hn = 1.0                       # end-work N of a wide job = hn x N (0.5: the split shares carry half-width partials, as in ship)

    def eAf(s, kind, n):
        c0, c1 = s.eA[kind]
        return c0 + c1 * n

    def eBf(s, kind, n):
        c0, c1 = s.eB[kind]
        return c0 + c1 * n

    def endN(s, jb):
        return jb.N * (s.hn if jb.wide else 1.0)

    def pa(s, n):
        return interp(dict(zip(NM, s.pA)), n)

    def pb(s, n):
        return interp(dict(zip(NM, s.pB)), n)


class Job:
    __slots__ = ("nt", "N", "pA", "fA", "pB", "wide")

    def __init__(s, nt, n, pa, fa, pb, wide=False):
        s.nt, s.N, s.pA, s.fA, s.pB, s.wide = nt, n, pa, fa, pb, wide


def ship_jobs(cls, prm):
    return [Job(k, NM[k], prm.pA[k], prm.fA[k], prm.pB[k]) for k in cls]


def load(path):
    """{(T, rs, layer): dict(recs = measured per-CTA records per rep, jobs = nt class list, ok = A and B decode agree)}."""
    raw = json.load(gzip.open(path, "rt"))
    out = {}
    for key, reps in raw.items():
        rs, ld, ly = key.split("|")
        T = dict(ae.LOADS)[ld]
        if T == 32:
            continue
        ja, jb = decode_jobs(reps[0], 512, 8), decode_jobs(reps[0], 256, 16)
        out[(T, int(rs), ly)] = dict(recs=[ae.parse(r) for r in reps], jobs=ja, ok=ja is not None and ja == jb, ld=ld)
    return out


def fit(items):
    """regress the generative parameters of one T. items: [(jobs cls, ctas)]; returns Prm (XL set by the caller)."""
    XS, YS, XF, YF, XE, YE, XB, YB, XEB, YEB = [], [], [], [], [], [], [], [], [], []
    for cls, ctas in items:
        res = ae.a_arm(ctas, ae.P(), False)["res"]
        for c, ct in ctas.items():
            xs, xf = np.zeros(8), np.zeros(8)
            xe = np.zeros(6)
            for i, (a, b) in enumerate(ct["segA"]):
                k, n = cls[a >> 9], b - a
                xs[k] += n - 1
                xf[k] += 1
                kind = 0 if n == 16 else 2
                xe[kind] += 1
                xe[kind + 1] += NM[k]
                if res.get((c, i)):
                    xe[4] += 1
                    xe[5] += NM[k]
            xb, xeb = np.zeros(8), np.zeros(4)
            for a, b in ct["segB"]:
                k = cls[a >> 8]
                xb[k] += b - a
                kind = 0 if b - a == 16 else 2
                xeb[kind] += 1
                xeb[kind + 1] += NM[k]
            XS.append(xs); YS.append(ct["S_A"]); XF.append(xf); YF.append(ct["F_A"])
            XE.append(xe); YE.append(ct["E_A"])
            XB.append(xb); YB.append(ct["F_B"] + ct["S_B"] - ct["wxB"])
            XEB.append(xeb); YEB.append(ct["E_B"])
    p = Prm()
    p.pA = list(nnls(np.array(XS), np.array(YS)))
    p.fA = list(nnls(np.array(XF), np.array(YF)))
    e = np.linalg.lstsq(np.array(XE), np.array(YE), rcond=None)[0]
    p.eA = {"full": (e[0], e[1]), "part": (e[2], e[3]), "res": (e[4], e[5])}
    p.pB = list(nnls(np.array(XB), np.array(YB)))
    e = np.linalg.lstsq(np.array(XEB), np.array(YEB), rcond=None)[0]
    p.eB = {"full": (e[0], e[1]), "part": (e[2], e[3])}
    return p


def simulate(jobs, cost, ctas, p, bchg=0.0, eps=None):
    """generative forward timeline of one run: ({c: (aend, bend)}, info). jobs: Job list in expert order; ctas: measured records (fc per CTA);
    eps = (epsA, epsB) per-CTA pace multipliers (noise); bchg = charge per CTA at every engine switch (w72 <-> wide job)."""
    bA, bB = plan([jb.nt for jb in jobs], cost)
    act = sorted(ctas)
    epA, epB = eps if eps else ({}, {})
    segA = {c: ae.segs_a(bA[c], bA[c + 1]) for c in act}
    segB = {c: ae.segs_b(bB[c], bB[c + 1]) for c in act}
    sh = {}
    for c in act:
        for i, (a, b) in enumerate(segA[c]):
            sh.setdefault(a >> 4, []).append((c, i, b - a == 16))
    res = {}
    for lst in sh.values():
        if len(lst) > 1:
            c, i, _ = max(lst, key=lambda x: (x[1], x[0]))
            res[(c, i)] = True

    def fwd(c):
        fc, K, out, mode = ctas[c]["fc"], 0, [], False
        t, e = fc, epA.get(c, 1.0)
        for i, (a, b) in enumerate(segA[c]):
            jb, n = jobs[a >> 9], b - a
            K += n
            if jb.wide != mode:
                t += bchg
                mode = jb.wide
            ne = p.endN(jb)
            te = max(t + (jb.pA if i == 0 else jb.fA) * e + (n - 1) * jb.pA * e, fc + (K - 1) * p.tau)
            if n == 16:
                pre, rsv = p.eAf("full", ne), 0.0
            else:
                pre, rsv = p.eAf("part", ne), ((p.eAf("res", ne) + (p.rbx if jb.wide else 0.0)) if res.get((c, i)) else 0.0)
            arr = te + p.d + pre
            t4 = arr + rsv
            out.append((t, te, arr, t4, jb.fA))
            t = t4 + p.g
        return out, mode

    for _ in range(4):
        tl = {c: fwd(c)[0] for c in act}
        new = {}
        for lst in sh.values():
            if len(lst) > 1:
                c, i, _ = max(lst, key=lambda x: (tl[x[0]][x[1]][2], x[1], x[0]))
                new[(c, i)] = True
        if new == res:
            break
        res = new
    fw = {c: fwd(c) for c in act}
    tl = {c: fw[c][0] for c in act}
    aend, rel = {}, {}
    for c in act:
        s_ = tl[c]
        aend[c] = s_[-1][3] + p.phiA if s_ else ctas[c]["fc"]
        rel[c] = [(s_[i + 1][0] + p.kf * s_[i + 1][4] + p.phiF) if i < len(s_) - 1 else aend[c] for i in range(len(s_))]
    R = {}
    for u, lst in sh.items():
        r = [(c, i) for c, i, f in lst if res.get((c, i)) or f] or [(c, i) for c, i, f in lst]
        R[u] = max(rel[c][i] for c, i in r)
    end, wsum = {}, {}
    for c in act:
        fc, e = ctas[c]["fc"], epB.get(c, 1.0)
        na = sum(b - a for a, b in segA[c])
        t, K, ws, mode = aend[c] + p.g0, na, 0.0, fw[c][1]
        for a, b in segB[c]:
            jb = jobs[a >> 8]
            if jb.wide != mode:
                t += bchg
                mode = jb.wide
            for gi in range(a, b):
                j, s_ = gi >> 8, gi & 15
                K += 1
                r = max(R.get(32 * j + 2 * s_, 0.0), R.get(32 * j + 2 * s_ + 1, 0.0)) + p.XL
                stt = max(t, r, fc + (K - 1) * p.tau, aend[c] + p.xb0 if K == na + 1 else 0.0)
                ws += max(0.0, r - t)
                t = stt + jobs[j].pB * e
            t += p.d + p.eBf("full" if b - a == 16 else "part", p.endN(jb)) + p.g
        end[c] = t
        wsum[c] = ws
    return {c: (aend[c], end[c]) for c in act}, dict(res=res, bA=bA, bB=bB, wait=wsum, R=R, nsh={u: len(l) for u, l in sh.items()})


# (cold - warm) / cold of the inc stage (R-xc-0): the part of a stage that sits on the HBM stream at full-GPU contention
GAP = {8: 0.102, 16: 0.0765, 32: 0.0655, 56: 0.049, 64: 0.051}
SIG = {256: (0.04, 0.04), 128: (0.05, 0.02)}   # per-CTA pace noise (persistent sd, extra B sd), calibrated to the measured A / B end spread (--calib=1)
PAD3 = 1040 * 0.75 / 768                       # Astra 3-WG x 192-row tile: 65 vs 48 row tiles x 16 stages at 0.75 of a 4-WG stage (optimistic, unmeasured)


# R-rev-1 resident stage C(N) = 0.82 + 0.0096 N (handoff_vmx.md: 0.90 / 0.98 / 1.12 / 1.44 at N 8 / 16 / 32 / 64) and streaming stage S (1.23 / 1.34 / 1.60 / 2.02):
# only the N-proportional part of C scales with the rows of a stage; handshakes, wgmma latency and loads are per stage, not per row.
S_TAB = {8: 1.23, 16: 1.34, 32: 1.60, 64: 2.02}
ST3 = 1040 / 768                               # 3-WG x 192-row tiles: 65 vs 48 row tiles x 16 stages


def s3(n):
    """stage time of a 192-row (3-WG) stage / a 256-row (4-WG) stage when 0.75 of the N-proportional part of C scales."""
    return 1.0 - 0.25 * 0.00964 * n / interp(S_TAB, n)


def wide_job(tot, p, fx, rr=None):
    """one job for an expert with tot tokens (73..128): PP stage / (the two w72 stages at the shipped split sizes) x the shipped stage paces."""
    b0, b1 = tsplit(tot, 72)
    r = interp(PP, tot) / (interp(W72, b0) + interp(W72, b1)) if rr is None else rr
    return Job(((tot + 7) >> 3) - 1, tot, r * (p.pa(b0) + p.pa(b1)), p.fA[7] + fx, r * (p.pb(b0) + p.pb(b1)), True), r


def merge_hot(jobs, p, fx, cap=128, rr=None):
    """merge every consistent hot pair (class pair of the TA split of 73..cap tokens), left to right; returns (jobs, [(index, tot, r)])."""
    out, i, mg = [], 0, []
    while i < len(jobs):
        if i + 1 < len(jobs) and not jobs[i].wide:
            key = (jobs[i].nt, jobs[i + 1].nt)
            if key in PAIRS:
                tots = PAIRS[key]
                tot = int(round(sum(tots) / len(tots)))
                if tot <= cap:
                    jb, r = wide_job(tot, p, fx, rr)
                    out.append(jb)
                    mg.append((i, tot, r))
                    i += 2
                    continue
        out.append(jobs[i])
        i += 1
    return out, mg


def e4_cost(T, p, jobs, lam):
    """planner cost table with entries for the wide classes: lam x (cost / pace of the widest shipped class) x wide stage pace."""
    cA, cB = ext_cost(COST[T])
    ra, rb = cA[7] / p.pA[7], cB[7] / p.pB[7]
    for jb in jobs:
        if jb.wide:
            cA[jb.nt], cB[jb.nt] = int(round(lam * ra * jb.pA)), int(round(lam * rb * jb.pB))
    return [cA, cB]


def scaled(jobs, f_ship=1.0, f_wide=1.0, relief=0.0, ppmin=None, fmin=0, eng3=False):
    """copy of the job list: shipped jobs with N >= fmin x f_ship, all shipped jobs x (1 - GAP x relief) [x pp / w72 if N >= ppmin], wide jobs x f_wide."""
    out = []
    for jb in jobs:
        if jb.wide:
            k = f_wide * (s3(jb.N) * ST3 if eng3 else 1.0)
            out.append(Job(jb.nt, jb.N, jb.pA * k, jb.fA, jb.pB * k, True))
            continue
        k = (f_ship if jb.N >= fmin else 1.0) * (1.0 - interp(GAP, jb.N) * relief)
        if ppmin is not None and jb.N >= ppmin:
            k *= interp(PP, jb.N) / interp(W72, jb.N)
        if eng3:
            k *= s3(jb.N) * ST3
        out.append(Job(jb.nt, jb.N, jb.pA * k, jb.fA, jb.pB * k, False))
    return out


def with_p(p, **kw):
    q = Prm.__new__(Prm)
    q.__dict__.update(p.__dict__)
    q.__dict__.update(kw)
    return q


class Arm:
    def __init__(s, name, bchg=0.0, fx=0.0, hn=0.5, relief=0.0, lam=1.0, f_ship=1.0, f_wide=1.0, ppmin=None, fmin=0, rr=None, eng3=False, rbx=0.0):
        s.__dict__.update(locals())


ARMS = [Arm("g", 0.0, 0.0, 0.5, 0.0),                  # stage-rate gain + removed passes, ship end work per share
        Arm("hbm", 0.0, 0.0, 0.5, 1.0),                # + HBM relief of the removed bytes (upper bound)
        Arm("end", 0.0, 0.0, 1.0, 1.0),                # + split shares / read-back at the wide N (2x the bytes per chain)
        Arm("fst", 0.0, 0.3, 1.0, 1.0),                # + first-stage extra of the wide engine
        Arm("e4", 1.0, 0.3, 1.0, 1.0),                 # + one boundary charge per engine switch  (= E4 central, oracle engine)
        Arm("lo", 1.0, 0.3, 1.0, 1.0, 0.9), Arm("hi", 1.0, 0.3, 1.0, 1.0, 1.1),              # wide cost-table error -10 % / +10 %
        Arm("pes", 2.0, 0.6, 1.0, 0.0),                # every charge doubled, no HBM credit
        Arm("p3a", 1.0, 0.3, 1.0, 1.0, 1.0, PAD3, PAD3),                     # 3-WG x 160: +1.56 % row padding on every stage
        Arm("p3c", 1.0, 0.3, 1.0, 1.0, 1.0, PAD3, PAD3, None, 24),           # ... only on the consumer-bound stages (N >= 24; the N 8 / 16 stream hides it)
        Arm("ppx", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 24),                   # PP engine for every job, pp / w72 stage ratio at N >= 24 (N 8 / 16 hidden by the stream)
        Arm("ppf", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 0),
        Arm("p3f", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, None, 0, None, True),     # 3-WG x 160 with the fixed part of the stage kept (S3 / S x 1.354 stages)
        Arm("r85", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, None, 0, 0.85), Arm("r90", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, None, 0, 0.90),   # wide stage / two w72 stages forced to 0.85 / 0.90
        Arm("rb3", 1.0, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, None, 0, None, False, 3.5),                      # +3.5 us per wide resolver (one more dependent L2 round trip, R-vmsp 2.5-4.5)
        Arm("nul", 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 1.0, None, 0, 1.0)]       # null: merged job at the SUM of the two shipped paces (no rate gain): must be ~0                    # PP engine for every job, probe ratio at face value on every N


def run_T(T, runs, p, nd, seed, arms=ARMS):
    """per routing: per (rep, draw) max end of each arm - ship (CRN noise) and the B-end spread (max - median); {(rs, layer): dict}."""
    rng = np.random.default_rng(seed)
    sp, sb = SIG[T]
    out = {}
    for k, v in sorted(runs.items()):
        if k[0] != T:
            continue
        sj = ship_jobs(v["jobs"], p)
        ship_cost = ext_cost(COST[T])
        e4j, mg = merge_hot(sj, p, 0.0)
        res = dict(mg=mg, d={a.name: [] for a in arms}, sp={a.name: [] for a in arms}, ship=[], sps=[], jobs=sj,
                   meas=[max(ct["bend"] for ct in ctas.values()) for ctas in v["recs"]])
        out[k[1:]] = res
        for ctas in v["recs"]:
            cs = sorted(ctas)
            for _ in range(nd):
                ea = {c: float(np.exp(rng.normal(0, sp))) for c in cs}
                eb = {c: ea[c] * float(np.exp(rng.normal(0, sb))) for c in cs}
                r0, _i = simulate(sj, ship_cost, ctas, p, eps=(ea, eb))
                b0 = np.array([b for a, b in r0.values()])
                res["ship"].append(b0.max())
                res["sps"].append(b0.max() - np.median(b0))
                for a in arms:
                    if not mg:
                        res["d"][a.name].append(0.0)
                        res["sp"][a.name].append(res["sps"][-1])
                        continue
                    jb, _m = merge_hot(sj, p, a.fx, 128, a.rr)
                    jb = scaled(jb, a.f_ship, a.f_wide, a.relief * len(mg) / len(sj), a.ppmin, a.fmin, a.eng3)
                    r1, _i = simulate(jb, e4_cost(T, p, jb, a.lam), ctas, with_p(p, hn=a.hn, rbx=a.rbx), bchg=a.bchg, eps=(ea, eb))
                    b1 = np.array([b for a_, b in r1.values()])
                    res["d"][a.name].append(b1.max() - b0.max())
                    res["sp"][a.name].append(b1.max() - np.median(b1))
    return out


def tier_gap(T, runs):
    """measured B end minus the run median, mean over CTAs by dominant phase-A class: (L = nt 0, W = nt 4..7)."""
    lw = {"L": [], "W": []}
    for k, v in runs.items():
        if k[0] != T:
            continue
        for ctas in v["recs"]:
            med = float(np.median([ct["bend"] for ct in ctas.values()]))
            for c, ct in ctas.items():
                cnt = np.zeros(8)
                for a_, b_ in ct["segA"]:
                    cnt[v["jobs"][a_ >> 9]] += b_ - a_
                kd = int(np.argmax(cnt))
                if kd == 0:
                    lw["L"].append(ct["bend"] - med)
                elif kd >= 4:
                    lw["W"].append(ct["bend"] - med)
    return float(np.mean(lw["L"])), float(np.mean(lw["W"]))


def per_seed(out, f):
    """f(result of one routing) -> value; mean over the layers of a seed, then median over seeds; also the per-seed list."""
    by = {}
    for (rs, ly), r in out.items():
        by.setdefault(rs, []).append(f(r))
    sv = {rs: float(np.mean(v)) for rs, v in by.items()}
    return float(np.median(list(sv.values()))), sv


def chains(T, p, runs, rs_ly):
    """hot-job split units / read-back chains of the A phase (no noise, rep 0): ship vs E4. returns dict."""
    v = runs[(T,) + rs_ly]
    sj = ship_jobs(v["jobs"], p)
    ej, mg = merge_hot(sj, p, 0.3)
    ctas = v["recs"][0]
    _r0, i0 = simulate(sj, ext_cost(COST[T]), ctas, p)
    _r1, i1 = simulate(ej, e4_cost(T, p, ej, 1.0), ctas, with_p(p, hn=1.0), bchg=1.0)
    su, eu, sb, eb, rs_, re_ = 0, 0, 0.0, 0.0, [], []
    for n, (i, tot, _r) in enumerate(mg):
        je = i - n
        for u in range(32 * i, 32 * (i + 2)):
            if i0["nsh"].get(u, 1) > 1:
                su += 1
                sb += sj[u >> 5].N / 1000.0
        for u in range(32 * je, 32 * (je + 1)):
            if i1["nsh"].get(u, 1) > 1:
                eu += 1
                eb += ej[u >> 5].N / 1000.0
        rs_.append(max(i0["R"][u] for u in range(32 * i, 32 * (i + 2))))
        re_.append(max(i1["R"][u] for u in range(32 * je, 32 * (je + 1))))
    return dict(su=su, eu=eu, sb=sb, eb=eb, rs=rs_, re=re_, n=len(mg))


NX2 = 4.2          # R-m32e-14 (notes/study/moe_t32plus.md s11): 2-slot x ring (w2j80) +1.0 vs w2j72 -3.2 us at T = 256
LINES = []


def out_(x):
    LINES.append(x)
    print(x)


def setup(T, runs):
    items = [(v["jobs"], c) for k, v in runs.items() if k[0] == T for c in v["recs"]]
    p = fit(items)
    p.XL = ae.calib_xl({(k[1], v["ld"]): v["recs"] for k, v in runs.items() if k[0] == T})[T]
    p.xb0 = p.XL
    return p


def calib(runs, T, nd=3):
    """--calib=1: grid of the noise sds vs the measured B end spread and max end (one line per cell)."""
    import itertools
    p = setup(T, runs)
    for sa, sb in itertools.product((0.03, 0.04, 0.05), (0.0, 0.02, 0.04)):
        SIG[T] = (sa, sb)
        o = run_T(T, runs, p, nd, 7, [])
        m = [np.mean(r["ship"]) for r in o.values()]
        s_ = [np.mean(r["sps"]) for r in o.values()]
        print(f"T{T} sd {sa} {sb}: gen max {np.mean(m):.1f} B spread {np.mean(s_):.1f} | measured max {np.mean([np.mean(r['meas']) for r in o.values()]):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default=str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    ap.add_argument("--nd", type=int, default=6, help="noise draws per rep")
    ap.add_argument("--calib", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11, help="noise rng seed")
    a = ap.parse_args()
    runs = load(a.marks)
    if a.calib:
        for T in (256, 128):
            calib(runs, T)
        return
    bad = [k for k, v in runs.items() if not v["ok"]]
    pexact = 0
    for k, v in runs.items():
        bA, bB = plan(list(v["jobs"]), ext_cost(COST[k[0]]))
        ctas = v["recs"][0]
        cA = np.concatenate([[0], np.cumsum([ctas[c]["nA"] for c in range(G)])])
        cB = np.concatenate([[0], np.cumsum([ctas[c]["nB"] for c in range(G)])])
        pexact += int(np.array_equal(cA, bA) and np.array_equal(cB, bB))
    p256, p128 = setup(256, runs), setup(128, runs)
    res = {256: run_T(256, runs, p256, a.nd, a.seed), 128: run_T(128, runs, p128, 1, 13)}
    fid = {}
    for T, p in ((256, p256), (128, p128)):
        e_a, e_b = [], []
        for k, v in runs.items():
            if k[0] != T:
                continue
            sj = ship_jobs(v["jobs"], p)
            for ctas in v["recs"]:
                r, _i = simulate(sj, ext_cost(COST[T]), ctas, p)
                e_a.append(np.mean([abs(r[c][0] - ctas[c]["aend"]) for c in ctas]))
                e_b.append(np.mean([abs(r[c][1] - ctas[c]["bend"]) for c in ctas]))
        fid[T] = (np.mean(e_a), np.mean(e_b))
    rr = [wide_job(t, p256, 0.0)[1] for t in (76, 84, 92, 100, 108)]
    o2 = res[256]
    gen = np.mean([np.mean(r["ship"]) for r in o2.values()])
    mea = np.mean([np.mean(r["meas"]) for r in o2.values()])
    out_("R-e4-0 ($0 CPU): E4 = each hot expert ONE wide job (<=128 tok) vs shipped <=72-tok passes; max-over-CTAs vm_main end, us (marks +9-11 % slow: deltas only)")
    out_(f"check: planner closed form exact on {pexact}/{len(runs)} routings (A = B decode {len(runs) - len(bad)}/{len(runs)}); gen-ship vs measured per-CTA MAE A {fid[256][0]:.1f} / B {fid[256][1]:.1f} us (T256), "
         f"max end gen {gen:.0f} vs meas {mea:.0f} (noise sd calibrated to the B spread); wide stage / two w72 stages = {min(rr):.2f}..{max(rr):.2f} (PP N 76..108, R-xc-0 cold)")
    out_("T    seed  | wide jobs sl/full | ship max | E4 max (d)   | gain only | +HBM | +split/rb | +first | +bound | all x2 no HBM")
    sm = {}
    for T in (256, 128):
        o = res[T]
        for rs in sorted({k[0] for k in o}):
            ly = {k[1]: o[k] for k in o if k[0] == rs}
            nw = "/".join(str(len(ly[x]["mg"])) for x in ("sliding", "full"))
            ship = np.mean([np.mean(ly[x]["meas"]) for x in ly])
            d = {n: np.mean([np.mean(ly[x]["d"][n]) for x in ly]) for n in ly["full"]["d"].keys()}
            out_(f"{T:<4d} {rs % 100000:>6d} | {nw:>17s} | {ship:8.1f} | {ship + d['e4']:7.1f} ({d['e4']:+5.1f}) | {d['g']:+9.1f} | {d['hbm'] - d['g']:+4.1f} | {d['end'] - d['hbm']:+9.1f} | {d['fst'] - d['end']:+6.1f} | {d['e4'] - d['fst']:+6.1f} | {d['pes']:+5.1f}")
        sm[T] = {n: per_seed(o, lambda r, n=n: float(np.mean(r["d"][n])))[0] for n in o[next(iter(o))]["d"]}
        shipm = per_seed(o, lambda r: float(np.mean(r["meas"])))[0]
        sm[T]["ship"] = shipm
        sv = per_seed(o, lambda r: float(np.mean(r["d"]["e4"])))[1]
        out_(f"T={T} median over seeds: ship {shipm:.1f} -> E4 {shipm + sm[T]['e4']:.1f} (d {sm[T]['e4']:+.1f}, seeds {min(sv.values()):+.1f}..{max(sv.values()):+.1f}); passes removed {np.mean([len(r['mg']) for r in o.values()]):.2f} per routing")
    d = sm[256]
    spd = per_seed(o2, lambda r: float(np.mean(r["sp"]["e4"]) - np.mean(r["sps"])))[0]
    gl, gw = tier_gap(256, runs)
    npass = float(np.mean([len(r["mg"]) for r in o2.values()]))
    traffic = npass * 28311552 / (G * 36864 / 1.09e-6) * 1e6          # removed weight bytes at the 1.09 us / 36 KB stage share of 132 CTAs
    out_(f"T256 d (median over seeds, cumulative): rate+removal {d['g']:+.1f} | HBM relief {d['hbm'] - d['g']:+.1f} | split/read-back {d['end'] - d['hbm']:+.1f} | first stage {d['fst'] - d['end']:+.1f} | "
         f"boundary {d['e4'] - d['fst']:+.1f} = {d['e4']:+.1f} (the {npass:.2f} removed passes x 28.3 MB = {traffic:.1f} us of traffic at 4.46 TB/s); end-spread (max - median) {spd:+.1f}. Charges hide: hot (W) CTAs end {gw:+.0f} vs the L-tier {gl:+.0f} us from the median (measured), the max is an L-tier CTA")
    ch = [chains(256, p256, runs, k) for k in sorted(k[1:] for k in runs if k[0] == 256)]
    ch = [c for c in ch if c["n"]]
    out_(f"split/read-back: hot-job split units (A) ship {np.mean([c['su'] / c['n'] for c in ch]):.1f} -> E4 {np.mean([c['eu'] / c['n'] for c in ch]):.1f} per merged pair (one resolver read-back each, chain width "
         f"{np.mean([c['sb'] / max(1, c['su']) for c in ch]) * 1000:.0f} -> {np.mean([c['eb'] / max(1, c['eu']) for c in ch]) * 1000:.0f} cols = 2x bytes; summed chain columns {np.mean([c['sb'] / c['n'] for c in ch]) * 1000:.0f} -> {np.mean([c['eb'] / c['n'] for c in ch]) * 1000:.0f}); latest hot doneA release "
         f"{np.mean([x for c in ch for x in c['rs']]):.1f} -> {np.mean([x for c in ch for x in c['re']]):.1f} us (model)")
    slope = (d["nul"] - d["e4"]) / (1.0 - float(np.mean(rr)))        # us per unit of the wide/2-pass stage ratio between the null (r = 1) and E4
    rstar = 1.0 - (d["nul"] + 5.0) / slope                           # ratio at which d = -5
    out_(f"envelope T256 d: wide cost table -10 % {d['lo']:+.1f} / +10 % {d['hi']:+.1f}; charges x2 and no HBM credit {d['pes']:+.1f}; stage ratio forced to 0.85 {d['r85']:+.1f} / 0.90 {d['r90']:+.1f} / 1.00 (null) {d['nul']:+.1f}; "
         f"+1 read-back round trip per wide resolver (+3.5 us) {d['rb3']:+.1f}; -5 needs a ratio <= {rstar:.2f} (PP table gives {np.mean(rr):.2f}); history: the s10 hot-wide design predicted -10..-15 and measured -3.2 (w2j72)")
    out_(f"closing plans, T256 d: 3-WG x 160 with Astra's 0.75 x stage {d['p3a']:+.1f} (padding on all stages) / {d['p3c']:+.1f} (consumer-bound only) but keeping the N-independent part of the measured stage "
         f"(job time x{s3(8) * ST3:.2f} at N 8 .. x{s3(68) * ST3:.2f} at N 68: 1.354 x stages, S3/S 0.92-0.98) {d['p3f']:+.1f}; PP engine (232 regs, 384 thr) for all jobs {d['ppf']:+.1f} at the probe pp / w72 "
         f"({d['ppx']:+.1f} only if the stream hid every N<24 penalty, against the measured +12.7 % at N 8); 2-slot x ring +{NX2:.1f} (R-m32e-14) not added")
    acc = 128 // 2
    out_(f"registers 640 thr x 112 (w72): N=128 acc = 64 x 128 / 128 = {acc} fp32; shipped consumer spill-free to N 80 (acc 40), spills 85 / 206 STL+LDL at N 96 / 112 => non-acc 65..72, need {acc + 65}..{acc + 72} > 112; "
         f"pool 640 x 96 = 61,440 caps consumers at (61,440 - 128 x 24) / 512 = 114 -> 112 (CREG 120 TRY_ALLOC spin, 5th WG C7600): does NOT close; "
         f"the 512-thread no-producer variant (s10 W512, cap 128) is short by {acc + 65 - 128}..{acc + 72 - 128} regs")
    out_("registers 3-WG x 160: 64 acc + 32 A (k128) + 16 codes/scales + ~24 addr/barrier = 136 <= 160 (24 spare); pool 384 x 160 + 128 x 32 = 65,536 = launch pool (0 spare; per SMSP 3 x 32 x 160 + 32 x 32 = 16,384); "
         "smem 3 x 27 + 2 x 64 = 209 of 227 KiB: closes on paper only (unbuilt, 4/3 row tiles); PP 232 regs compiles at N 8..128 but is a 384-thread CTA (cannot share the w72 launch: 2 x 128 x 232 + 3 x 128 x 24 > 61,440)")
    u3 = s3(8) * ST3                                                  # job-time factor of the critical small-N jobs under the measured fixed part
    fstar = (PAD3 + (-5.0 - d["p3a"]) * (u3 - PAD3) / (d["p3f"] - d["p3a"])) / ST3
    best = min(d["p3f"], d["ppf"])
    verdict = "GO" if (d["e4"] <= -5.0 and best <= -5.0) else "KILL"
    out_(f"VERDICT {verdict}: oracle E4 {d['e4']:+.1f} us at T256 (margin {-5.0 - d['e4']:+.1f} vs the -5 gate; T128 {sm[128]['e4']:+.1f}: no expert > 72 tokens in 6/6 routings) needs the measured PP rate inside the w72 CTA, which does not close in registers; "
         f"the plans that close land at {d['p3f']:+.1f} (3-WG) / {d['ppf']:+.1f} (PP for all) = margin {-5.0 - d['p3f']:+.1f} / {-5.0 - d['ppf']:+.1f}; "
         f"biggest charge = that engine, not the three modeled E4 charges (split/read-back, first stage, boundary: total {d['e4'] - d['hbm']:+.1f}); 3-WG holds -5 only if a 192-row stage costs <= {fstar:.2f} of a 256-row stage (measured S3/S 0.92-0.98)")
    return LINES


if __name__ == "__main__":
    main()

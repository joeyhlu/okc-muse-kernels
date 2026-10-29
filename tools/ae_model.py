"""ae_model.py: $0 CPU model of an ASYNC EPILOGUE in the VML MoE (T >= 32) of the decode_layer kernel (lab R-ae-0).

    python tools/ae_model.py [--marks=box_out/lat32/m_ship/marks_raw.json.gz] [--h0=0.1] [--h1=0.008] [--m=1.0] [--tau=1.09] [--sens=1]
    python tools/ae_model.py --scan=1,3,4 [--set=escale=0.5,h0=1.0,sel=1]     (one line per worker speed m: AE1 / AE0 max-end deltas)

Idea: all 4 consumer warpgroups run drain + epilogue (SwiGLU / h stores / split red.add + counter + read-back / doneA release) at each
segment end, ~1.6-2.2 us (A) / 0.7-0.8 us (B), while no weight slot frees. Async: the consumers hand their accumulators to smem (cost h)
and start the next segment; a spare warp group runs the epilogue (m x today's duration, one worker, one hand-off slot) one segment behind
and releases doneA from there. Question (R-slc-0 died here): does the MAX over CTAs of the vm_main end move, not only the per-CTA busy time?

Input: per-CTA marks of the gated in-layer ship VML (tools/box/lat32_marks.py -> box_out/lat32/m_ship/marks_raw.json.gz: 3 router seeds x
{decode 32, verify 16, verify 32} = T 32 / 128 / 256 x {full, sliding} x 5 reps). The marks perturb the layer +9-11 %, so only differences count.
They hold per-CTA SUMS (first stage / steady / drain / end work per phase, segment + stage counts, waits), not per-segment times, so:
  * plan: CTA stage ranges = cumulative sum of the per-CTA stage counts (A 512 steps / job, B 256); segments cut at 16-aligned steps, A in
    split_point rotation order (flags 193) -> reproduces the measured segment counts of all 11880 CTA-reps (printed as the plan check).
  * per CTA: first-stage, steady pace, drain, B work per stage from the measured sums; end work per segment by kind (full unit / split share /
    resolver = last arriver, found by arrival time) from the CTA's measured split-atomic, read-back and epilogue sums; timeline anchored at the
    measured phase-A end; the unmeasured per-segment gap of the marked kernel (~0.73 us) kept in both arms.
  * doneA releases (ship = DEFER: flush at the next segment's first stage, last unit at the A end) -> slice readiness
    R(j, s) = max(release of A units 2s, 2s + 1); B stage k starts at max(consumer free, R + XL, stream floor); XL = measured base x wait.
Arms: E0 = ship model (checked against the measured per-CTA ends). AE1 = consumer resumes after drain + h, epilogue + release on the worker,
readiness recomputed from the async A timelines (the design as specified). AE0 / AE00 = same consumers, but the releases of the OTHER CTAs / of ALL
CTAs frozen at the ship model's times ("release times as measured"; stale-release lower bound: every CTA runs ahead of releases that did not move).
Stream floor: stage K of a CTA starts no earlier than fc + (K - 1) x tau (tau = 1.09 us, its HBM share). The CTA ends when its worker is done.
"""
import argparse
import gzip
import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = 132
LOADS = (("decode 32", 32), ("verify 16", 128), ("verify 32", 256))
NMEAN = [8, 16, 24, 32, 40, 48, 56, 68]   # token columns N of an nt class (class 8 = nt 8 or 9)


def segs_a(g0, g1):
    """phase-A segments (gs, ge) in execution order: split_point rotation (flags bit 0) runs the last partial unit first."""
    if g1 <= g0:
        return []
    ue = ((g1 - 1) >> 4) << 4
    sp = ue if ((g1 & 15) != 0 and ue > g0) else g0
    out = []
    for lo, hi in (((sp, g1), (g0, sp)) if sp > g0 else ((g0, g1),)):
        gs = lo
        while gs < hi:
            e = min(hi, ((gs >> 4) + 1) << 4)
            out.append((gs, e))
            gs = e
    return out


def segs_b(g0, g1):
    out, gs = [], g0
    while gs < g1:
        e = min(g1, ((gs >> 4) + 1) << 4)
        out.append((gs, e))
        gs = e
    return out


def parse(rows):
    """rows (265 x 32 int64 marks) -> {c: record} in us relative to the earliest CTA entry; plan segments attached."""
    act = [c for c in range(G) if rows[c][0] > 0]
    t0 = min(rows[c][0] for c in act)
    u = lambda x: (x - t0) / 1e3  # noqa: E731
    nA = [rows[G + 1 + c][9] for c in range(G)]
    nB = [rows[G + 1 + c][10] for c in range(G)]
    bA, bB = [0], [0]
    for c in range(G):
        bA.append(bA[-1] + nA[c])
        bB.append(bB[-1] + nB[c])
    out = {}
    for c in act:
        r, e = rows[c], rows[G + 1 + c]
        ca = r[8:16]
        tot = max(1, sum(ca))
        out[c] = dict(
            c=c, fc=u(e[1]), aend=u(r[6]), bend=u(max(r[1:5])), nA=e[9], nB=e[10], sA=e[15], sB=e[21],
            F_A=e[11] / 1e3, S_A=e[12] / 1e3, D_A=e[13] / 1e3, E_A=e[14] / 1e3,
            F_B=e[17] / 1e3, S_B=e[18] / 1e3, D_B=e[19] / 1e3, E_B=e[20] / 1e3,
            at=e[23] / 1e3, rb=e[24] / 1e3, ep=e[25] / 1e3, wxB=(e[4] - e[5]) / 1e3, wB=r[24] / 1e3,
            Nm=sum(ca[k] * NMEAN[k] for k in range(8)) / tot,
            segA=segs_a(bA[c], bA[c + 1]), segB=segs_b(bB[c], bB[c + 1]))
    return out


def fit_b(runs):
    """per-T regression of the measured B end-work sums: b_full x nfull + b_part x npart (us), clamped >= 0.02."""
    X, Y = [], []
    for ctas in runs:
        for ct in ctas.values():
            nf = sum(1 for a, b in ct["segB"] if b - a == 16)
            X.append([nf, len(ct["segB"]) - nf])
            Y.append(ct["E_B"])
    return np.maximum(np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0], 0.02)


class P:
    """model parameters (us)."""
    def __init__(s, **kw):
        s.h0, s.h1 = 0.10, 0.008       # hand-off (stmatrix of N/2 fp32 per thread + WG barrier): h0 + h1 x N
        s.m = 1.0                       # worker epilogue duration / today's end work
        s.tau = 1.09                    # HBM share per 36 KB stage (floor on the consumption rate)
        s.g = 0.73                      # unmeasured per-segment gap of the marked kernel (kept in both arms)
        s.g0 = 0.6                      # B start overhead
        s.phiA = 0.5                    # end-of-A pend_flush (ship)
        s.phiF = 0.5                    # pend_flush at the next first stage (ship)
        s.phiR = 0.4                    # fence + release on the worker
        s.XL = 1.0                      # release -> x landed (consumer side)
        s.kf = 0.7                      # flush point inside the next first stage
        s.escale = 1.0                  # scale on all end work (mark-overhead sensitivity)
        s.of = 0.3                      # per-segment end-work overhead of full units (marks)
        s.sel = 0                       # 1: hand off only segments whose end work exceeds h + selm (the rest keep the inline epilogue)
        s.selm = 0.3
        for k, v in kw.items():
            setattr(s, k, v)

    def h(s, ct):
        return s.h0 + s.h1 * ct["Nm"]


def alloc_a(ct, res, p):
    """per-segment end work (pre, resolve) in us: pre = red.add + counter (split) or the whole epilogue (full unit); resolve = read-back +
    epilogue of the last arriver. Sums to the CTA's measured end work."""
    segs = ct["segA"]
    ns = len(segs)
    if ns == 0:
        return []
    isres = [bool(res.get((ct["c"], i))) for i in range(ns)]
    full = [b - a == 16 for a, b in segs]
    nf, npart, nres = sum(full), ns - sum(full), sum(isres)
    other = max(0.0, ct["E_A"] - ct["at"] - ct["rb"] - ct["ep"])
    of = min(p.of, other / ns)
    op = (other - of * ns)
    pe = ct["ep"] / (nf + nres) if (nf + nres) else 0.0
    rb = ct["rb"] / nres if nres else 0.0
    out = []
    for i in range(ns):
        if full[i]:
            out.append([pe + of + (op / nf if npart == 0 else 0.0), 0.0])
        else:
            x = (ct["at"] + op) / npart + of + ((ct["rb"] / npart) if not nres else 0.0)
            out.append([x, (rb + pe) if isres[i] else 0.0])
    if nf == 0 and nres == 0 and npart:           # epilogue measured but no resolver found: spread over the split shares
        for o in out:
            o[0] += ct["ep"] / npart
    tot = sum(x + y for x, y in out)
    sc = ct["E_A"] * p.escale / tot if tot > 0 else 1.0
    return [(x * sc, y * sc) for x, y in out]


def base_a(ct, E, p):
    """ship phase-A timeline anchored at the measured A end: (start, steady end, drain end, end-work end) per segment + f, s, d."""
    segs = ct["segA"]
    ns = len(segs)
    if ns == 0:
        return [], [], [], [], 0.0, 0.0, 0.0
    f = ct["F_A"] / ns
    nst = ct["nA"] - ns
    s = ct["S_A"] / nst if nst > 0 else 0.0
    d = ct["D_A"] / ns
    st_, ts, td, t4 = [0.0] * ns, [0.0] * ns, [0.0] * ns, [0.0] * ns
    t = ct["aend"] - p.phiA
    for i in range(ns - 1, -1, -1):
        n = segs[i][1] - segs[i][0]
        t4[i] = t
        td[i] = t - (E[i][0] + E[i][1])
        ts[i] = td[i] - d
        st_[i] = ts[i] - (n - 1) * s - f
        t = st_[i] - p.g
    return st_, ts, td, t4, f, s, d


def units_of(ctas):
    """unit -> [(c, i, full)] shares of phase A."""
    sh = {}
    for c, ct in ctas.items():
        for i, (a, b) in enumerate(ct["segA"]):
            sh.setdefault(a >> 4, []).append((c, i, b - a == 16))
    return sh


def a_arm(ctas, p, asyn):
    """phase-A timelines + unit releases R. Ship (asyn False): resolver = last arriver, iterated. Async: consumer forward from the ship start,
    one worker per CTA, one hand-off slot, releases from the worker. Returns dict(R, own (resolver CTA), thA (consumer A end), we (worker free))."""
    sh = units_of(ctas)
    res = {}
    for u, lst in sh.items():
        if len(lst) > 1:
            c, i, _ = max(lst, key=lambda x: (x[1], x[0]))
            res[(c, i)] = True
    tl, E = {}, {}
    for _ in range(4):
        E = {c: alloc_a(ct, res, p) for c, ct in ctas.items()}
        tl = {c: base_a(ct, E[c], p) for c, ct in ctas.items()}
        new = {}
        for u, lst in sh.items():
            if len(lst) > 1:
                new[max((tl[c][2][i] + E[c][i][0], c, i) for c, i, _f in lst)[1:]] = True
        if new == res:
            break
        res = new
    E = {c: alloc_a(ct, res, p) for c, ct in ctas.items()}
    out = dict(res=res, R={}, own={}, thA={}, we={})
    if not asyn:
        rel = {}
        for c, ct in ctas.items():
            st_, ts, td, t4, f, s, d = tl[c]
            ns = len(ct["segA"])
            rel[c] = [(st_[i + 1] + p.kf * f + p.phiF) if i < ns - 1 else ct["aend"] for i in range(ns)]
            out["thA"][c] = ct["aend"]
        for u, lst in sh.items():
            r = [(c, i) for c, i, _f in lst if res.get((c, i)) or _f] or [(c, i) for c, i, _f in lst]
            out["R"][u] = max(rel[c][i] for c, i in r)
            out["own"][u] = r[0][0]
        return out
    arrs, done = {}, {}
    for c, ct in ctas.items():
        segs = ct["segA"]
        if not segs:
            out["thA"][c], out["we"][c] = ct["aend"], -1e9
            continue
        st0, _, _, _, f, s, d = tl[c]
        hh = p.h(ct)
        t, we, K = st0[0], -1e9, 0
        for i, (a, b) in enumerate(segs):
            n = b - a
            K += n
            ts = max(t + f + (n - 1) * s, ct["fc"] + (K - 1) * p.tau)
            if p.sel and E[c][i][0] + E[c][i][1] <= hh + p.selm:       # inline epilogue (ship style), immediate release
                th = ts + d + E[c][i][0] + E[c][i][1]
                arr = we_i = th
            else:
                th = max(ts + d, we) + hh
                arr = th + p.m * E[c][i][0]
                we = we_i = arr + p.m * E[c][i][1]
            arrs[(c, i)], done[(c, i)] = arr, we_i
            t = th + p.g
        out["thA"][c], out["we"][c] = th, we
    for u, lst in sh.items():
        if len(lst) == 1:
            c, i, _f = lst[0]
            out["R"][u] = done[(c, i)] + p.phiR
        else:
            gm = max(E[c][i][1] for c, i, _f in lst)
            out["R"][u] = max(arrs[(c, i)] for c, i, _f in lst) + p.m * gm + p.phiR
    return out


def b_arm(ctas, cb, aout, Rf, p, asyn):
    """phase-B recursion per CTA: stage k starts at max(consumer free, R(j, s) + XL, stream floor). Ship: end work inline; async: hand-off h,
    worker runs the end work (m x), CTA ends when the worker is done. Returns end, summed wait per CTA."""
    end, wait = {}, {}
    for c, ct in ctas.items():
        segs = ct["segB"]
        if not segs:
            end[c], wait[c] = (aout["thA"][c] if asyn else ct["bend"]), 0.0
            continue
        ns = len(segs)
        w = max(0.3, (ct["F_B"] + ct["S_B"] - ct["wxB"]) / ct["nB"])
        wt = [cb[0] if b - a == 16 else cb[1] for a, b in segs]
        sc = ct["E_B"] * p.escale / sum(wt)
        EB = [x * sc for x in wt]
        d = ct["D_B"] / ns
        hh = p.h(ct)
        t = aout["thA"][c] + p.g0
        we, wsum, K = (aout["we"][c] if asyn else -1e9), 0.0, ct["nA"]
        for i, (a, b) in enumerate(segs):
            for gi in range(a, b):
                j, s = gi >> 8, gi & 15
                K += 1
                r = max(Rf(c, 32 * j + 2 * s), Rf(c, 32 * j + 2 * s + 1)) + p.XL
                stt = max(t, r, ct["fc"] + (K - 1) * p.tau)
                wsum += max(0.0, r - t)
                t = stt + w
            if asyn and not (p.sel and EB[i] <= hh + p.selm):
                th = max(t + d, we) + hh
                we = th + p.m * EB[i]
                t = th + p.g
            else:
                t += d + EB[i] + p.g
        end[c] = max(t, we + p.g) if asyn else t
        wait[c] = wsum
    return end, wait


def run(ctas, cb, p):
    """one (seed, T, layer, rep): per-CTA ends of the ship model and the three async arms."""
    a0, a1 = a_arm(ctas, p, False), a_arm(ctas, p, True)
    R0, R1, own = a0["R"], a1["R"], a0["own"]
    e0, _ = b_arm(ctas, cb, a0, lambda c, u: R0[u], p, False)
    e00, _ = b_arm(ctas, cb, a1, lambda c, u: R0[u], p, True)                              # all releases frozen at the ship times
    e_f, _ = b_arm(ctas, cb, a1, lambda c, u: R1[u] if own[u] == c else R0[u], p, True)    # only the CTA's own releases move
    e_s, _ = b_arm(ctas, cb, a1, lambda c, u: R1[u], p, True)                              # releases move with the producers
    return dict(ship={c: ct["bend"] for c, ct in ctas.items()}, e0=e0, ae00=e00, ae0=e_f, ae1=e_s)


def calib_xl(runs):
    """x landing latency after a release, per T: median consumer-visible B x wait of the CTAs whose x producer never waited on doneA."""
    out = {}
    for T in (32, 128, 256):
        v = [ct["wxB"] for (rs, ld), rr in runs.items() if dict(LOADS)[ld] == T for ctas in rr for ct in ctas.values() if ct["wB"] == 0]
        out[T] = st.median(v) if v else 1.0
    return out


def load(path):
    raw = json.load(gzip.open(path, "rt"))
    runs = {}
    for key, reps in raw.items():
        rs, ld, ly = key.split("|")
        runs.setdefault((int(rs), ld), []).extend(parse(rows) for rows in reps)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marks", default=str(ROOT / "box_out/lat32/m_ship/marks_raw.json.gz"))
    ap.add_argument("--h0", type=float, default=0.10)
    ap.add_argument("--h1", type=float, default=0.008)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.09)
    ap.add_argument("--sens", type=int, default=1)
    ap.add_argument("--scan", default="", help="comma list of m values: print only the AE1 / AE0 max-end deltas per m")
    ap.add_argument("--set", default="", help="extra model parameters for the scan, e.g. escale=0.5,h0=1.0")
    a = ap.parse_args()
    runs = load(a.marks)
    allc = [ct for v in runs.values() for ctas in v for ct in ctas.values()]
    bad = sum(len(ct["segA"]) != ct["sA"] or len(ct["segB"]) != ct["sB"] for ct in allc)
    base = dict(h0=a.h0, h1=a.h1, m=a.m, tau=a.tau)
    xl = calib_xl(runs)
    Ts = (32, 128, 256)
    cbs = {T: fit_b([x for (rs, ld), v in runs.items() if dict(LOADS)[ld] == T for x in v]) for T in Ts}
    res, fid = {}, {T: [] for T in Ts}
    seeds = sorted({rs for rs, _ in runs})

    def sweep(tag, **kw):
        for (rs, ld), v in runs.items():
            T = dict(LOADS)[ld]
            p = P(**{"XL": xl[T], **base, **kw})
            acc = {k: [] for k in ("ship", "e0", "ae00", "ae0", "ae1")}
            for ctas in v:
                r = run(ctas, cbs[T], p)
                for k in acc:
                    vals = sorted(r[k].values())
                    acc[k].append((vals[-1], vals[len(vals) // 2]))
                if tag == "P":
                    fid[T].append((st.mean(abs(r["e0"][c] - r["ship"][c]) for c in ctas), max(r["e0"].values()) - max(r["ship"].values())))
            res[(tag, T, rs)] = {k: (st.median(x[0] for x in acc[k]), st.median(x[1] for x in acc[k])) for k in acc}

    def d(tag, k, T, i=0):
        return st.median(res[(tag, T, rs)][k][i] - res[(tag, T, rs)]["e0"][i] for rs in seeds)

    go = lambda k, tag="P": d(tag, k, 32) <= -3.0 and min(d(tag, k, 128), d(tag, k, 256)) <= -3.0  # noqa: E731

    if a.scan:
        extra = {k: float(v) for k, v in (kv.split("=") for kv in a.set.split(",") if kv)}
        for mv in [float(x) for x in a.scan.split(",")]:
            tag = f"m{mv}"
            sweep(tag, **{**extra, "m": mv})
            print(f"m {mv:4.1f} {a.set}: AE1 d " + " ".join(f"{d(tag, 'ae1', T):+6.2f}" for T in Ts) + " | AE0 d " + " ".join(f"{d(tag, 'ae0', T):+6.2f}" for T in Ts)
                  + f" | {'GO' if go('ae1', tag) else 'KILL'}")
        return
    sweep("P")
    print(f"R-ae-0: XL {xl[32]:.1f}/{xl[128]:.1f}/{xl[256]:.1f} us, h {a.h0}+{a.h1}N, m {a.m}, tau {a.tau}; us, marks ~10 % slow, medians over 2 layers x 5 reps")
    print("T    seed  | ship max  p50 | E0 max | AE1 max  p50  (d max, d p50) | AE0 max (d) | AE00 max (d)")
    for T in Ts:
        for rs in seeds:
            r = res[("P", T, rs)]
            print(f"{T:<4d} {rs % 100000:>6d} | {r['ship'][0]:7.1f} {r['ship'][1]:6.1f} | {r['e0'][0]:6.1f} | {r['ae1'][0]:7.1f} {r['ae1'][1]:6.1f} "
                  f"({r['ae1'][0] - r['e0'][0]:+5.1f}, {r['ae1'][1] - r['e0'][1]:+5.1f}) | {r['ae0'][0]:6.1f} ({r['ae0'][0] - r['e0'][0]:+5.1f})"
                  f" | {r['ae00'][0]:6.1f} ({r['ae00'][0] - r['e0'][0]:+5.1f})")
    for T in Ts:
        print(f"median over seeds T={T}: ship max {st.median(res[('P', T, rs)]['ship'][0] for rs in seeds):6.1f} -> AE1 d max {d('P', 'ae1', T):+5.2f} "
              f"d p50 {d('P', 'ae1', T, 1):+5.2f} | AE0 {d('P', 'ae0', T):+5.2f} | AE00 {d('P', 'ae00', T):+5.2f}")
    print(f"GATE (>= 3 us at T=32 and at T=128 or 256): AE1 {'GO' if go('ae1') else 'KILL'} | AE0 {'GO' if go('ae0') else 'KILL'} | AE00 {'GO' if go('ae00') else 'KILL'}")
    if a.sens:
        for tag, kw in (("m2", dict(m=2.0)), ("m3", dict(m=3.0)), ("m2+combo", dict(m=2.0, escale=0.85, h0=0.2, h1=0.016, tau=1.3)),
                        ("ideal", dict(m=0.0, h0=0.0, h1=0.0, phiR=0.0))):
            sweep(tag, **kw)
            print(f"sens {tag:8s}: AE1 d {d(tag, 'ae1', 32):+5.2f} {d(tag, 'ae1', 128):+5.2f} {d(tag, 'ae1', 256):+5.2f} ({'GO' if go('ae1', tag) else 'KILL'})"
                  f" | AE0 d {d(tag, 'ae0', 32):+5.2f} {d(tag, 'ae0', 128):+5.2f} {d(tag, 'ae0', 256):+5.2f}")
    print(f"plan check {len(allc)} CTA-reps, {bad} segment-count mismatches | ship model vs measured per-CTA end: " + " ".join(
        f"T{T} MAE {st.mean(x[0] for x in fid[T]):.2f} max-bias {st.mean(x[1] for x in fid[T]):+.2f}" for T in Ts))


if __name__ == "__main__":
    main()

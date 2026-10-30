"""plan_search.py: $0 CPU search over STATIC VML plans (T >= 32 MoE) with a sub-step cost model (plan search, Oct 30).

Extends tools/moe_deal_sim.py (R-deal-0; not edited) with the shipped planner reproduced exactly and the measured
segment-end sub-steps of R-vse-1 / R-vse-2 (notes/lab.md, commit 5fd4ccee):
  planner  route() under VM_RT (kernels/muse_moe_w4a16/mb_21.py ~1287-1424): per expert in id order, VTR jobs
           (VM_TA: nj = ceil(n / JN), base = ceil(ceil(n / 8) / nj) * 8), JN 64 (bf2, T <= 48) / 72 (w72, T >= 49);
           A / B cost scans 512 / 256 stages x MB_LAYER_COST[bucket] (extended to NTM 9 like _vml_cost_ext);
           cut x = floor(tot * c / G), st = floor((x - cp[j]) / cost) clamped to ns - 1 (stage_at);
           A order per CTA: split_point rotation (flags bit 0): [last unit edge, g1) first, then [g0, edge).
  timeline per CTA: start + stages x pace(nt) (+ first-stage-of-segment saving) + per A piece end work:
           whole unit   EPI + PF                          (SwiGLU epilogue + deferred pend_flush release)
           split share  RED + CNT                         (red.add / PS plain store issue + bar/atom/bar count step)
           resolver     + RB + EPI + PF                   (last arriver, found by fixed-point iteration over CTAs)
           B piece end  BEND; B stage s of job j waits doneA of A units 2s, 2s+1 (+ HOP + XLAT x copy).
  fluid credit phi: the chip is HBM-bound, so a stalled CTA's bandwidth goes to the others. phi = 1 hides the MEAN
           per-CTA overhead (stage pace sped up by mean overhead / mean stage time, capped at PMIN); phi = 0 = additive.
           R-vse-2 says phi ~0 at T 32 (PS saved 0.71 us / CTA -> end -0.8), ~0.8 at 128 (0.86 -> -0.15), ~1 at 256.
  calibration: pace scale a and constant C0 per bucket, least squares on the 12 measured per-routing medians of
           R-vse-1 / R-vse-2 (box_out/vse/run{1,2}.json, the same 6 board routings regenerated on CUDA; bf2 at 32, w72 at 128 / 256).
Plan knobs (static, the shipped planner or a small extension of it):
  jn, ta            job size / tile-aligned split          tabA / tabB   cost tables (per nt)
  sg                VM_SC per-unit A cost in the cut      snap          move an A cut <= snap stages from a unit edge onto it
  grp               every grp-th A cut on a unit edge, the cuts between them split the group's work evenly
  rot               split_point rotation on / off         order         job order: id / nt_desc / nt_asc / zig
  bbal              B cut evens predicted A end + B work per CTA (A cut unchanged)
  segc              A cut charges the predicted per-CTA overhead (whole / share / resolver), fixed point, A only
Usage:
  python tools/plan_search.py calib            # fit a / C0, show model vs measured, model A-end p50 / max vs marks
  python tools/plan_search.py search [--n=40]  # candidate plans vs ship on board bs 16/32/128/256 + layer T 32/128/256
  python tools/plan_search.py sens   [--n=40]  # best plans under cost / phi / pace-error / KPERM3 variants
Board routings: tools/plan_search_routings.json = _mb_board_input(bs, seed 100..139) generated on CUDA (seeds 100-105 are
the R-vse runs); layer routings: MB_LAYER_PROFILES of mb_21 + box_out/*/run.json (tools/slice_plan.py), 3 id placements.
"""
import bisect
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
G = 132
MB_LAYER_COST = {   # kernels/muse_moe_w4a16/mb_21.py MB_LAYER_COST (A nt1..8, B nt1..8)
    256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
    128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
    32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}
MB_LAYER_PROFILES = {   # mb_21 MB_LAYER_PROFILES (in-layer per-expert counts)
    32: [[26, 11, 7, 6, 3, 3, 2, 1, 1, 1], [17, 17, 4, 2, 1, 1, 1, 1, 1], [22, 19, 11, 4, 3, 3, 3, 1, 1, 1], [23, 13, 11, 10],
         [24, 16, 10, 6, 4, 3, 3], [19, 15, 6, 5, 4]],
    256: [[122, 83, 45, 39, 30, 28, 15, 13, 9, 6, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1], [107, 91, 44, 36, 35, 10, 9, 6, 6, 5, 5, 5, 4, 3, 2, 2, 1, 1, 1],
          [81, 76, 65, 45, 43, 31, 27, 25, 24, 9, 9, 5, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1], [72, 67, 50, 47, 41, 41, 28, 22, 20, 16, 6, 4, 2, 1, 1, 1, 1, 1],
          [107, 84, 38, 28, 26, 26, 17, 9, 8, 8, 7, 5, 5, 4, 4, 2, 2, 2, 1, 1], [117, 100, 48, 36, 25, 13, 8, 8, 8, 7, 5, 5, 3, 2, 2, 1, 1, 1, 1, 1]],
    128: [[49, 27, 27, 21, 13, 13, 8, 6, 4, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1], [50, 45, 15, 14, 12, 7, 4, 4, 3, 2, 2, 2, 1, 1, 1, 1],
          [47, 37, 32, 28, 20, 19, 17, 11, 8, 5, 3, 2, 2, 2, 1, 1, 1, 1, 1], [43, 26, 25, 25, 16, 12, 10, 10, 9, 8, 3, 2, 2, 2, 1, 1],
          [53, 41, 27, 24, 18, 12, 6, 6, 5, 5, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1], [54, 49, 28, 21, 20, 7, 6, 5, 3, 3, 2, 2, 2, 2, 2, 1, 1]]}
# measured per-routing medians (us, vm_main + vm_fin, gated timer), board routings seeds 100..105: run1 / run2 base arms
MEAS = {32: [[116.0, 139.1, 145.6, 115.0, 122.0, 134.7], [117.3, 141.2, 148.1, 116.8, 123.9, 136.3]],
        128: [[206.0, 198.4, 194.4, 176.3, 150.5, 178.3], [208.5, 200.4, 197.4, 178.0, 152.9, 181.4]],
        256: [[231.5, 219.5, 231.3, 229.3, 198.8, 212.9], [232.9, 221.5, 233.7, 231.4, 201.0, 215.6]]}
# marks (marked builds, avg of 6 routings): A segs / CTA, splits / CTA, Aend p50, Aend max, end max (run2 base)
MARKS = {32: (4.856, 1.874, 85.3, 96.7, 132.2), 128: (6.528, 1.823, 121.2, 132.1, 184.8), 256: (7.232, 1.859, 144.2, 152.8, 215.5)}
# measured sub-steps per piece (us), R-vse-1 / R-vse-2 bs32 / 128 / 256 (base, and PS = the shipped mb_21 / rider3+PS path)
SUB = {"base": dict(RED=0.46, CNT=1.86, RB=0.79, EPI=0.60, PF=0.73, BEND=0.62),
       "ps": dict(RED=0.68, CNT=1.50, RB=0.75, EPI=0.59, PF=0.68, BEND=0.62)}
DEF = dict(START=8.0, HOP=0.5, XLAT=0.6, FS=0.33, PMIN=0.67, phi={32: 0.0, 128: 0.8, 256: 1.0}, sub="ps",
           kA=0.0, kB=0.0, err=None, ext_pro=0.5)


def bucket(T):
    return 32 if T <= 48 else (128 if T <= 192 else 256)


def cost_ext(c, ntm):
    out = []
    for pb in (0, 1):
        h = list(c[8 * pb:8 * pb + 8])
        step = max(1, h[7] - h[6])
        out.append(h + [h[7] + step * (k - 8) for k in range(9, ntm + 1)])
    return out


def jobs_of(counts, jn, ta=True, order="id"):
    """counts in expert-id order (zeros allowed) -> list of nt per job, in planner order"""
    per = []
    for n in counts:
        if n <= 0:
            continue
        nj = -(-n // jn)
        base = (-(-(-(-n // 8)) // nj) * 8) if ta else -(-n // nj)
        js = []
        for k in range(nj):
            sz = min(base, n - k * base)
            if sz > 0:
                js.append(-(-sz // 8))
        per.append(js)
    if order == "nt_desc":
        per.sort(key=lambda js: -max(js))
    elif order == "nt_asc":
        per.sort(key=lambda js: max(js))
    elif order == "zig":   # heavy, light, heavy, light ...
        s = sorted(per, key=lambda js: -max(js))
        per = [s[(i // 2) if i % 2 == 0 else len(s) - 1 - i // 2] for i in range(len(s))]
    return [nt for js in per for nt in js]


def cut(nts, ns, tab, Gc, sg=0, snap=0, grp=0, targets=None):
    """stage_at for c = 0..G (VM_RT floor rule); sg = VM_SC per-unit cost; snap / grp = unit-edge snapping extensions;
    targets = explicit cumulative cost targets (len G + 1) instead of tot * c / G"""
    sh = 9 if ns == 512 else 8
    cp = [0]
    for nt in nts:
        cp.append(cp[-1] + ns * tab[nt - 1] + (ns // 16) * sg)
    tot = cp[-1]
    nj = len(nts)

    def at(x):
        j = min(nj - 1, bisect.bisect_right(cp, x) - 1)
        cs, xr = tab[nts[j] - 1], x - cp[j]
        if sg:
            uc = 16 * cs + sg
            uu = xr // uc
            rr = xr - uu * uc
            s = 16 * uu + ((rr - sg) // cs if rr > sg else 0)
        else:
            s = xr // cs
        return (j << sh) + min(ns - 1, int(s))

    def snapto(g):
        w = g & 15
        return g - w if w <= 8 else g + 16 - w

    b = []
    for c in range(Gc + 1):
        if c >= Gc:
            b.append(nj << sh)
            continue
        x = targets[c] if targets is not None else tot * c // Gc
        g = at(int(x))
        if snap:
            w = g & 15
            if w and w <= snap:
                g -= w
            elif w and 16 - w <= snap:
                g += 16 - w
        b.append(g)
    if grp:
        for c in range(0, Gc, grp):
            if 0 < c < Gc:
                b[c] = snapto(b[c])
        for c0 in range(0, Gc, grp):   # inside a group: even split of the group's stage-cost (contiguous)
            c1 = min(Gc, c0 + grp)
            lo, hi = b[c0], b[c1]
            # cumulative cost inside [lo, hi)
            def cc(g):
                j, s = g >> sh, g & ((1 << sh) - 1)
                return cp[j] + s * tab[nts[j] - 1] if j < nj else tot
            a0, a1 = cc(lo), cc(hi)
            for k in range(1, c1 - c0):
                b[c0 + k] = max(lo, min(hi, at(int(a0 + (a1 - a0) * k / (c1 - c0)))))
    for c in range(1, Gc + 1):
        b[c] = max(b[c], b[c - 1])
    return b


def pieces(g0, g1, rot, sh):
    if g1 <= g0:
        return []
    ue = ((g1 - 1) >> 4) << 4
    rs = [(ue, g1), (g0, ue)] if (rot and (g1 & 15) and ue > g0) else [(g0, g1)]
    out = []
    for a, b in rs:
        gs = a
        while gs < b:
            s0 = gs & 15
            s1 = min(16, s0 + (b - gs))
            out.append((gs >> 4, s0, s1))
            gs += s1 - s0
    return out


def paces(T, P, a, ntm, tabA, tabB):
    """true per-stage us by nt: shipped table shape x a (the L tables are per-CTA fits), KPERM3 / error variants"""
    c0 = cost_ext(MB_LAYER_COST[bucket(T)], ntm)
    pa = [a * c0[0][k] / 128.0 for k in range(ntm)]
    pb = [a * c0[1][k] / 128.0 for k in range(ntm)]
    if P.get("err"):
        rng = random.Random(P["err"][1])
        pa = [p * (1 + rng.gauss(0, P["err"][0])) if k else p for k, p in enumerate(pa)]
        pb = [p * (1 + rng.gauss(0, P["err"][0])) if k else p for k, p in enumerate(pb)]
    kA, kB = P.get("kA", 0.0), P.get("kB", 0.0)   # KPERM3: (uniform, nt >= 2 only) stage-time cut
    pa = [p * (1 - kA) * (1 - (kB if k else 0.0)) for k, p in enumerate(pa)]
    pb = [p * (1 - kA) * (1 - (kB if k else 0.0)) for k, p in enumerate(pb)]
    return pa, pb


def simulate(plan, counts, T, P, a, C0, detail=False):
    jn = plan.get("jn", 64 if bucket(T) == 32 else 72)
    nts = jobs_of(counts, jn, plan.get("ta", True), plan.get("order", "id"))
    if not nts:
        return None
    ntm = max(8, max(nts), -(-jn // 8))
    b8 = MB_LAYER_COST[bucket(T)]
    tabA, tabB = cost_ext(list(plan.get("tabA") or b8[:8])[:8] + list(plan.get("tabB") or b8[8:])[:8], ntm)
    S = dict(SUB[P.get("sub", "ps")])
    S.update(P.get("subov", {}))
    pa, pb = paces(T, P, a, ntm, tabA, tabB)
    rot = plan.get("rot", True)
    # ---- cuts
    bA = cut(nts, 512, tabA, G, plan.get("sg", 0), plan.get("snap", 0), plan.get("grp", 0))
    if plan.get("segc"):
        u2c = tabA[0] / pa[0]   # planner's cost units per us (compile-time constant in a real build)
        for _ in range(3):
            ov = []
            for c in range(G):
                ps = pieces(bA[c], bA[c + 1], rot, 9)
                ns_ = sum(1 for _, s0, s1 in ps if not (s0 == 0 and s1 == 16))
                nw = len(ps) - ns_
                ov.append(nw * (S["EPI"] + S["PF"]) + ns_ * (S["RED"] + S["CNT"] + 0.5 * (S["RB"] + S["EPI"] + S["PF"])))
            # per-CTA stage-cost target = (total stage cost + total overhead) / G - own overhead
            cp = [0]
            for nt in nts:
                cp.append(cp[-1] + 512 * tabA[nt - 1])
            totc = cp[-1] + sum(ov) * u2c * plan["segc"]
            tg, acc = [0], 0.0
            for c in range(G):
                acc += max(0.0, totc / G - ov[c] * u2c * plan["segc"])
                tg.append(acc)
            sc = cp[-1] / tg[-1]
            bA = cut(nts, 512, tabA, G, snap=plan.get("snap", 0), targets=[x * sc for x in tg])
    pA = [pieces(bA[c], bA[c + 1], rot, 9) for c in range(G)]
    nsh = sum(1 for ps in pA for _, s0, s1 in ps if not (s0 == 0 and s1 == 16))
    if plan.get("bbal"):
        # predicted A time per CTA (planner view: stage costs + whole / share / half resolver), B cut evens A + B
        apred = []
        for c in range(G):
            t = 0.0
            for u, s0, s1 in pA[c]:
                nt = nts[u >> 5]
                t += (s1 - s0) * tabA[nt - 1] / tabA[0] * pa[0]
                t += (S["EPI"] + S["PF"]) if (s0 == 0 and s1 == 16) else (S["RED"] + S["CNT"] + 0.5 * (S["RB"] + S["EPI"] + S["PF"]))
            apred.append(t)
        cpb = [0]
        for nt in nts:
            cpb.append(cpb[-1] + 256 * tabB[nt - 1])
        u2c = tabB[0] / pb[0]
        tstar = (cpb[-1] / u2c + sum(apred)) / G
        want = [max(0.0, tstar - x) for x in apred]
        sc = cpb[-1] / u2c / sum(want)
        tg, acc = [], 0.0
        for c in range(G):
            tg.append(acc * u2c * sc)
            acc += want[c]
        bB = cut(nts, 256, tabB, G, targets=tg + [cpb[-1]])
    else:
        bB = cut(nts, 256, tabB, G)
    pB = [pieces(bB[c], bB[c + 1], False, 8) for c in range(G)]
    # ---- fluid credit: mean overhead hidden by bandwidth sharing
    nwA = sum(1 for ps in pA for _, s0, s1 in ps if s0 == 0 and s1 == 16)
    nsu = len({u for ps in pA for u, s0, s1 in ps if not (s0 == 0 and s1 == 16)})
    ovm = (nwA * (S["EPI"] + S["PF"]) + nsh * (S["RED"] + S["CNT"]) + nsu * (S["RB"] + S["EPI"] + S["PF"])
           + sum(len(ps) for ps in pB) * S["BEND"]) / G
    stm = sum((s1 - s0) * pa[nts[u >> 5] - 1] for ps in pA for u, s0, s1 in ps) / G + \
        sum((s1 - s0) * pb[nts[u >> 4] - 1] for ps in pB for u, s0, s1 in ps) / G
    phi = P["phi"][bucket(T)] if isinstance(P["phi"], dict) else P["phi"]
    f = max(P["PMIN"] / max(pa[0], 1e-9), 1.0 - phi * ovm / stm)
    pa = [p * f for p in pa]
    pb = [p * f for p in pb]
    # ---- phase A with resolver fixed point
    start = P["START"] + (P["ext_pro"] if (plan.get("bbal") or plan.get("segc") or plan.get("grp") or plan.get("sg")) else 0.0)
    owners = {}
    for c in range(G):
        for k, (u, s0, s1) in enumerate(pA[c]):
            if not (s0 == 0 and s1 == 16):
                owners.setdefault(u, []).append((c, k))
    res = {u: max(v, key=lambda ck: ck[1]) for u, v in owners.items()}   # guess: the later piece in its own order
    for _ in range(6):
        aend, arr, done = [0.0] * G, {}, {}
        for c in range(G):
            t, pend = start, None
            for k, (u, s0, s1) in enumerate(pA[c]):
                p = pa[nts[u >> 5] - 1]
                t += p * (s1 - s0) - (P["FS"] * f if k else 0.0)
                if pend is not None:   # VM_DEFER: previous release after this segment's first stage
                    done[pend[0]] = t - p * (s1 - s0 - 1) + S["PF"]
                    pend = None
                if s0 == 0 and s1 == 16:
                    t += S["EPI"] + S["PF"]
                    pend = (u,)
                else:
                    t += S["RED"]
                    arr[(c, k)] = t + 0.5 * S["CNT"]
                    t += S["CNT"]
                    if res.get(u) == (c, k):
                        t += S["RB"] + S["EPI"] + S["PF"]
                        pend = (u,)
            if pend is not None:
                done[pend[0]] = t
            aend[c] = t
        new = {u: max(v, key=lambda ck: arr[ck]) for u, v in owners.items()}
        if new == res:
            break
        res = new
    # ---- phase B
    bend = [0.0] * G
    wait = 0.0
    for c in range(G):
        t = aend[c]
        for u, s0, s1 in pB[c]:
            j = u >> 4
            p = pb[nts[j] - 1]
            for s in range(s0, s1):
                r = max(done.get(j * 32 + 2 * s, 0.0), done.get(j * 32 + 2 * s + 1, 0.0)) + P["HOP"] + P["XLAT"]
                if r > t:
                    wait += r - t
                    t = r
                t += p
            t += S["BEND"]
        bend[c] = t
    act = [c for c in range(G) if pA[c] or pB[c]]
    end = max(bend) + C0
    if not detail:
        return end
    ae = sorted(aend[c] for c in act)
    nres = [0] * G
    for (c, k) in res.values():
        nres[c] += 1
    return dict(end=end, aend_p50=ae[len(ae) // 2], aend_max=ae[-1], segA=sum(len(pA[c]) for c in act) / len(act),
                split=nsh / len(act), wait=wait / len(act), ovm=ovm, f=f, nj=len(nts),
                bal=max(bend) - st.mean(bend[c] for c in act), res2=sum(1 for c in act if nres[c] >= 2) / len(act),
                res0=sum(1 for c in act if nres[c] == 0) / len(act))


# ------------------------------------------------------------------------------------------------ routings
def board(bs, n):
    d = json.load(open(os.path.join(HERE, "plan_search_routings.json")))
    return d[str(bs)][:n]


def layer(T, n_place=3):
    out = [list(x) for x in MB_LAYER_PROFILES[T]]
    try:
        sys.path.insert(0, HERE)
        import slice_plan
        cwd = os.getcwd()
        os.chdir(os.path.dirname(HERE))
        out += slice_plan.routings().get(T, [])
        os.chdir(cwd)
    except Exception:
        pass
    rs = []
    for i, cnt in enumerate(out):   # place experts at random local ids (like _mb_layer_input)
        for p in range(n_place):
            ids = random.Random(1000 * T + 17 * i + p).sample(range(32), len(cnt))
            v = [0] * 32
            for e, n in zip(ids, cnt):
                v[e] = n
            rs.append(v)
    return rs


# ------------------------------------------------------------------------------------------------ calibration
def calibrate(P, verbose=False):
    fit = {}
    for T in (32, 128, 256):
        rs = board(T, 6)
        ys = [(st.mean(r[i] for r in MEAS[T])) for i in range(6)]
        best = None
        for a100 in range(80, 131):
            a = a100 / 100.0
            e = [simulate({}, rs[i], T, P, a, 0.0) for i in range(6)]
            C0 = st.mean(ys[i] - e[i] for i in range(6))
            sse = sum((ys[i] - e[i] - C0) ** 2 for i in range(6))
            if best is None or sse < best[0]:
                best = (sse, a, C0, e)
        fit[T] = (best[1], best[2])
        if verbose:
            sse, a, C0, e = best
            print(f"T{T}: pace a {a:.2f} us / nt1 stage, C0 {C0:5.1f} us, rms {(sse / 6) ** 0.5:4.2f} us | model "
                  + " ".join(f"{x + C0:6.1f}" for x in e) + " | meas " + " ".join(f"{y:6.1f}" for y in ys))
            dd = [simulate({}, rs[i], T, P, a, C0, detail=True) for i in range(6)]
            m = MARKS[T]
            print(f"     model segA/CTA {st.mean(x['segA'] for x in dd):.2f} (marks {m[0]}), split/CTA {st.mean(x['split'] for x in dd):.2f}"
                  f" ({m[1]}), Aend max - p50 {st.mean(x['aend_max'] - x['aend_p50'] for x in dd):4.1f} (marks {m[3] - m[2]:.1f}),"
                  f" end - Aend max {st.mean(x['end'] - C0 - x['aend_max'] for x in dd):5.1f} (marks {m[4] - m[3]:.1f}), B wait/CTA"
                  f" {st.mean(x['wait'] for x in dd):.2f}, fluid f {st.mean(x['f'] for x in dd):.3f}")
    fit[16] = fit[32]
    return fit


# ------------------------------------------------------------------------------------------------ search
def plans_for(T):
    b8 = MB_LAYER_COST[bucket(T)]
    base = (b8[:8], b8[8:])
    ps = {"ship": {}}
    for m in (1, 2, 3, 4):
        ps[f"snap{m}"] = {"snap": m}
    for g in (2, 3, 4, 6):
        ps[f"grp{g}"] = {"grp": g}
    for sg in (64, 128, 256):
        ps[f"sc{sg}"] = {"sg": sg}
    ps["norot"] = {"rot": False}
    for o in ("nt_desc", "nt_asc", "zig"):
        ps[o] = {"order": o}
    ps["bbal"] = {"bbal": 1}
    ps["segc"] = {"segc": 1.0}
    ps["segc+bbal"] = {"segc": 1.0, "bbal": 1}
    for jn in (56, 64, 72, 80, 96):
        ps[f"jn{jn}"] = {"jn": jn}
    ps["noTA"] = {"ta": False}
    for x in (90, 95, 105, 110):   # nt >= 2 entries scaled (the board's Lx<pct> arms)
        ps[f"Lx{x}"] = {"tabA": [base[0][0]] + [round(v * x / 100) for v in base[0][1:]],
                        "tabB": [base[1][0]] + [round(v * x / 100) for v in base[1][1:]]}
    return ps


def evaluate(names_plans, sets, P, fit):
    out = {}
    for key, (T, rs) in sets.items():
        a, C0 = fit[bucket(T) if T != 16 else 16]
        ship = [simulate({}, r, T, P, a, C0) for r in rs]
        for nm, pl in names_plans(T).items():
            v = [simulate(pl, r, T, P, a, C0) for r in rs]
            d = sorted(x - y for x, y in zip(v, ship))
            out[(key, nm)] = (st.median(d), d[len(d) // 10], d[(9 * len(d)) // 10], st.median(v))
    return out


def sets_of(n):
    s = {}
    for bs in (16, 32, 128, 256):
        s[f"b{bs}"] = (bs, board(bs, n))
    for T in (32, 128, 256):
        s[f"L{T}"] = (T, layer(T))
    return s


def table(res, sets, names):
    print(f"{'plan':12s}" + "".join(f"{k:>18s}" for k in sets))
    for nm in names:
        print(f"{nm:12s}" + "".join(f"  {res[(k, nm)][0]:+5.1f} [{res[(k, nm)][1]:+5.1f},{res[(k, nm)][2]:+5.1f}]" for k in sets))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "calib"
    n = int(next((x.split("=")[1] for x in sys.argv if x.startswith("--n=")), 40))
    P = dict(DEF)
    if mode == "calib":
        for sub in ("ps", "base"):
            print(f"== sub-step set {sub}")
            calibrate(dict(P, sub=sub), verbose=True)
        return
    fit = calibrate(P)
    sets = sets_of(n)
    if mode == "search":
        res = evaluate(plans_for, sets, P, fit)
        print("median d(max end) vs ship, us [p10, p90] over routings; ship median per set:",
              {k: round(res[(k, 'ship')][3], 1) for k in sets})
        table(res, sets, list(plans_for(32).keys()))
    if mode == "bound":   # how much ANY static re-partition could gain in the model: max - mean per-CTA end (same work)
        for key, (T, rs) in sets.items():
            a, C0 = fit[bucket(T) if T != 16 else 16]
            for nm, pl in (("ship", {}), ("norot", {"rot": False})):
                dd = [simulate(pl, r, T, P, a, C0, detail=True) for r in rs]
                print(f"{key:5s} {nm:6s} end {st.median(x['end'] for x in dd):6.1f}  max - mean CTA end {st.median(x['bal'] for x in dd):4.1f}"
                      f"  Aend max - p50 {st.median(x['aend_max'] - x['aend_p50'] for x in dd):4.1f}  B wait/CTA {st.median(x['wait'] for x in dd):4.2f}"
                      f"  CTAs resolving 2 / 0 units {st.median(x['res2'] for x in dd):.2f} / {st.median(x['res0'] for x in dd):.2f}"
                      f"  mean overhead/CTA {st.median(x['ovm'] for x in dd):4.1f}")
    if mode == "sens":
        names = ["norot", "snap2", "grp3", "segc", "bbal", "segc+bbal", "sc64"]
        variants = {"default (ps, phi 0/.8/1)": {}, "sub base": {"sub": "base"}, "phi 0 all": {"phi": 0.0},
                    "phi 1 all": {"phi": 1.0}, "split CNT x1.5": {"subov": {"CNT": 2.25}}, "resolver x2": {"subov": {"RB": 1.5}},
                    "pace err 5% s1": {"err": (0.05, 1)}, "pace err 5% s2": {"err": (0.05, 2)},
                    "KPERM3 -3%": {"kA": 0.03}, "KPERM3 -6%": {"kA": 0.06}, "KPERM3 -6% nt>=2": {"kB": 0.06}}
        sub = {k: v for k, v in sets.items() if k in ("b32", "b128", "b256", "L32", "L128", "L256")}
        for vn, ov in variants.items():
            P2 = dict(P, **ov)
            fit2 = fit if not ov or vn.startswith("KPERM3") else calibrate(P2)
            r = evaluate(lambda T: {k: v for k, v in plans_for(T).items() if k in names + ["ship"]}, sub, P2, fit2)
            ship = " ".join(f"{k} {r[(k, 'ship')][3]:6.1f}" for k in sub)
            print(f"-- {vn}: ship {ship}")
            for nm in names:
                print(f"   {nm:10s}" + "".join(f" {k} {r[(k, nm)][0]:+5.1f}" for k in sub))
        # KPERM3: does the best snap / segc trade-off shift? cost-table refit = nt>=2 entries x (1 - k)
        for k in (0.03, 0.06):
            P2 = dict(P, kB=k)
            def kp(T, k=k):
                b8 = MB_LAYER_COST[bucket(T)]
                fa = [b8[0]] + [round(v * (1 - k)) for v in b8[1:8]]
                fb = [b8[8]] + [round(v * (1 - k)) for v in b8[9:]]
                return {"ship": {}, "refit": {"tabA": fa, "tabB": fb}, "refit+norot": {"tabA": fa, "tabB": fb, "rot": False},
                        "refit+snap2": {"tabA": fa, "tabB": fb, "snap": 2}, "refit+segc": {"tabA": fa, "tabB": fb, "segc": 1.0}}
            r = evaluate(kp, sub, P2, fit)
            print(f"-- KPERM3 nt>=2 -{int(k * 100)}% with the cost tables re-fit (vs the old tables on the faster kernel)")
            for nm in ("refit", "refit+norot", "refit+snap2", "refit+segc"):
                print(f"   {nm:12s}" + "".join(f" {kk} {r[(kk, nm)][0]:+5.1f}" for kk in sub))


if __name__ == "__main__":
    main()

"""xcf_plan.py: XCF work plan (notes/handoff_xcf.md) and its cost projection on the layer routings (R-xcf-0 probe table).

    python tools/xcf_plan.py [--probe=box_out/xcf/probe.json] [--probe20=box_out/xcf/probe_ey20.json] [--G=132]

Plan: items = (expert, 32-dim slice), expert-major; slice cost = 6 stage-equivalents (A: 64 W13 rows x 4096 K = 4 stages of
36 KB, B: 4096 W2 rows x 32 K = 2) x c(N_e), c = the incumbent stage time at N_e tokens (passes of <= 64 tokens). Each CTA gets
one contiguous range (cuts at the k / G quantiles of the cumulative cost). A CTA's segments = its runs of one expert;
segment K extent for phase B = 32 x slices.
Projection per routing (us, per CTA, the max over CTAs is the call's end):
  base   = sum of stage-equivalents x c(N) (the XCF plan with free B reductions and free partial row groups)
  b_red  = B stages x c(N) x rel(N, KB) from the probe (kb arms vs inc, interpolated in log2 KB; KB >= 4096 -> kb4096)
  a_tail = per segment: the last A row group of r = slices mod 4 WGs costs aw_r / inc of a full row group, not r / 4
Layer routings: box_out/{vms,vms6,m32,sk}/run.json (counts per expert).
--clusters: the cluster variants (Astra, relayed 2026-10-29): (a) C CTAs per expert, experts never split (full-K B, no red),
whole experts LPT-placed on 128 / C clusters -> makespan / ideal; (b) XCF-C: clusters of C CTAs get contiguous cost-balanced
slice ranges, h shared in the cluster, B K = the cluster's segment (red vs the fair kb4096 zero), busiest CTA's A share
(ceil + partial row group via aw) -> worst CTA / ideal.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROUTING_FILES = ("box_out/vms/run.json", "box_out/vms6/run.json", "box_out/m32/run.json", "box_out/sk/run.json")
C_INC = {8: 1.155, 16: 1.307, 24: 1.46, 32: 1.617, 40: 1.76, 48: 1.90, 56: 2.040, 64: 2.167}   # R-xc-0 inc cold, us / stage


def c_stage(n, table=C_INC):
    """incumbent stage time (us) at n tokens: passes of <= 64 tokens, each padded to 8."""
    if n <= 0:
        return 0.0
    p = math.ceil(n / 64)
    m = -(-math.ceil(n / p) // 8) * 8
    return p * table[min(64, max(8, m))]


def routings():
    out = {}
    for f in ROUTING_FILES:
        p = ROOT / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["routings"]:
            T = r.get("T", r.get("bs"))
            out.setdefault(T, []).append([int(c) for c in r["counts"] if c > 0])
    return out


def plan(counts, G=132, cfun=c_stage):
    """contiguous cost-balanced ranges of (expert, slice) items -> per CTA [(expert, n_tokens, slices)]."""
    sl = [(e, n, 6.0 * cfun(n)) for e, n in enumerate(counts) for _ in range(128)]
    tot = sum(c for _, _, c in sl)

    def pack(cap):   # greedy fill up to cap: the optimal contiguous partition is the least feasible cap
        out, cur, acc = [], [], 0.0
        for e, n, c in sl:
            if cur and acc + c > cap:
                out.append(cur)
                cur, acc = [], 0.0
            cur.append((e, n))
            acc += c
        out.append(cur)
        return out
    lo, hi = tot / G, tot / G + max(c for _, _, c in sl)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if len(pack(mid)) <= G:
            hi = mid
        else:
            lo = mid
    ctas = pack(hi)
    while len(ctas) < G:
        ctas.append([])
    segs = []
    for cl in ctas:
        s = []
        for e, n in cl:
            if s and s[-1][0] == e:
                s[-1][2] += 1
            else:
                s.append([e, n, 1])
        segs.append([tuple(x) for x in s])
    return segs, tot / G


def load_probe(path):
    p = ROOT / path
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rel, aw, inc = {}, {}, {}
    for row in d["rows"]:
        if row["mode"] != "cold":
            continue
        N = int(row["N"])
        st = {int(k): v for k, v in row["stage_us"].items()}
        inc[N] = st[0]
        for k, v in st.items():
            if k > 0:
                rel[(N, k)] = v / st[0] - 1.0
            elif k < 0:
                aw[(N, -k)] = v / st[0]
    return {"rel": rel, "aw": aw, "inc": inc}


def interp_rel(pr, n, kb):
    """rel cost of a B stage at n tokens (one pass <= 64) and tile K extent kb, from the probe grid (log2 kb, linear n)."""
    ns = sorted({N for N, _ in pr["rel"]})
    kbs = sorted({k for _, k in pr["rel"]})
    n = min(max(n, ns[0]), ns[-1])
    kb = min(max(kb, kbs[0]), kbs[-1])

    def at(N, k):
        lo = max([x for x in kbs if x <= k])
        hi = min([x for x in kbs if x >= k])
        if lo == hi:
            return pr["rel"][(N, lo)]
        f = (math.log2(k) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
        return pr["rel"][(N, lo)] * (1 - f) + pr["rel"][(N, hi)] * f
    nlo = max([x for x in ns if x <= n])
    nhi = min([x for x in ns if x >= n])
    if nlo == nhi:
        return at(nlo, kb)
    f = (n - nlo) / (nhi - nlo)
    return at(nlo, kb) * (1 - f) + at(nhi, kb) * f


def project(counts, pr, G=132):
    segs, avg = plan(counts, G)
    per = []
    for s in segs:
        base = b_red = a_tail = 0.0
        for e, n, k in s:
            cn = c_stage(n)
            base += 6.0 * k * cn
            p = math.ceil(n / 64)
            npass = -(-math.ceil(n / p) // 8) * 8
            if pr:
                b_red += 2.0 * k * cn * interp_rel(pr, npass, 32 * k)
                r = k % 4
                if r and pr["aw"]:
                    awr = pr["aw"].get((min(64, max(8, npass)), r), r / 4)
                    a_tail += 16.0 * cn * (awr - r / 4.0)   # one row group = 16 stages x 4 WGs; cost aw_r instead of r / 4
        per.append((base, b_red, a_tail))
    mx = max(range(G), key=lambda i: sum(per[i]))
    return {"avg_base": avg, "max_base": max(p[0] for p in per), "max_total": sum(per[mx]), "b_red_mean": sum(p[1] for p in per) / G,
            "a_tail_mean": sum(p[2] for p in per) / G, "segs_mean": sum(len(s) for s in segs) / G,
            "kb_med": sorted(32 * k for s in segs for _, _, k in s)[len([1 for s in segs for _ in s]) // 2]}


def ecost(n):
    return 768.0 * c_stage(n)   # one expert: W13 512 + W2 256 stage-equivalents


def astra(counts, C, ncta=128):
    loads = [0.0] * (ncta // C)
    for c in sorted((ecost(n) for n in counts), reverse=True):
        i = min(range(len(loads)), key=lambda k: loads[k])
        loads[i] += c / C
    return max(loads) / (sum(ecost(n) for n in counts) / 132)


def xcfc(counts, pr, C, ncl):
    fair = {"rel": {}, "aw": pr["aw"]}
    for (N, k), v in pr["rel"].items():
        fair["rel"][(N, k)] = (1 + v) / (1 + pr["rel"][(N, 4096)]) - 1.0
    segs, _ = plan(counts, ncl)
    worst = 0.0
    for s in segs:
        base = red = tail = 0.0
        for e, n, k in s:
            cn = c_stage(n)
            p = math.ceil(n / 64)
            npass = -(-math.ceil(n / p) // 8) * 8
            base += 6.0 * k * cn
            red += 2.0 * k * cn * max(0.0, interp_rel(fair, npass, min(32 * k, 4096)))
            per = math.ceil(k / C)
            r = per % 4
            extra = 4.0 * (per - k / C) + (16.0 * (pr["aw"].get((min(64, max(8, npass)), r), r / 4) - r / 4) if r else 0.0)
            tail += extra * cn
        worst = max(worst, (base + red) / C + tail)
    return worst / (sum(ecost(n) for n in counts) / (ncl * C))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    arg = lambda k, d: next((a.split("=", 1)[1] for a in argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    G = int(arg("G", 132))
    pr = load_probe(arg("probe", "box_out/xcf/probe.json"))
    print(f"probe: {'loaded' if pr else 'MISSING (base plan only)'}")
    for T, L in sorted(routings().items()):
        if T < 17:
            continue
        rs = [project(c, pr, G) for c in L]
        med = lambda k: sorted(r[k] for r in rs)[len(rs) // 2]  # noqa: E731
        print(f"T={T:3d} ({len(rs)} routings): per-CTA base avg {med('avg_base'):.1f} max {med('max_base'):.1f} us | "
              f"+ B red {med('b_red_mean'):.1f} + A tail {med('a_tail_mean'):.1f} -> max CTA {med('max_total'):.1f} us | "
              f"segments/CTA {med('segs_mean'):.2f}, median segment K {med('kb_med')}")
        if "--clusters" in argv and pr:
            md = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
            print("        astra (C per expert, LPT, 128 CTAs) makespan/ideal: " + "  ".join(
                f"C{C} {md([astra(c, C) for c in L]):.2f}" for C in (4, 8, 16)))
            print("        xcf-c (cluster ranges, fair red, A share) worst/ideal: " + "  ".join(
                f"C{C} {md([xcfc(c, pr, C, n) for c in L]):.3f}" for C, n in ((1, 132), (2, 66), (4, 32), (8, 16), (16, 8))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R-t1-4 analysis: split the K1 / K2 end spread of K2A vid 0 at T = 1 / 8 into the pre-registered parts
(notes/memory/gpu_requests.md R-t1-4) from the raw per-CTA rows in box_out/t1/raw_*.json.

  python -I tools/t1_raw_split.py box_out/t1/raw_*.json

Per marked replay and kernel: end_i = max(cons_end, epi_exit, prod_exit); spread = max - p50 (and max - min). For the
latest-ending decile, (end_i - p50 end) = (1) start skew + (2) work imbalance + (3) per-SM speed + (4) older / younger
+ (5) finalize / exit + residual, with per-stage time tau_i = (cons_end - first_landed) / stages.
"""
import json
import statistics as S
import sys


def p50(v):
    v = sorted(v)
    return v[len(v) // 2]


def split(rows, F):
    ix = {f: i for i, f in enumerate(F)}
    R = []
    for r in rows:
        g = lambda f: r[ix[f]]  # noqa: E731
        if g("cons_end") is None or g("first_landed") is None or not g("stages"):
            continue
        end = max(x for x in (g("cons_end"), g("epi_exit"), g("prod_exit")) if x is not None)
        R.append({"sm": g("smid"), "slot": g("slot"), "st": g("stages"), "land": g("first_landed"), "entry": g("entry"),
                  "cons": g("cons_end"), "end": end, "tau": (g("cons_end") - g("first_landed")) / g("stages")})
    sm_tau = {}
    for r in R:
        sm_tau.setdefault(r["sm"], []).append(r["tau"])
    sm_mean = {k: S.mean(v) for k, v in sm_tau.items()}
    e50, l50, s50, t50 = p50([r["end"] for r in R]), p50([r["land"] for r in R]), p50([r["st"] for r in R]), p50([r["tau"] for r in R])
    x50 = p50([r["end"] - r["cons"] for r in R])
    late = sorted(R, key=lambda r: -r["end"])[:max(1, len(R) // 10)]
    parts = {"start": [], "work": [], "sm_speed": [], "old_young": [], "finalize": [], "total": []}
    for r in late:
        parts["start"].append(r["land"] - l50)
        parts["work"].append((r["st"] - s50) * t50)
        parts["sm_speed"].append((sm_mean[r["sm"]] - t50) * r["st"])
        parts["old_young"].append((r["tau"] - sm_mean[r["sm"]]) * r["st"])
        parts["finalize"].append((r["end"] - r["cons"]) - x50)
        parts["total"].append(r["end"] - e50)
    m = {k: S.mean(v) for k, v in parts.items()}
    m["residual"] = m["total"] - sum(m[k] for k in ("start", "work", "sm_speed", "old_young", "finalize"))
    ends = [r["end"] for r in R]
    m["max_p50"], m["max_min"] = max(ends) - e50, max(ends) - min(ends)
    slots = {}
    for r in R:
        slots.setdefault(r["slot"], []).append(r["tau"])
    m["tau_by_slot"] = {k: round(S.mean(v), 3) for k, v in sorted(slots.items())}
    m["late_slots"] = [r["slot"] for r in late]
    m["entry_spread"] = max(r["entry"] for r in R) - min(r["entry"] for r in R)
    m["land_spread"] = max(r["land"] for r in R) - min(r["land"] for r in R)
    return m


def main(paths):
    acc = {}
    for p in paths:
        for rep in json.load(open(p)):
            for k in ("k1", "k2"):
                acc.setdefault((rep["T"], k), []).append(split(rep[k], rep["fields"]))
    keys = ("max_p50", "max_min", "total", "start", "work", "sm_speed", "old_young", "finalize", "residual", "entry_spread",
            "land_spread")
    verdict = []
    for (T, k), ms in sorted(acc.items()):
        med = {x: S.median(m[x] for m in ms) for x in keys}
        share = med["start"] / med["total"] if med["total"] > 0 else 0.0
        print(f"T={T} {k} n={len(ms)}: " + " ".join(f"{x} {med[x]:+.2f}" for x in keys) + f" | start share {100 * share:.0f} %")
        print(f"      per-stage time by SM slot (first replay): {ms[0]['tau_by_slot']}; latest-decile slots {ms[0]['late_slots'][:12]}")
        if k == "k1":
            verdict.append((T, share))
    hit = [T for T, sh in verdict if sh >= 0.5]
    print("-> " + (f"late landing >= 50 % of the K1 end spread at T={hit}: design the dynamic last-stages pool" if hit
                   else "late landing < 50 % of the K1 end spread at T=1 and T=8: close the T <= 8 tails line"))


if __name__ == "__main__":
    main(sys.argv[1:])

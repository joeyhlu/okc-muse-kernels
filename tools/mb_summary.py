"""mb_summary.py: judge result of an MoE board loop check -> per-shape speedups + the in-prepare [mb] sweep lines.

    python -I tools/mb_summary.py <job> [<job> ...]
"""
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    for job in sys.argv[1:]:
        p = ROOT / "kernels" / "results" / f"{job}.json"
        if not p.exists():
            print(f"{job}: no result file yet")
            continue
        d = json.loads(p.read_text())
        r = d.get("result") or {}
        recs = r.get("records") or []
        setup = [x for x in recs if x.get("event") == "setup"]
        tests = [x for x in recs if x.get("event") == "test"]
        bad = [x for x in tests if x.get("status") != "pass"]
        print(f"# {job} {d.get('mode')} status {d.get('status')}/{r.get('status')} passed {r.get('passed')} "
              f"speedup {r.get('speedup')} prepare {setup[0].get('prepare_s') if setup else '-'} s; tests "
              f"{len(tests) - len(bad)}/{len(tests)}" + (f" FAIL {bad[0].get('case')} {bad[0].get('message')}" if bad else ""))
        if r.get("error"):
            print("  error:", str(r.get("error"))[:600])
        sl = []
        for x in recs:
            if x.get("event") != "benchmark":
                continue
            c = x.get("case") or {}
            sp = x["baseline_ns"] / x["median_ns"] if x.get("median_ns") else float("nan")
            sl.append(math.log(sp))
            print(f"  bs{c.get('bs'):<4} SGLang {x['baseline_ns'] / 1e3:7.1f} us  ours {x['median_ns'] / 1e3:7.1f} us  x{sp:.3f}")
        if sl:
            print(f"  geomean x{math.exp(sum(sl) / len(sl)):.4f} over {len(sl)} shapes")
        lt = r.get("log_tail") or ""
        for ln in lt.splitlines():
            if ln.startswith("[mb]"):
                print("  " + ln)


if __name__ == "__main__":
    main()

"""rev_judge_watch.py: per-shape regression flags between two judge results (read-only on kernels/results/*.json, stdlib).

  python -I tools/rev_judge_watch.py <job> [--vs <job>]

<job> = a job id (or unique prefix) with kernels/results/<id>.json. Default --vs: the previous done run (by finished_at) of
the same board. Prints at most ~20 lines: per-shape flags only, then prepare_s and node of both runs.
Flags per shape (record index = the board's own order):
  RATIO  paired_ratio fell by more than 2x the combined run-to-run spread (std_ns / median_ns of both runs)
  BASE   SGLang's baseline_ns moved > 3 % (judge-machine drift: normalize before calling a regression)
  TAIL   std_ns grew > 50 % (routing- / race-dependent tails)
  TEST   a correctness test's error / bound ratio rose by > 0.10 (or a test did not pass)
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "kernels", "results")


def load(job):
    hits = [p for p in glob.glob(os.path.join(RES, f"{job}*.json"))]
    if len(hits) != 1:
        sys.exit(f"rev_judge_watch: {len(hits)} result files match {job!r}")
    with open(hits[0], encoding="utf-8") as f:
        return json.load(f)


def bench(d):
    return {r["index"]: r for r in (d.get("result") or {}).get("records", []) if r.get("event") == "benchmark"}


def tests(d):
    out = {}
    for r in (d.get("result") or {}).get("records", []):
        if r.get("event") != "test":
            continue
        m = re.findall(r"error ([0-9.e+-]+) .*?\[bound ([0-9.e+-]+)\]", r.get("message", ""))
        worst = max((float(e) / float(b) for e, b in m if float(b) > 0), default=0.0)
        out[r["index"]] = (worst, r.get("status"))
    return out


def setup(d):
    for r in (d.get("result") or {}).get("records", []):
        if r.get("event") == "setup":
            return r.get("prepare_s")
    return None


def previous(d):
    best = None
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                e = json.load(f)
        except (OSError, ValueError):
            continue
        if (e.get("board") != d.get("board") or e.get("status") != "done" or e.get("id") == d.get("id")
                or (e.get("finished_at") or 0) >= (d.get("finished_at") or 0) or not bench(e)):
            continue
        if best is None or e["finished_at"] > best["finished_at"]:
            best = e
    return best


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cur = load(argv[0])
    ref = load(argv[argv.index("--vs") + 1]) if "--vs" in argv else previous(cur)
    if ref is None:
        sys.exit("rev_judge_watch: no earlier done run of this board")
    bc, br = bench(cur), bench(ref)
    tc, tr = tests(cur), tests(ref)
    rc, rr = cur.get("result") or {}, ref.get("result") or {}
    print(f"{cur.get('board')} {cur['id']} x{rc.get('speedup') or 0:.4f} vs {ref['id']} x{rr.get('speedup') or 0:.4f}")
    flags = []
    for i in sorted(set(bc) & set(br)):
        a, b = bc[i], br[i]
        spread = (a["std_ns"] / a["median_ns"] + b["std_ns"] / b["median_ns"]) / max(a.get("runs", 20), 1) ** 0.5
        dr = a["paired_ratio"] / b["paired_ratio"] - 1
        if dr < -2 * spread:
            flags.append(f"[{i:2d}] RATIO x{b['paired_ratio']:.3f} -> x{a['paired_ratio']:.3f} ({100 * dr:+.1f} %, spread {100 * spread:.1f} %)")
        db = a["baseline_ns"] / b["baseline_ns"] - 1
        if abs(db) > 0.03:
            flags.append(f"[{i:2d}] BASE  SGLang {b['baseline_ns'] / 1e3:.1f} -> {a['baseline_ns'] / 1e3:.1f} us ({100 * db:+.1f} %)")
        if a["std_ns"] > 1.5 * b["std_ns"]:
            flags.append(f"[{i:2d}] TAIL  std {b['std_ns'] / 1e3:.1f} -> {a['std_ns'] / 1e3:.1f} us (worst {a['worst_ns'] / 1e3:.1f})")
    for i in sorted(tc):
        e, st = tc[i]
        e0 = tr.get(i, (0.0, "pass"))[0]
        if st != "pass" or e - e0 > 0.10:
            flags.append(f"[t{i:2d}] TEST  error/bound {e0:.2f} -> {e:.2f} ({st})")
    for line in flags[:16]:
        print(line)
    if len(flags) > 16:
        print(f"... {len(flags) - 16} more flags")
    if not flags:
        print("no per-shape flags")
    for tag, d in (("cur", cur), ("ref", ref)):
        m = ((d.get("result") or {}).get("machine") or {})
        g = m.get("gpu_after") or {}
        print(f"{tag}: prepare_s {setup(d) or 0:.1f} node {m.get('node')} driver {g.get('driver_version')}")


if __name__ == "__main__":
    main(sys.argv[1:])

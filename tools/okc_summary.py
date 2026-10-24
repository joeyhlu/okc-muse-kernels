"""Summarize a saved okc result (kernels/results/<job>.json) into kernels/results/<job>.md and print it.

  python3 -I tools/okc_summary.py <job> [--profile-top N] [--log-grep REGEX]
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def us(ns):
    return f"{ns / 1000:.1f}" if isinstance(ns, (int, float)) else "-"


def main():
    job = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--profile-top") + 1]) if "--profile-top" in sys.argv else 5
    grep = sys.argv[sys.argv.index("--log-grep") + 1] if "--log-grep" in sys.argv else r"^\[(k1|k2a|k3)\]|LAB|FAILED|Error|error:"
    d = json.loads((ROOT / "kernels" / "results" / f"{job}.json").read_text())
    r = d.get("result") or {}
    lines = [f"# {job} {d.get('board')} {d.get('mode')} status={d.get('status')}/{r.get('status')} passed={r.get('passed')}",
             f"speedup={r.get('speedup')} ci={r.get('speedup_ci')} score_us={r.get('score_us')} baseline_us={r.get('baseline_us')}"
             f" run_s={round((d.get('finished_at') or 0) - (d.get('started_at') or 0))} error={r.get('error')}"]
    recs = r.get("records") or []
    tests = [x for x in recs if x.get("event") == "test"]
    bad = [x for x in tests if x.get("status") != "pass"]
    lines.append(f"tests: {len(tests) - len(bad)}/{len(tests)} pass")
    for x in bad[:3]:
        lines.append(f"  FAIL {x.get('case')}: {x.get('message')}")
    worst = []
    for x in tests:
        for m in re.finditer(r"error ([0-9.e+-]+) \(worst row ([0-9.e+-]+)\) \[bound ([0-9.e+-]+)\]", x.get("message") or ""):
            worst.append(float(m.group(1)) / float(m.group(3)))
    if worst:
        lines.append(f"  max error/bound over tests: {max(worst):.2f}")
    bench = [x for x in recs if x.get("event") == "benchmark"]
    if bench:
        lines.append("| load | shape | w | SGLang µs | ours µs | speedup | flags |")
        lines.append("|---|---|---|---|---|---|---|")
        for x in bench:
            c = x.get("case", {})
            shape = f"B{c.get('batch', c.get('bs', c.get('m', '')))} L{c.get('q_len', 1)} {c.get('layer', '')}".strip()
            sp = x.get("speedup") or x.get("paired_speedup")
            if sp is None and x.get("baseline_ns") and x.get("median_ns"):
                sp = x["baseline_ns"] / x["median_ns"]
            lines.append(f"| {c.get('load', '')} | {shape} | {c.get('weight', '')} | {us(x.get('baseline_ns'))} | {us(x.get('median_ns'))}"
                         f" | {sp:.3f} | {','.join(map(str, x.get('flags') or []))} |" if sp else
                         f"| {c.get('load', '')} | {shape} | {c.get('weight', '')} | {us(x.get('baseline_ns'))} | {us(x.get('median_ns'))} | - | |")
    profs = [x for x in recs if x.get("event") == "profile"]
    for x in profs:
        c = x.get("case", {})
        rep = (x.get("report") or "").splitlines()
        lines.append(f"profile {c.get('load', '')} {c.get('layer', '')}: " + " || ".join(l.strip()[:110] for l in rep[1:1 + top]))
    log = r.get("log_tail") or ""
    hits = [l for l in log.splitlines() if re.search(grep, l)]
    if hits:
        lines.append("log:")
        lines += ["  " + l[:300] for l in hits[-40:]]
    text = "\n".join(lines)
    (ROOT / "kernels" / "results" / f"{job}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

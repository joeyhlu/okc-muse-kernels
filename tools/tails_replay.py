#!/usr/bin/env bash
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export PYTHONDONTWRITEBYTECODE=1
if command -v python3 >/dev/null 2>&1; then
    exec python3 -B -I "$0" "$@"
fi
exec python -B -I "$0" "$@"
":"""
"""TAILS0: CPU-only, fail-closed audit of the saved HYB replay evidence.

Python and bash entry points share this file; no CUDA imports or subprocesses.
Default: JSON to stdout, no writes. --write-report writes only the task's MD.
--check-report verifies that the saved report equals the regenerated report.
The supplied captures have no stage samples. In that case the requested 200
Monte Carlo draws are NOT fabricated from endpoint residuals or nt averages.
Exit 0 = audit/gates passed (not TAIL-S GO); 1 = input/gate/report failure.
"""
import argparse
import ast
import bisect
import gzip
import json
import math
import re
import statistics as stats
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
REPORT = ROOT / "box_out/reset/tails_report.md"
NAMES = {8: (128, "v16s"), 9: (128, "v16f"), 10: (256, "v32s"), 11: (256, "v32f")}
COSTS = (0.0, 0.6, 1.1, 1.88)
G = 132


def require(ok, message):
    if not ok:
        raise ValueError(message)


def grep(path, pattern):
    """Stream matching lines, never import/read the whole base layer."""
    rx = re.compile(pattern)
    with path.open(encoding="utf-8") as fh:
        return [(n, line.rstrip()) for n, line in enumerate(fh, 1) if rx.search(line)]


def literal(path, name):
    hits = grep(path, rf"^{re.escape(name)}\s*=")
    require(len(hits) == 1, f"ambiguous/missing assignment: {name}")
    return ast.literal_eval(ast.parse(hits[0][1]).body[0].value)


def function_slice(path, name):
    lines, start = [], None
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.startswith(f"def {name}("):
                require(start is None, f"duplicate function {name}")
                start = n
            elif start is not None and line.strip() and not line[0].isspace():
                break
            if start is not None:
                lines.append(line)
    require(start is not None, f"missing function {name}")
    return "".join(lines)


def load_json(path):
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz"
          else path.open(encoding="utf-8")) as fh:
        return json.load(fh)


def shipped_tables():
    tables = []
    for name in ("OKC_WGSA", "OKC_WGSB"):
        hits = grep(BASE, rf"^#define {name} ")
        require(len(hits) == 1, f"ambiguous/missing {name}")
        row = [int(v) for v in hits[0][1].split(maxsplit=2)[2].split(",")]
        require(len(row) == 8 and all(0 < x < 256 for x in row), "invalid cost table")
        tables.append(row)
    for anchor in (
        "return (int)(((pb ? WGS_PB : WGS_PA)",
        "const uint32_t cs = (uint32_t)wgs_cost(pb, S.jC[lo]);",
        "wgs_range(S, nj, 0, blockIdx.x, gridDim.x, S.wgr[0], S.wgr[1]);",
        "wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);",
    ):
        require(len(grep(BASE, re.escape(anchor))) == 1, f"planner anchor changed: {anchor}")
    return tables


def wgs_cost(tables, pb, cnt):
    require(pb in (0, 1) and 1 <= cnt <= 64, "invalid phase or job size")
    return tables[pb][((cnt + 7) >> 3) - 1]


def wgs_prefix(jobs, tables, pb):
    pre = [0]
    for cnt in jobs:
        pre.append(pre[-1] + (wgs_cost(tables, pb, cnt) << (9 - pb)))
    require(pre[-1] * G < 2**32, "C++ uint32 multiplication would overflow")
    return pre


def wgs_stage_at(jobs, tables, pb, pre, x):
    require(bool(jobs) and 0 <= x <= pre[-1], "invalid cost coordinate")
    lo, hi = 0, len(jobs)
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if pre[mid] <= x:
            lo = mid
        else:
            hi = mid
    return (lo << (9 - pb)) + (x - pre[lo]) // wgs_cost(tables, pb, jobs[lo])


def wgs_ranges(jobs, tables, pb, grid=G):
    require(grid > 0, "empty grid")
    if not jobs:
        return [(0, 0)] * grid
    pre = wgs_prefix(jobs, tables, pb)
    cuts = [wgs_stage_at(jobs, tables, pb, pre, pre[-1] * c // grid) for c in range(grid)]
    cuts.append(len(jobs) << (9 - pb))
    return list(zip(cuts, cuts[1:]))


def range_hist(jobs, pb, lo, hi):
    hist = [0] * 8
    sh = 9 - pb
    for j in range(lo >> sh, min(len(jobs), ((hi - 1) >> sh) + 1) if hi > lo else lo >> sh):
        hist[((jobs[j] + 7) >> 3) - 1] += min(hi, (j + 1) << sh) - max(lo, j << sh)
    return hist


def planner_gate(tables, hybw):
    # Execute only the proven, pure CPU plan() slice; no runner imports or entry point.
    ns = {"bisect": bisect, "G": G}
    exec(compile(function_slice(ROOT / "tools/box/hybw_run.py", "plan"), "hybw_plan_slice", "exec"), ns)
    diagnostics = []
    for row in hybw["inlayer"]:
        jobs, reference = ns["plan"](row["counts"], tables[0] + tables[1])
        for pb in (0, 1):
            ranges = wgs_ranges(jobs, tables, pb)
            pre = wgs_prefix(jobs, tables, pb)
            for c, (lo, hi) in enumerate(ranges):
                require(range_hist(jobs, pb, lo, hi) == reference[pb][c][0], "planner mismatch")
                units = ((hi - 1) >> 4) - (lo >> 4) + 1 if hi > lo else 0
                require(units == reference[pb][c][1], "unit count mismatch")
            require(sum(b - a for a, b in ranges) == len(jobs) << (9 - pb), "stage coverage")
            # Explicit boundary checks, independent of the binary-search port.
            for x in {0, pre[-1], *(max(0, v - 1) for v in pre), *pre}:
                j = min(bisect.bisect_right(pre, x) - 1, len(jobs) - 1)
                expected = (j << (9 - pb)) + (x - pre[j]) // wgs_cost(tables, pb, jobs[j])
                require(wgs_stage_at(jobs, tables, pb, pre, x) == expected, "stage_at boundary")
            cut = wgs_stage_at(jobs, tables, pb, pre, pre[-1] * 94 // 100)
            tail_stages = (len(jobs) << (9 - pb)) - cut
            diagnostics.append({"T": row["T"], "shape": NAMES[row["idx"]][1], "seed": row["si"],
                                "phase": "AB"[pb], "last_6pct_stages": tail_stages,
                                "pieces_2stage": (tail_stages + 1) // 2})
    require(wgs_ranges([], tables, 0) == [(0, 0)] * G, "empty plan")
    return diagnostics


def route_dry_run():
    """Run the real _hyb_call Python against its literal state schema and CPU spies."""
    calls = []

    class Module:
        def __init__(self, name):
            self.name = name

        def okc_set_pdl(self, pdl):
            require(pdl in (0, 1), "PDL value")

        def okc_run(self, *args):
            require(args[0] is state["w"] and args[1] is state["sc"], "scratch/weights accessor")
            require(args[-3:] == (0, 32, G), "launch arguments")
            calls.append(self.name)

    state = literal(BASE, "_HYB")
    require(all(k in state for k in ("mod", "mod7", "w", "sc", "nsm", "pdl", "pdl7")), "state schema")
    state.update(mod=Module("mod"), mod7=Module("mod7"), w=object(), sc=object(), nsm=G)
    ns = {"_HYB": state, "HYB_T7": literal(BASE, "HYB_T7"), "HYB_PDL": literal(BASE, "HYB_PDL"),
          "torch": SimpleNamespace(empty=lambda *a, **k: object(), bfloat16="cpu-spy")}
    exec(compile(function_slice(BASE, "_hyb_call"), "base_hyb_call_slice", "exec"), ns)
    for pdl in (True, False):
        ns["HYB_PDL"] = pdl
        for T in (128, 256):
            ns["_hyb_call"](SimpleNamespace(shape=(T, 4096), device="cpu-spy"), object(), object())
    state["mod7"] = None
    ns["_hyb_call"](SimpleNamespace(shape=(256, 4096), device="cpu-spy"), object(), object())
    require(calls == ["mod", "mod7", "mod", "mod7", "mod"], "route changed")
    return {"status": "PASS", "calls": calls, "state_keys_from_base": sorted(state)}


def q_capture(values):
    # Exact hybpro convention: upper middle, not statistics.median for even G.
    return sorted(values)[len(values) // 2]


def endpoint_audit(raw, summary):
    out = {}
    require(set(raw) == {f"{i}|{s}" for i in NAMES for s in range(3)}, "missing/extra shape-seed")
    for idx, (T, name) in NAMES.items():
        replays = []
        for seed in range(3):
            samples = raw[f"{idx}|{seed}"]
            require(len(samples) == 5, "expected five replays per seed")
            for rows in samples:
                require(len(rows) == G, "missing CTA")
                require(all(len(r) == 32 and all(type(v) is int and v >= 0 for v in r) for r in rows), "bad row")
                require(all(0 < r[1] <= r[11] <= r[12] <= r[13] <= r[14] for r in rows), "non-monotone endpoint")
                nj = (rows[0][15] >> 32) & 65535
                require(all(((r[15] >> 32) & 65535) == nj for r in rows), "job count differs across CTAs")
                require(sum(r[15] & 65535 for r in rows) == nj * 512, "A stages missing")
                require(sum((r[15] >> 16) & 65535 for r in rows) == nj * 256, "B stages missing")
                origin = min(r[1] for r in rows)
                metrics = {}
                for label, slot in (("A", 11), ("B", 12), ("kernel", 14)):
                    values = [(r[slot] - origin) / 1000 for r in rows]
                    metrics[label] = {"min": min(values), "p50": q_capture(values), "max": max(values)}
                metrics["B_spread"] = metrics["B"]["max"] - metrics["B"]["p50"]
                replays.append(metrics)
        ends = {p: {k: stats.median(r[p][k] for r in replays) for k in ("min", "p50", "max")}
                for p in ("A", "B", "kernel")}
        err = {p: max(abs(ends[p][k] - summary[name][f"g_{p}_end_{k}"]) for k in ("min", "p50", "max"))
               for p in ("A", "B")}
        require(max(err.values()) < 1e-9, f"{name}: raw/summary mismatch {err}")
        out[name] = {"T": T, "replays": len(replays), "cta_endpoint_pairs": len(replays) * G,
                     "measured_end_us": ends, "measured_B_max_p50_us": stats.median(r["B_spread"] for r in replays),
                     "summary_reconstruction_max_error_us": err, "timing_model_fit_error_us": None}
    return out


def audit_inputs(raw, summary, hovl, hybw):
    require(set(hovl) == {f"T{t}_p{p}" for t in (128, 256) for p in (0, 1)}, "hovl schema changed; re-audit")
    expected = {"stages", "wait_w", "wait_x", "stage_clk", "stage_us", "ghz"}
    require(all(set(r) == expected and all(math.isfinite(v) and v > 0 for v in r.values())
                for group in hovl.values() for r in group.values()), "hovl schema/value changed; re-audit")
    require(len(hybw["inlayer"]) == 12, "hybw profiles missing")
    endpoints = endpoint_audit(raw, summary)
    tables = shipped_tables()
    plans = planner_gate(tables, hybw)
    return {"status": "INSUFFICIENT_DATA", "numeric_kill_rule_evaluable": False,
            "requested_draws": 200, "completed_draws": 0,
            "cpu_gates": {"planner_reference_comparisons": len(plans) * G,
                          "route_dry_run": route_dry_run(), "endpoint_summary": "PASS"},
            "shapes": endpoints, "cost_tables": tables, "plan_geometry": plans,
            "historical_fit": {p: hybw["fit"][f"info{p}"] for p in ("A", "B")},
            "sweep": [{"T": t, "share_cost_us": c, "makespan_gain_us": None,
                       "makespan_gain_90pct_us": None, "B_max_p50_reduction_us": None,
                       "B_max_p50_reduction_90pct_us": None, "verdict": "INSUFFICIENT_DATA"}
                      for t in (128, 256) for c in COSTS]}


def negative_gates(tables, raw, summary):
    """Malformed input must fail locally, without creating fixture files."""
    rejected = 0
    for thunk in (lambda: wgs_cost(tables, 0, 0), lambda: wgs_cost(tables, 1, 65),
                  lambda: wgs_ranges([1], tables, 0, 0), lambda: endpoint_audit({}, summary)):
        try:
            thunk()
        except ValueError:
            rejected += 1
    broken = dict(summary)
    broken["v16s"] = dict(summary["v16s"], g_A_end_max=summary["v16s"]["g_A_end_max"] + 1)
    try:
        endpoint_audit(raw, broken)
    except ValueError:
        rejected += 1
    require(rejected == 5, "negative gate did not reject bad input")
    return rejected


def render_report(result):
    nodes = {n.name: n for n in ast.parse(Path(__file__).read_text(encoding="utf-8")).body
             if isinstance(n, ast.FunctionDef)}
    def span(first, last=None):
        return f"{nodes[first].lineno}-{nodes[last or first].end_lineno}"
    lines = ["| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |",
             "|---|---|---|---|---|---|---|"]
    for name, r in result["shapes"].items():
        e = r["measured_end_us"]
        lines.append(f"| TAIL-S | T={r['T']} {name} | A/B max {e['A']['max']:.3f}/{e['B']['max']:.3f}; "
                     f"B max-p50 {r['measured_B_max_p50_us']:.3f} (hybpro raw) | N/A | B end precedes grid barrier; "
                     "movable stall location unknown | R-hybtab-1; R-tq-2 | INSUFFICIENT DATA |")
    lines += ["", "Kill reasons first: (1) the phase residual may accumulate across stages or doneAs waits, so shortening the last segment need not remove it; "
              "(2) added partials and finisher read-backs can consume the gain. Neither is identifiable from the saved aggregates. No implementation or speedup is proposed.", "",
              "All endpoint times below are MEASURED from `box_out/hybpro/marks_raw.json.gz`, relative to the earliest CTA wait-done timestamp "
              "in each replay: `(mark[k] - min(mark[1])) / 1000` ns/us. Each shape has 3 seeds x 5 replays x 132 CTAs. "
              "The table uses the median of each per-replay statistic; B spread is the median of per-replay differences, not a difference of medians.", "",
              "| shape | MEASURED A min / p50 / max us | MEASURED B min / p50 / max us | MEASURED kernel max us | A / B summary reconstruction error us | timing-model fit error |",
              "|---|---|---|---|---|---|"]
    for name, r in result["shapes"].items():
        e, err = r["measured_end_us"], r["summary_reconstruction_max_error_us"]
        fmt = lambda p: " / ".join(f"{e[p][k]:.3f}" for k in ("min", "p50", "max"))
        lines.append(f"| {name} | {fmt('A')} | {fmt('B')} | {e['kernel']['max']:.3f} | "
                     f"{err['A']:.6f} / {err['B']:.6f} | N/A: no stage trace |")
    lines += ["", "This reproduces the saved summaries exactly; it is a decoding check, not a fitted timing model. "
              "The capture records thread 0 phase endpoints, not independently observed endpoints of all four consumer WGs. "
              "R-hybtab-1's historical standalone residual SD is MEASURED 2.416 us (A) / 2.375 us (B), "
              "`box_out/hybw/pre.json:fit.infoA/infoB`; it is not a per-stage distribution or a per-shape replay fit.", "",
              "| T | assumed share cost us/piece | EST makespan gain us [90% interval] | EST B max-p50 reduction us [90% interval] | verdict |",
              "|---|---|---|---|---|"]
    for row in result["sweep"]:
        lines.append(f"| {row['T']} | {row['share_cost_us']:g} | N/A [N/A] | N/A [N/A] | INSUFFICIENT DATA |")
    lines += ["", "Pre-registered rule retained: at assumed 0.6 us/piece, KILL for gain <5.0 us at T=128 or <6.0 us at T=256, "
              "or B max-p50 reduction <3.0 us. These are decision thresholds, not measurements. The rule is **not evaluable**, "
              "so neither GO nor an empirical KILL is claimed. Requested Monte Carlo draws: 200; completed: 0. "
              "Sampling endpoint totals, spreading a residual uniformly over stages, or treating 2.4 us as independent stage noise would invent the central assumption.", "",
              "Missing evidence:", "",
              "- `hybpro` saves 32 u64/CTA: globaltimer slots 0..14, matching clock64 slots 16..30, and packed A/B stage counts, job count and padded-row total in slot 15. "
              "There are no per-stage durations, stage/nt identities, segment epilogues, per-job ready times, or WG-resolved samples "
              "(`tools/hybpro_patches.py:12-23,79-83`). Prefix sums of stage counts recover the contiguous range boundaries and thus A's rotation, "
              "but not the ordered job/nt identity of each stage.",
              "- `hovl/mk.json` contains only counts and mean wait-W/wait-X/stage clocks/ns by nt. The device already accumulates stages; "
              "`tools/box/hovl_run.py:160-166` then sums CTAs/WGs. A and B are combined. The keys p0/p1 are input profile numbers, not phases. "
              "No residual samples, variance, stage order, within-CTA correlation or cross-CTA covariance survive.",
              "- `hybw/pre.json` retains sorted expert counts and endpoint summaries plus fitted coefficients/SD. "
              "`tools/box/hybw_run.py:200-201,237-245` discards the busy arrays and fitting rows. Expert IDs/order, matching hybpro routing provenance, "
              "and the A-to-B readiness/finisher event graph are absent. Separate captures cannot be joined by nt alone.", "",
              "Code routes in `kernels/lab/k3_diet_rq128.py`: both T=128 and T=256 select HYB (6166,6285-6286), "
              "`okc::moe_fused<true>` (BIGM=64 at 4965; launch 6151/6155). T=128 selects `_HYB[\"mod\"]`; "
              "T=256 selects `_HYB[\"mod7\"]` when available, else `mod` (6279). The T=256 build adds HYBST=7 and HN=1 "
              "(6169-6170), while the saved hybpro st16 uses HYBST=0 (`tools/hybpro_patches.py:134`); no current-ship fit is established. "
              "Both execute `wgs_phase<false/true>` (5988/5990), dispatching `wgs_seg<PB,NTT=1..8>` by ordered `S.jC` (5855-5864). "
              "Exact per-CTA NTT instances cannot be recovered from these marks.", "",
              "Planner CPU gate ports `wgs_cost` (5294-5295), prefix/stage_at/ranges (5875-5902) and compares all stage-class histograms and unit counts "
              "with the isolated proven `tools/box/hybw_run.py:85-112` plan function on the 12 saved count profiles. "
              "This is a geometry check on historical profiles, not a timing fit to hybpro.", "",
              "| T | EST 2-stage pieces in literal last 6% cost, A | EST pieces, B | one-piece-per-CTA capacity |",
              "|---|---|---|---|"]
    for T in (128, 256):
        ranges = []
        for phase in "AB":
            v = [r["pieces_2stage"] for r in result["plan_geometry"] if r["T"] == T and r["phase"] == phase]
            ranges.append(f"{min(v)}..{max(v)}")
        lines.append(f"| {T} | {ranges[0]} | {ranges[1]} | 132 |")
    lines += ["", "EST geometry arithmetic: `cut=wgs_stage_at(floor(0.94*total_cost)); pieces=ceil((total_stages-cut)/2)`. "
              "Thus literal 6%, exactly two stages, and exactly one tail piece per CTA do not describe the same schedule on these profiles. "
              "A future replay must explicitly choose the reserved tail budget and assignment. It must also model A's partial-unit-first rotation "
              "(`wgs_split`, 5388-5390), `doneAs` gating (5464), designated finishers, and the stage-to-final-barrier chain. "
              "The planner's cost integers are scheduling weights, not microseconds.", "",
              "Cheapest sufficient capture (specification only; not run): one marks-only build per actual ship flag set, "
              "the four scored shapes with the same fixed router and 3 input seeds x 5 replays, using the existing gated hybpro vehicle. "
              "Save unsorted `jE/jS/jC/jrow`, both CTA ranges, source/flags hash and routing identity once per routing; "
              "save `(CTA, WG, phase, seq, job, unit, stage, nt, globaltimer, clock64)` at every stage start/end, "
              "segment flush entry/exit, A publication and B ready-wait entry/exit, plus existing endpoints. "
              "Use one writer per consumer WG and preallocated per-CTA buffers; keep stage samples ordered, with no device aggregation. "
              "Retain a paired unmarked control for instrumentation perturbation. Capture stage residuals by CTA/phase/nt/segment position, "
              "including contiguous samples and cross-CTA replay identity; resample paired control/TAIL-S draws only after held-out baseline endpoints fit. "
              "No TAIL-S arm, judge run, or full benchmark sweep is needed for that capture. "
              "The cited precedent is 3.9-4.4 pod-min / about $0.35-0.45 (`hybx_muse.md:56`), not a measured cost guarantee for this richer trace.", "",
              "No hot loop, prepare path, kernel generator, barrier, counter, cleanup or combine loop was changed. "
              "Device bytes/instructions before=after (change 0); no stage traffic or launch saving is claimed. "
              "The CPU dry run executes the real `_hyb_call` slice against the literal `_HYB` schema with CPU spies, "
              "covering both modules, PDL values and mod7 fallback. No base import, compilation, GPU API or subprocess occurs.", "",
              f"CPU gates PASS: {result['cpu_gates']['planner_reference_comparisons']} CTA/phase planner comparisons; "
              "7,920 endpoint pairs; exact summary checks at all four shapes; route dry run; five malformed-input rejection checks. "
              "CUDA compilation/SASS and GPU capture were not run and are inapplicable to the unchanged device code. "
              "The timing-model fit and Monte Carlo sweep remain blocked by missing samples.", "",
              "Reproduce on Windows: `python -B -I tools/tails_replay.py --check-report`. "
              "Regenerate only this report: `python -B -I tools/tails_replay.py --write-report`. "
              "The LF-only Python file is also a self-contained bash launcher and adds cuobjdump's directory to PATH. Lead's Linux command:", "",
              "```bash",
              'MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/tails_replay.py" --check-report',
              "```", "",
              "Both named files are new in full; no CUDA bits were implemented. Functional changes in `tools/tails_replay.py`: "
              f"lines 1-11 bash launcher; {span('require', 'shipped_tables')} input/source gates; "
              f"{span('wgs_cost', 'planner_gate')} planner; {span('route_dry_run')} route dry run; "
              f"{span('q_capture', 'audit_inputs')} endpoint audit and insufficient-data result; "
              f"{span('negative_gates')} negative gates; {span('render_report', 'main')} report generator/CLI. "
              "Only the two named files are written. Machine-readable results go to stdout; no third shell/JSON file is created. "
              "This report embeds the compact JSON result below to retain results within the write allowlist.", "",
              "```json", json.dumps({k: result[k] for k in ("status", "numeric_kill_rule_evaluable", "requested_draws", "completed_draws", "cpu_gates", "sweep")}, indent=2), "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-report", action="store_true")
    group.add_argument("--check-report", action="store_true")
    args = parser.parse_args()
    # compile() creates no __pycache__; all other reads are capture JSON or grep slices.
    compile(Path(__file__).read_bytes(), str(Path(__file__)), "exec")
    raw = load_json(ROOT / "box_out/hybpro/marks_raw.json.gz")
    summary = load_json(ROOT / "box_out/hybpro/marks.json")
    result = audit_inputs(raw, summary, load_json(ROOT / "box_out/hovl/mk.json"), load_json(ROOT / "box_out/hybw/pre.json"))
    result["cpu_gates"]["negative_cases"] = negative_gates(result["cost_tables"], raw, summary)
    report = render_report(result)
    if args.write_report:
        with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(report)
    if args.check_report:
        require(REPORT.read_text(encoding="utf-8") == report, "report stale: regenerate with --write-report")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, SyntaxError) as ex:
        print(json.dumps({"status": "ERROR", "error": str(ex)}), file=sys.stderr)
        sys.exit(1)

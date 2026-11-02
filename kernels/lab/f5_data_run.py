#!/usr/bin/env python3
"""F5-data: operand-DATA contrasts on the shipped mma.sync consumer (ms, NS=3) of the F3 micro.

Which data pull the SM clock down?  Same kernel, same ring/barrier protocol, same weight-stream bytes; only the bytes
in the weight bank (random vs zero) and in the x tile (random / zero / zero-padded token columns) change.
Arms (per N in {8, 16}): base, twin (A/A), w_zero, x_zero, x_pad2 (only the first 2 token columns non-zero, what a real
small-T job looks like), wx_zero; plus the null stream as the clock reference.  Gated cold-L2 protocol of F3 (flush +
300 us gate, 12 rotated rounds, 3 banks), in-kernel MHz (clock64 / globaltimer on CTA 0), and a SEPARATE continuous
>= 3 s replay per arm with nvidia-smi power/clock sampling outside the timed loop.  No correctness claims for the
zero-data arms (outputs are zero by construction); base/twin are the shipped kernel.
Run on the pod:  python3 -u -B kernels/lab/f5_data_run.py --out box_out/f5/results_data.json
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m16xc_micro as mc  # noqa: E402  (its __main__ guard keeps the unfinished F5 entry inert)

NS = 3
NS_ARMS = (8, 16)
VARIANTS = ("base", "twin", "w_zero", "x_zero", "x_pad2", "wx_zero")


def smi_sample(seconds, replay_fn, torch):
    """Replay back-to-back for `seconds` while nvidia-smi samples power/clock every 50 ms (separate process)."""
    cmd = ["nvidia-smi", "--query-gpu=power.draw,clocks.sm", "--format=csv,noheader,nounits", "-lms", "50"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        time.sleep(0.3)
        t0 = time.monotonic()
        n = 0
        while time.monotonic() - t0 < seconds:
            for _ in range(100):
                replay_fn()
            torch.cuda.synchronize()
            n += 100
    finally:
        p.terminate()
        try:
            out, _ = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            out, _ = p.communicate()
    rows = []
    for line in out.splitlines():
        try:
            a, b = (float(s.strip()) for s in line.split(","))
            rows.append((a, b))
        except ValueError:
            continue
    # drop the first and last 10 % (ramp / tail)
    k = len(rows) // 10
    core = rows[k:len(rows) - k] if len(rows) > 2 * k + 2 else rows
    if not core:
        return {"samples": 0, "launches": n}
    return {"samples": len(core), "launches": n,
            "power_w_median": statistics.median(r[0] for r in core),
            "power_w_max": max(r[0] for r in core),
            "smi_sm_mhz_median": statistics.median(r[1] for r in core)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("box_out/f5/results_data.json"))
    ap.add_argument("--units", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--power-seconds", type=float, default=3.0)
    args = ap.parse_args()
    assert args.units == 48
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"task": "F5-data", "status": "RUNNING", "rows": []}
    t_start = time.monotonic()
    try:
        import torch
        torch.set_num_threads(4)
        mod, comp = mc.build(out)
        assert comp["status"] == "PASS", comp["gates"]
        report["compiler_gates"] = comp["gates"]
        print("[f5d] build PASS", flush=True)
        prop = torch.cuda.get_device_properties(0)
        assert "H200" in prop.name and prop.multi_processor_count == 132
        timer = mc.Timer(torch)
        stream = timer.stream.cuda_stream
        empty = timer.capture(lambda: mod.empty(stream))
        grid, units = mc.GRID, args.units
        # three distinct banks (random weights); zero banks are device-side zeros of identical size
        banks = []
        for b in range(3):
            weights = mc.make_weights(math.ceil(grid * units / 512), 103 + b)
            banks.append(torch.from_numpy(mc.prepare_bank(weights, units, grid)).cuda())
        zero_banks = [torch.zeros_like(w) for w in banks]
        print("[f5d] banks ready", flush=True)
        xs = {}
        for n in NS_ARMS:
            g = torch.Generator().manual_seed(103 + n)
            x = torch.randn(n, 1024, generator=g).to(torch.bfloat16)
            bits = x.view(torch.int16).numpy().view(np.uint16)
            variants = {"base": bits.copy(), "twin": bits.copy(), "w_zero": bits.copy(),
                        "x_zero": np.zeros_like(bits), "wx_zero": np.zeros_like(bits)}
            pad = bits.copy()
            pad[2:] = 0
            variants["x_pad2"] = pad
            xs[n] = {k: torch.from_numpy(mc.pack_x(v).view(np.int16)).cuda() for k, v in variants.items()}
        output = torch.empty((grid, units, 64, 32), device="cuda", dtype=torch.float32)
        sm = torch.empty(grid, device="cuda", dtype=torch.int32)
        err = torch.empty_like(sm)
        marks = torch.empty(2, device="cuda", dtype=torch.int64)
        timer.stream.wait_stream(torch.cuda.current_stream())
        arms = []  # (name, n, variant)
        for n in NS_ARMS:
            arms.append((f"null_n{n}", n, "null"))
            for v in VARIANTS:
                arms.append((f"{v}_n{n}", n, v))
        graphs = {}
        # cudaFuncSetAttribute (dynamic smem) happens inside info(); the F3 benchmark calls it for every arm before launching
        for key in (("null", 8, NS), ("null", 16, NS), ("ms", 8, NS), ("ms", 16, NS), ("ms_twin", 16, NS)):
            mod.info(mc.ARM_ID[key[0]], key[1], key[2])
        for name, n, v in arms:
            key = ("null", n, NS) if v == "null" else (("ms_twin", 16, NS) if (v == "twin" and n == 16) else ("ms", n, NS))
            px = xs[n]["base" if v == "null" else v]
            graphs[name] = []
            for b in range(3):
                w = zero_banks[b] if v in ("w_zero", "wx_zero") else banks[b]
                graphs[name].append(timer.capture(
                    lambda key=key, w=w, px=px: mc.launch_arm(mod, key, units, grid, w, px, output, sm, err, marks, stream)))
                assert not err.any().item(), (name, b, "protocol error")
        print("[f5d] graphs captured", flush=True)
        before = statistics.median(timer.replay(empty) for _ in range(15))
        samples = {a[0]: [] for a in arms}
        mhz = {a[0]: [] for a in arms}
        names = [a[0] for a in arms]
        for rnd in range(args.rounds):
            order = names[rnd % len(names):] + names[:rnd % len(names)]
            for name in order:
                b = (2 * rnd + names.index(name)) % 3
                samples[name].append(timer.replay(graphs[name][b]))
                cyc, ns_delta = marks.cpu().tolist()
                assert cyc > 0 and ns_delta > 0, (name, "invalid loop clock")
                mhz[name].append(1000.0 * cyc / ns_delta)
        assert not err.any().item(), "device protocol error"
        after = statistics.median(timer.replay(empty) for _ in range(15))
        overhead = (before + after) / 2
        print(f"[f5d] timing done, empty-graph overhead {overhead:.3f} us", flush=True)
        rows = []
        for name, n, v in arms:
            vals = [(s - overhead) / units for s in samples[name]]
            row = {"arm": name, "n": n, "variant": v, "us_per_unit": statistics.median(vals),
                   "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90)),
                   "mhz_median": statistics.median(mhz[name]), "mhz_p10": float(np.percentile(mhz[name], 10)),
                   "mhz_p90": float(np.percentile(mhz[name], 90))}
            rows.append(row)
            print(f"[f5d] {name:14s} {row['us_per_unit']:.4f} us/unit  {row['mhz_median']:.0f} MHz", flush=True)
        report["rows"] = rows
        # continuous power runs (outside the timed protocol)
        for name, n, v in arms:
            g = graphs[name][0]
            with torch.cuda.stream(timer.stream):
                pw = smi_sample(args.power_seconds, g.replay, torch)
            for r in rows:
                if r["arm"] == name:
                    r["power"] = pw
            print(f"[f5d] power {name:14s} {pw}", flush=True)
        report["empty_us"] = {"before": before, "after": after}
        report["status"] = "COMPLETE"
    except Exception as ex:  # noqa: BLE001
        report["status"] = "FAIL"
        report["error"] = repr(ex)
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - t_start
        out.write_text(json.dumps(report, indent=2))
        write_summary(report, out.with_name("summary_data.md"))


def write_summary(report, path):
    rows = report.get("rows", [])
    L = ["# F5-data: operand-data contrasts on the shipped ms consumer (NS=3)", "",
         "Reading (pre-registered, diagnostic): a data class 'moves the clock' if its arm raises the in-kernel MHz by >= 40 vs base "
         "with |twin - base| <= 15 MHz; if x_pad2 >= 1850 MHz the micro's dense random x overstates the real clock loss at small T.",
         f"status: {report.get('status')}  elapsed {report.get('elapsed_s', 0):.0f} s", "",
         "| arm | us/unit | in-kernel MHz (p10..p90) | smi power W (median/max) | smi MHz |", "|---|---:|---|---|---:|"]
    for r in rows:
        pw = r.get("power", {})
        L.append("| %s | %.4f | %.0f (%.0f..%.0f) | %s | %s |" % (
            r["arm"], r["us_per_unit"], r["mhz_median"], r["mhz_p10"], r["mhz_p90"],
            ("%.0f/%.0f" % (pw["power_w_median"], pw["power_w_max"])) if pw.get("power_w_median") is not None else "n/a",
            ("%.0f" % pw["smi_sm_mhz_median"]) if pw.get("smi_sm_mhz_median") is not None else "n/a"))
    by = {r["arm"]: r for r in rows}
    for n in NS_ARMS:
        b, t = by.get(f"base_n{n}"), by.get(f"twin_n{n}")
        if b and t:
            L.append("")
            L.append(f"N={n}: |twin-base| = {abs(t['mhz_median'] - b['mhz_median']):.0f} MHz, "
                     f"{abs(t['us_per_unit'] - b['us_per_unit']):.4f} us/unit")
            for v in ("w_zero", "x_zero", "x_pad2", "wx_zero"):
                r = by.get(f"{v}_n{n}")
                if r:
                    L.append(f"  {v}: dMHz {r['mhz_median'] - b['mhz_median']:+.0f}, d us/unit {r['us_per_unit'] - b['us_per_unit']:+.4f}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

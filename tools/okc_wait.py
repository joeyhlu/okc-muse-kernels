"""Block until at least one more okc job finishes (or all are finished, or the time cap passes), then exit.

  python3 -I tools/okc_wait.py [max_minutes=180] [interval_s=120] [min_new=1]

Run it in the background; its exit is the wake-up signal. Uses tools/okc_wave.py poll, which saves finished
results to kernels/results/<job>.json.
"""
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "kernels" / "results"
JOBS = ROOT / "notes" / "jobs.jsonl"


def counts():
    jobs = [json.loads(l).get("job") for l in JOBS.read_text().splitlines()] if JOBS.exists() else []
    jobs = [j for j in jobs if j]
    done = [j for j in jobs if (RESULTS / f"{j}.json").exists()]
    return len(jobs), len(done)


max_min = float(sys.argv[1]) if len(sys.argv) > 1 else 180
interval = float(sys.argv[2]) if len(sys.argv) > 2 else 120
min_new = int(sys.argv[3]) if len(sys.argv) > 3 else 1
total, start_done = counts()
t0 = time.time()
while True:
    out = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "okc_wave.py"), "poll"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    total, done = counts()
    if done >= start_done + min_new or done == total or time.time() - t0 > max_min * 60:
        print(out)
        print(f"finished {done}/{total} (was {start_done}) after {(time.time() - t0) / 60:.1f} min")
        break
    time.sleep(interval)

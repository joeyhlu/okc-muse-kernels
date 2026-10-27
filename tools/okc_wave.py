"""Submit a wave of kernel check runs and poll their results.

  python3 tools/okc_wave.py submit tools/wave1.txt     # okc check --no-wait for every line (problem file)
  python3 tools/okc_wave.py poll                       # okc status for every pending job; saves results

The token is read from ~/.okc/token (mode 600) into OKC_TOKEN for the okc child process only; it is never
printed or written anywhere else. Rate limit: 6 submissions a minute per token, so one every 11 s.
Jobs are appended to notes/jobs.jsonl; results land in kernels/results/<job>.json.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
OKC = pathlib.Path.home() / ".okc" / "bin" / "okc"
TOKEN_FILE = pathlib.Path.home() / ".okc" / "token"
JOBS = ROOT / "notes" / "jobs.jsonl"
RESULTS = ROOT / "kernels" / "results"
DONE = {"done", "failed", "rejected", "duplicate"}


def okc(*args):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())}
    if TOKEN_FILE.exists():  # else okc uses its own signed-in config (okc login on this machine)
        env["OKC_TOKEN"] = TOKEN_FILE.read_text().strip()
    p = subprocess.run([str(OKC), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def submit(wave_file, mode="check"):
    lines = [l.split() for l in pathlib.Path(wave_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    for i, (problem, path) in enumerate(lines):
        if i:
            time.sleep(11)
        rc, out, err = okc(mode, problem, path, "--no-wait", "--json")
        try:
            job = json.loads(out).get("job_id") or json.loads(out).get("job")
        except Exception:  # noqa: BLE001
            job = None
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode, "problem": problem,
               "file": path, "job": job, "rc": rc, "out": out[-400:], "err": err[-400:]}
        if os.environ.get("OKC_SESSION"):  # set by tools/harness/guard.py: lets it count checks per session
            rec["session"] = os.environ["OKC_SESSION"]
        with JOBS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{problem:24s} {path:60s} -> {job or 'ERROR rc=%d %s' % (rc, (err or out)[-160:])}")


def poll():
    RESULTS.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in JOBS.read_text().splitlines()] if JOBS.exists() else []
    for r in recs:
        job = r.get("job")
        if not job or (RESULTS / f"{job}.json").exists():
            continue
        rc, out, err = okc("status", job, "--json")
        try:
            d = json.loads(out)
        except Exception:  # noqa: BLE001
            print(f"{job} status rc={rc} {(err or out)[-160:]}")
            continue
        status = d.get("status")
        if status in DONE:
            (RESULTS / f"{job}.json").write_text(json.dumps(d, indent=1))
        sp = d.get("speedup") or (d.get("result") or {}).get("speedup")
        print(f"{job} {r['problem']:24s} {pathlib.Path(r['file']).name:28s} {status} speedup={sp}")


if __name__ == "__main__":
    if sys.argv[1] == "submit":
        submit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "check")
    else:
        poll()

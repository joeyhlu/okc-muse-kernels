"""Print finished-but-unsummarized jobs, summarizing each (speedup table only)."""
import json, pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
seen = {p.stem for p in (ROOT / "kernels" / "results").glob("*.md")}
for l in (ROOT / "notes" / "jobs.jsonl").read_text().splitlines():
    r = json.loads(l); j = r.get("job")
    if j and (ROOT / "kernels" / "results" / f"{j}.json").exists() and j not in seen:
        print(f"=== {j} {r['file']}")
        out = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "okc_summary.py"), j], capture_output=True, text=True).stdout
        print("\n".join(x[:220] for x in out.splitlines() if not x.startswith(("profile", "  ptxas", "      ", "log:"))))

#!/bin/bash
# poll the judge until job $1 is no longer pending (max ~20 min)
cd "<repo>" || exit 1
for i in $(seq 1 40); do
  out=$(python3 -I tools/okc_wave.py poll 2>&1)
  if echo "$out" | grep "$1" | grep -q -i -E "running|queued|pending|started|received|submitted"; then sleep 30; else break; fi
done
echo "$out" | tail -8
ls kernels/results/ | grep "$1" && python3 - "$1" <<'PY'
import json, sys, glob
for f in glob.glob(f"kernels/results/{sys.argv[1]}*.json"):
    d = json.load(open(f))
    s = d.get("summary") or d
    print(f, {k: s.get(k) for k in ("status", "passed", "speedup", "speedup_ci")} if isinstance(s, dict) else str(s)[:300])
PY

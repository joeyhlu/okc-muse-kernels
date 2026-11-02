#!/bin/bash
# R-qfix local batch (WSL, 3070, $0): Q/P rounding attribution + policy simulation for the q_len 8 loads. Logs: notes/study/qfix_<load>_<layer>.log
cd "$(dirname "$0")/.."
source tools/local/cudaenv.sh >/dev/null 2>&1
run() {   # $1 load  $2 layer  $3 reps
  python3 -u tools/qfix_noise.py "$1" "$2" "$3" > "notes/study/qfix_${1// /}_$2.log" 2>&1
  echo "[qfix-run] $1 $2 x$3 rc $?"
}
run "verify 1" full ${QF_V1:-300}
run "verify 16" full ${QF_V16:-150}
run "verify 16" sliding ${QF_V16:-150}
run "verify 32" full ${QF_V32:-80}
run "verify 32" sliding ${QF_V32:-80}
echo "[qfix-run] done"

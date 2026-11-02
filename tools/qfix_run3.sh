#!/bin/bash
# R-qfix local batch 3 (WSL, 3070, $0): the simulated policy on EXACT scores (variant T0) = the simulator's own failure floor (distinct-experts accounting), v1f and v16s.
cd "$(dirname "$0")/.."
source tools/local/cudaenv.sh >/dev/null 2>&1
export QF_SIM_EXTRA="T0 sanity (bf16,exact)"
python3 -u tools/qfix_noise.py "verify 1" full ${QF_V1:-300} > notes/study/qfix_verify1_full_t0.log 2>&1; echo "[qfix-run3] v1f rc $?"
python3 -u tools/qfix_noise.py "verify 16" sliding ${QF_V16:-100} > notes/study/qfix_verify16_sliding_t0.log 2>&1; echo "[qfix-run3] v16s rc $?"
echo "[qfix-run3] done"

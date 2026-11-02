#!/bin/bash
# R-qfix local batch 2 (WSL, 3070, $0): the same emulation with one extra distinct-experts cushion in the simulated policy (QF_CUSH=1) at the loads that showed distinct-floor failures.
cd "$(dirname "$0")/.."
source tools/local/cudaenv.sh >/dev/null 2>&1
export QF_CUSH=1
python3 -u tools/qfix_noise.py "verify 1" full ${QF_V1:-300} > notes/study/qfix_verify1_full_cush1.log 2>&1; echo "[qfix-run2] v1f rc $?"
python3 -u tools/qfix_noise.py "verify 16" sliding ${QF_V16:-150} > notes/study/qfix_verify16_sliding_cush1.log 2>&1; echo "[qfix-run2] v16s rc $?"
echo "[qfix-run2] done"

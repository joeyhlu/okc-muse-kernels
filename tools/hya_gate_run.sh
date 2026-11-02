#!/bin/bash
# local compile gate (WSL, $0): HYA module of the base file and of the Q2 lab file for sm_90a, regs / spills side by side
cd "$(dirname "$0")/.."
source tools/local/cudaenv.sh >/dev/null 2>&1
python3 -u tools/hya_compile_gate.py kernels/muse_decode_layer/k3_fin2_c1.py kernels/lab/k3_q2_ab.py

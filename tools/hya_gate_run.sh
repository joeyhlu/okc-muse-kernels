#!/bin/bash
# local compile gate (WSL, $0): HYA module of the base file and of the patched lab files for sm_90a; logs box_out/hya_gate_*.log, then the parsed table
cd "$(dirname "$0")/.."
source tools/local/cudaenv.sh >/dev/null 2>&1
mkdir -p box_out
python3 -u tools/hya_compile_gate.py kernels/muse_decode_layer/k3_fin2_c1.py > box_out/hya_gate_base.log 2>&1 &
python3 -u tools/hya_compile_gate.py kernels/lab/k3_q2p2_ab.py > box_out/hya_gate_q2.log 2>&1 &
wait
python3 tools/hya_gate_parse.py box_out/hya_gate_base.log box_out/hya_gate_q2.log
grep -ci serializ box_out/hya_gate_q2.log

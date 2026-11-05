#!/usr/bin/env bash
# Compile/SASS/CPU only. No CUDA context, kernel launch, pod, judge, or deletion.
# Lead command:
# MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash \
#   "<repo>/tools/local/cudaenv.sh" bash \
#   "<repo>/tools/s7_ks_gate.sh"
set -euo pipefail
S7_KS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CUDA_HOME:-}" != /root/cudaenv/lib/python3.14/site-packages/nvidia/cu13 ]]; then
    exec bash "$S7_KS_ROOT/tools/local/cudaenv.sh" bash "$S7_KS_ROOT/tools/s7_ks_gate.sh" "$@"
fi
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export PYTHONDONTWRITEBYTECODE=1
python -B -I - "$S7_KS_ROOT" "$@" <<'PY'
import collections
import hashlib
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(sys.argv[1])
report_path = root / 'tools/s7_ks_gate.json'
result = {'status': 'FAIL', 'cuda_compile': 'NOT RUN', 'sass': 'NOT RUN',
          'gpu_correctness': 'NOT RUN (compile-only gate; no CUDA context)'}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def run(command):
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = proc.stdout + proc.stderr
    if proc.returncode:
        raise RuntimeError({'command': command, 'returncode': proc.returncode, 'output': text[-24000:]})
    return text


TARGETS = {'d16s': 'ks3_kernelILi16ELi1E', 'd32s': 'ks3_kernelILi8ELi0E',
           'fallback16': 'ks3_kernelILi16ELi0E', 'fallback4': 'ks3_kernelILi4ELi0E'}


def sass_functions(text):
    parsed, current = {}, None
    for line in text.splitlines():
        m = re.search(r'Function\s*:\s*(\S+)', line)
        if m:
            current = m.group(1)
            parsed[current] = []
        m = re.match(r'\s*/\*[0-9a-fA-F]+\*/\s*(.*?)\s*(?:/\*.*)?$', line)
        if current is not None and m:
            instruction = m.group(1).strip()
            if instruction:
                parsed[current].append(instruction)
    selected = {}
    for label, signature in TARGETS.items():
        matches = [(name, ins) for name, ins in parsed.items() if signature in name]
        check(len(matches) == 1, f'{label}: expected one SASS function, got {len(matches)}')
        name, ins = matches[0]
        check(ins, f'{label}: empty SASS')
        opcodes = collections.Counter()
        for instruction in ins:
            op = re.sub(r'^@\S+\s+', '', instruction).split()[0].split('.')[0]
            opcodes[op] += 1
        selected[label] = dict(function=name, instructions=ins, count=len(ins),
                               opcode_counts=dict(sorted(opcodes.items())),
                               wait_indices=[i for i, x in enumerate(ins) if 'ACQBULK' in x],
                               gather_indices=[i for i, x in enumerate(ins) if 'LDGSTS' in x],
                               ldg_indices=[i for i, x in enumerate(ins) if re.search(r'\bLDG\.', x)])
    return selected


def resources(log):
    props = list(re.finditer(r'Function properties for\s+[\'\"]?([^\s\'\"]+)', log))
    selected = {}
    for label, signature in TARGETS.items():
        matches = [(i, m) for i, m in enumerate(props) if signature in m.group(1)]
        check(len(matches) == 1, f'{label}: missing/duplicate ptxas resource report')
        i, m = matches[0]
        section = log[m.end():props[i+1].start() if i+1 < len(props) else len(log)]
        spill = re.search(r'(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*(\d+) bytes spill loads', section)
        regs = re.search(r'Used\s+(\d+)\s+registers', section)
        check(spill is not None and regs is not None, f'{label}: unrecognized ptxas report')
        stack, stores, loads = map(int, spill.groups())
        selected[label] = dict(registers=int(regs.group(1)), stack_bytes=stack,
                               spill_store_bytes=stores, spill_load_bytes=loads)
    return selected


def parser_gate():
    # A missing resource line must fail; a nonzero spill must remain nonzero.
    log = ''.join(f"ptxas info : Function properties for _{sig}_test\n"
                  "  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
                  "ptxas info : Used 120 registers\n" for sig in TARGETS.values())
    parsed = resources(log)
    check(len(parsed) == 4 and all(x['registers'] == 120 for x in parsed.values()), 'resource parser self-check')
    bad = resources(log.replace('0 bytes spill stores', '16 bytes spill stores', 1))
    check(bad['d16s']['spill_store_bytes'] == 16, 'spill parser masked a nonzero spill')
    try:
        resources('')
    except AssertionError:
        pass
    else:
        raise AssertionError('empty resource log accepted')
    text = ''.join(f'Function : _{sig}_test\n'
                   ' /*0000*/ LDG.E R2, [R4]; /* 0x0000 */\n'
                   ' /*0010*/ ACQBULK; /* 0x0000 */\n'
                   ' /*0020*/ LDGSTS.E [R8], [R10]; /* 0x0000 */\n'
                   for sig in TARGETS.values())
    check(all(x['count'] == 3 and x['wait_indices'] == [1] and x['gather_indices'] == [2]
              for x in sass_functions(text).values()), 'SASS parser self-check')


try:
    check(len(sys.argv) <= 3 and (len(sys.argv) == 2 or sys.argv[2] == '--cpu-only'), 'only --cpu-only is accepted')
    api = runpy.run_path(str(root / 'tools/s7_ks_patches.py'))
    base_raw = api['BASE'].read_bytes()
    row = base_raw.decode('ascii')
    candidate = api['OUT'].read_bytes().decode('ascii')
    bits = api['literal'](candidate, 'S7_KS_BITS')
    check(api['patch'](row, bits) == candidate, 'candidate is not the deterministic generator output')
    result.update(api['cpu_gates'](row, candidate))
    parser_gate()
    result['compile_report_parsers'] = 'PASS (synthetic positive/negative CPU cases)'
    for selector in range(4):
        variant = api['patch'](row, selector)
        check(api['literal'](variant, 'S7_KS_BITS') == selector, 'generator arm selection')
        check(api['device_unit'](variant, selector).count('template __global__ void ks3::ks3_kernel<') == 4,
              'compile unit is missing an explicit instance')
    result['generator_arms'] = 'PASS (0, 1, 2, 3 generated in memory only)'
    if '--cpu-only' not in sys.argv[2:]:
        check(shutil.which('nvcc') is not None, 'nvcc not on PATH')
        check(shutil.which('cuobjdump') is not None, 'cuobjdump not on PATH')
        result['nvcc_version'] = run(['nvcc', '--version'])
        work = Path(tempfile.mkdtemp(prefix='s7_ks_gate_'))
        result['work_dir'] = str(work)  # Keep all artifacts; no automatic deletion.
        result['compile_arms'] = {}
        for name, text, selector in [('base', row, 0), ('off', candidate, 0),
                                     ('bit1', candidate, 1), ('bit2', candidate, 2), ('both', candidate, 3)]:
            cu, cubin = work / (name + '.cu'), work / (name + '.cubin')
            cu.write_bytes(api['device_unit'](text, selector).encode('ascii'))
            command = ['nvcc', '-arch=sm_90a', '-std=c++17', '-O3', '-lineinfo', '-Xptxas', '-v',
                       '-cubin', str(cu), '-o', str(cubin)]
            arm = {'command': command, 'status': 'FAIL'}
            result['compile_arms'][name] = arm
            log = run(command)
            (work / (name + '.ptxas.log')).write_bytes(log.encode('utf-8'))
            check(not re.search(r'\bC75\d\d\b', log), f'{name}: ptxas C75xx diagnostic')
            arm['resources'] = resources(log)
            sass = run(['cuobjdump', '-sass', str(cubin)])
            (work / (name + '.sass')).write_bytes(sass.encode('utf-8'))
            arm['sass'] = sass_functions(sass)
            # All emitted KS3 specializations must pass the zero-spill gate.
            for label, r in arm['resources'].items():
                check(r['spill_store_bytes'] == 0 and r['spill_load_bytes'] == 0,
                      f'{name}/{label}: spills {r}')
                check(not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)', x)
                              for x in arm['sass'][label]['instructions']), f'{name}/{label}: local-memory SASS')
            arm['status'] = 'PASS'
            print(f'{name}: compiled, zero spill stores/loads; SASS collected', flush=True)
        arms = result['compile_arms']
        for label in TARGETS:
            check(arms['base']['sass'][label]['instructions'] == arms['off']['sass'][label]['instructions'],
                  f'{label}: default-off SASS differs from original')
        result['off_sass_equals_base'] = True
        # The early-Q arm moves exactly one existing PDL wait before the gather.
        for name in ('bit2', 'both'):
            d32 = arms[name]['sass']['d32s']
            check(d32['wait_indices'] and d32['gather_indices'], f'{name}: missing PDL/gather instructions')
            check(min(d32['wait_indices']) < min(d32['gather_indices']), f'{name}: early wait was not emitted')
        result['sass_instruction_counts'] = {
            label: {name: arm['sass'][label]['count'] for name, arm in arms.items()} for label in TARGETS}
        result['sass_review_note'] = ('Static instruction counts are not elapsed time; inspect the kept .sass files '
                                      'for load dataflow. No speed or numerical claim follows from this gate.')
        result['cuda_compile'] = 'PASS: sm_90a, base/off/bit1/bit2/both, 0 spills'
        result['sass'] = 'PASS: original == off; d32 early-wait ordering; all target dumps retained'
        result['status'] = 'CPU/COMPILE/SASS PASS; GPU NUMERICS/TIMING NOT RUN'
    else:
        result['status'] = 'CPU PASS; CUDA/SASS/GPU NOT RUN'
    check(api['BASE'].read_bytes() == base_raw, 'immutable base changed')
except Exception as exc:
    result['status'] = 'FAIL'
    result['error'] = repr(exc)
finally:
    report_path.write_bytes((json.dumps(result, indent=2) + '\n').encode('ascii'))
    print(json.dumps({k: result.get(k) for k in ('status', 'cuda_compile', 'sass', 'error', 'work_dir')}, indent=2))
sys.exit(1 if result['status'] == 'FAIL' else 0)
PY

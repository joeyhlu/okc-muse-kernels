#!/usr/bin/env python3
"""P5 SEG: compile-time, timing-only VML segment-end bound; never submit.

Kill risks first: (1) the removed work may be covered by ring/other-WG slack;
(2) compilation may change occupancy/spills or serialize WGMMA (hard gate).
No saving is asserted. MEASURED R-lat32-1 (notes/lab.md:569): A end 2.2 us,
3.4 segments/CTA at bs32, marks inflate 9-11%. EST gross end-work envelope
= 3.4*2.2 = 7.48 us/CTA, BEFORE unmeasured overlap; not a kernel saving.
MEASURED fnc experiment (task): acq_rel 3.744 vs relaxed 0.352 us under
3*36 KiB inflight; EST contrast 3.392 us, not additive or transferable here.
Only exposed resolver -> epilogue -> doneA -> B dependencies can save time.

Exact base route: mb_layer.py:6187-6188, bs25/32/40 -> bf2; defines:5991
VM_BRED=1, VM_DEFER=1, VM_SYNC7=1, VSE_PS=1. vm_main:5565 -> phase:5521
-> seg<PB,N>, N=8*ceil(job_count/8). N<=16 uses stage_m, above uses stage.
PB=true split segments return through bred:5357, so these edits target A.
The brief's 5439-5443 loop is INACTIVE in bf2: live VSE_PS read/zero is
5431-5434 -> vse_take:4817-4824. SEG1 removes BOTH forms, retaining fc=0.

Bytes/instructions (source/PTX operations, EST; SASS counts need gate run):
* All stage loops unchanged: 36,864 weight bytes/stage, zero extra ops/thread.
* SEG0: one fence/participating resolver thread -> zero; payload bytes same.
* SEG1: same fence cut. Let L=N/8 and R=number of vse_take calls (0..3),
  Z=number with M!=0 (0..1). Per resolver thread: 16*L*(R+Z) bytes -> 0,
  L*R vector loads + L*Z vector zero stores + 4*L*Radd adds -> 0; M=2 uses
  4*L assignments instead of adds. Per 128-thread WG: 2048*L*(R+Z) bytes.
  Fallback (!VSE_PS) loop: 32*L bytes/thread, L vector loads + L vector
  stores + 4*L assignments -> 0. Partial publication stores/reductions stay:
  16*L bytes/thread, one vector store/reduction per nt, before=after.
* SEG2: wt0 one acq_rel atomic -> one relaxed atomic, same 4-byte counter
  RMW; other 127 threads unchanged; fence and all payload loops unchanged.
  No stream grows. No launch, wait, release or tensor instruction is added.

Protocol coverage, all arms, all three consecutive replays:
* Every vm_main CTA 0..G-1 keeps its producer and consumer lifetime. Consumers
  tid=128*q..128*q+127 (q=0..3) keep both bar_sync(1+q,128) around the
  split-count increment. wt0 alone increments and publishes T.wlast[q].
  Sum of disjoint s1-s0 remains 16; high VSE_PS bits cannot affect low 8 bits.
  Exactly the last arriver resolves; every other WG returns at the OLD point.
* Resolver's wt0 still zeros fc (both preprocessor paths); all 128 threads
  run epilogue<false,N>. Deferred doneA publication: pend_flush's proxy fence,
  128-thread barrier, wt0 release-add and pd.i=-1 survive, including phase
  end flush:5563. All xready producers, ring arrivals and B waits survive.
* SEG0/SEG2 may read stale partials: numeric correctness is NOT claimed.
  SEG2 keeps atomic indivisibility and values; ordering removed applies only
  to numeric partial visibility. No numeric value controls routing or waits.
* SEG1 leaves stale fix floats (including accumulating vr=2), deliberately;
  they are no longer read by any active split resolver in this build. They
  are NOT control state. No reset launch is added to distort the bound.
* vm_fin CTA0 tid0..255 retains stride256 resets: xready[0:80],
  doneA[0:1280], cnt[0:1088]. VM_DP=0: gbar reset is inactive. vm_fin CTAs
  t=0..M-1 tid0..255 retain local-pair combine and y clearing: columns
  1024*m+4*tid+[0:4], m=0..3, exactly [0:4096]. rmap is rebuilt by route.
  No new exit can bypass any cleanup. Replays on one ordered stream start
  with all live control words reset; GPU replays must still validate this.

The CPU dry run executes the actual base bank constructor, allocator,
dispatch and new module preparation with strict shape-only CUDA substitutes.
It proves Python paths/coverage, not CUDA memory-order behavior or performance.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

from bt4_patches import environment

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
DEST = ROOT / 'kernels/muse_moe_w4a16/mb_layer_seg.py'

FENCE = 'SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\nfence_acq_rel();\n'
PS_READ = '''{
const int lw = T.wlast[q];
float* f2 = A.fix + ((size_t)((2 * SLOTS + slot) * 4 + q)) * FIXW + wt * 4;
if (vr == 2) vse_take<N, 2>(f2, d);
if (vr != 0 && (lw & 256)) vse_take<N, 0>(A.fix + ((size_t)(slot * 4 + q)) * FIXW + wt * 4, d);
if (vr != 1 && (lw & 512)) vse_take<N, 0>(A.fix + ((size_t)((SLOTS + slot) * 4 + q)) * FIXW + wt * 4, d);
if (vr != 2 && (lw >> 10)) vse_take<N, 1>(f2, d);
}
'''
OLD_READ = '''#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
*reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
d[4 * nt] = v.x; d[4 * nt + 1] = v.y; d[4 * nt + 2] = v.z; d[4 * nt + 3] = v.w;
}
'''
OLD_BLOCK = '#else\n' + OLD_READ + 'if (wt == 0) *fc = 0;\n#endif\n'
ATOM = '''__device__ __forceinline__ int seg_relaxed_add(int* p, int v) {
int o;
asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\\n" : "=r"(o) : "l"(p), "r"(v) : "memory");
return o;
}
'''
ATOMIC_LINES = (
    'const int r_x = atom_acqrel_add(fc, (s1 - s0) | (256 << vr));',
    'const int r_x = atom_acqrel_add(fc, s1 - s0);',
    'if (wt == 0) T.wlast[q] = (atom_acqrel_add(fc, s1 - s0) + (s1 - s0) == 16);',
)
EDITS = {
    'base': [],
    'SEG0': [(FENCE, FENCE.replace('fence_acq_rel();', '// SEG0: resolver fence omitted.'))],
    'SEG1': [(FENCE, FENCE.replace('fence_acq_rel();', '// SEG1: resolver fence omitted.')),
             (PS_READ, '// SEG1: live VSE_PS read/zero omitted; fc reset follows.\n'),
             (OLD_BLOCK, '#else\n// SEG1: fallback read/zero omitted; fc reset follows.\nif (wt == 0) *fc = 0;\n#endif\n')],
    'SEG2': [('__device__ __forceinline__ void red_add_v4(', ATOM + '__device__ __forceinline__ void red_add_v4(')] +
            [(s, s.replace('atom_acqrel_add', 'seg_relaxed_add')) for s in ATOMIC_LINES],
}

EXTENSION = r'''
# SEG_BEGIN -- generated compile-time timing probes, never submit probe outputs.
_SEG_EDITS = @EDITS@
SEG_SHAPES = (25, 32, 40)
SEG_ARMS = ('base', 'base_twin', 'SEG0', 'SEG0_twin', 'SEG1', 'SEG1_twin', 'SEG2', 'SEG2_twin')
_SEG_STATES = {}

def seg_sources(name):
 if name not in SEG_ARMS:
  raise ValueError(name)
 arm = name.removesuffix('_twin')
 src = _vml_CUDA
 for old, new in _SEG_EDITS[arm]:
  assert src.count(old) == 1, (arm, old[:90])
  src = src.replace(old, new, 1)
 return src, _vml_CPP, list(_vml_FLAGS) + list(_VML_DEFS['bf2'])

def seg_prepare_module(shared, name, deadline):
 # Prepare each compiled module ONCE; share immutable packed weights, not scratch.
 if name not in SEG_ARMS or name in _SEG_STATES:
  raise ValueError(('unknown or already prepared', name))
 for m in SEG_SHAPES:
  assert _mbl_candidates(m) == ('bf2',) and shared['route'][m] == 'bf2'
 _mbl_deadline(deadline, 'SEG total prepare')
 if name == 'base':
  arm = shared
 else:
  src, cpp, flags = seg_sources(name)
  tag = hashlib.sha1((cpp + src + ' '.join(flags)).encode()).hexdigest()[:12]
  # Distinct twin name means a separately compiled/loaded module with identical CUDA.
  mod = _mbl_load_inline(name='vml_moe_seg_' + name.lower() + '_' + tag,
                         cpp_sources=cpp, cuda_sources=src, functions=['vml_run', 'vml_cfg'],
                         extra_cuda_cflags=flags, verbose=False)
  assert _vml_cfg(mod) == _vml_cfg(shared['mods']['bf2'])
  arm = dict(shared)
  arm['mods'] = dict(shared['mods'], bf2=mod)
  dev = shared['out'][32].device
  arm['vml_sc'] = dict(shared['vml_sc'], bf2=_vml_alloc(dev, shared['nsm']))
  output = torch.empty((320, 4096), dtype=torch.bfloat16, device=dev)
  arm['out'] = {m: output[:m] for m in range(1, 321)}
 _mbl_deadline(deadline, 'SEG module prepared')
 _SEG_STATES[name] = arm
 return arm

def seg_arm(states, name, bs):
 if name not in SEG_ARMS or bs not in SEG_SHAPES:
  raise ValueError((name, bs))
 arm = states[name]
 assert arm['route'][bs] == 'bf2'
 assert arm['banks'][0]['k2a33']['st'] is states['base']['banks'][0]['k2a33']['st']
 return arm
# SEG_END
'''


def generate(base):
    # Append only: original definitions, runtime sources and bf2 flags stay exact.
    out = base + '\n' + EXTENSION.replace('@EDITS@', repr(EDITS))
    ast.parse(out)
    return out


def extension_env(source):
    env = environment(source)
    exec(compile(source[source.index('# SEG_BEGIN'):], '<SEG extension>', 'exec'), env)
    return env


def changes(base, generated):
    env = extension_env(generated)
    src = env['seg_sources']('base')[0]
    answer = {}
    for arm, edits in EDITS.items():
        rows = []
        for old, new in edits:
            assert base.count(old) == 1
            start = base[:base.index(old)].count('\n') + 1
            rows.append({'base_file': str(BASE.relative_to(ROOT)), 'base_lines': [start, start + len(old.splitlines()) - 1],
                         'before': old, 'after': new})
        probe = env['seg_sources'](arm)[0]
        rows.append({'rendered_cuda_diff': list(difflib.unified_diff(src.splitlines(), probe.splitlines(),
                                                                    fromfile='base.cu', tofile=arm+'.cu', n=2))})
        answer[arm] = rows
    return answer


def protocol_model():
    """Exhaust all split compositions; atomic algebra only, not a GPU simulator."""
    compositions = 0
    for mask in range(1, 1 << 15):
        ends = [0] + [k for k in range(1,16) if mask & (1 << (k-1))] + [16]
        packets = [(b-a, 0 if a==0 else 1 if b==16 else 2) for a,b in zip(ends,ends[1:])]
        counter = 0
        for replay in range(3):
            # Different arrival orders across three invocations, no reset helper.
            order = packets if replay==0 else packets[::-1] if replay==1 else packets[1:]+packets[:1]
            resolvers = 0
            for length, vr in order:
                old = counter
                counter += length | (256 << vr)
                if (old & 255) + length == 16:
                    resolvers += 1
                    counter = 0  # retained wt0 store, before unchanged epilogue
            assert resolvers == 1 and counter == 0
        compositions += 1
    return {'split_compositions':compositions, 'consecutive_replays':3,
            'one_resolver_each_replay':True, 'counter_rearmed':True,
            'limitation':'atomic counts only; CUDA ordering still requires GPU checks'}


def cpu_gates(base, generated):
    assert generated == generate(base), 'generated artifact drift'
    assert generated.startswith(base), 'base prefix edited'
    a, b = environment(base), environment(generated)
    # All existing strings, including every CUDA/C++ global, and every define bank.
    keys = [k for k, v in a.items() if isinstance(v, str)]
    hashes = {}
    for key in keys:
        assert a[key] == b[key], key
        hashes[key] = hashlib.sha256(a[key].encode()).hexdigest()
    assert a['_VML_DEFS'] == b['_VML_DEFS']
    env = extension_env(generated)
    src, cpp, flags = env['seg_sources']('base')
    assert src == a['_vml_CUDA'] and cpp == a['_vml_CPP']
    assert flags == a['_vml_FLAGS'] + list(a['_VML_DEFS']['bf2'])
    assert '-DVSE_PS=1' in flags and '-DVM_BRED=1' in flags
    # Protected protocol and hot-loop spans must be literally unchanged.
    spans = [('__device__ __forceinline__ void pend_flush(', '#ifndef VCP_NB'),
             ('__device__ __forceinline__ void stage_m(', 'template <bool PB, int N>'),
             ('__device__ __forceinline__ void epilogue(', '#if VM_BRED'),
             ('template <bool PB>\n#if VM_DP\n__device__ __forceinline__ void phase(', '\n}\nvoid vml_run(')]
    rendered = {}
    for arm in env['SEG_ARMS']:
        p, c, f = env['seg_sources'](arm)
        assert c == cpp and f == flags
        for begin, end in spans:
            assert begin in src and end in src.split(begin, 1)[1], (begin, end)
            assert p.split(begin, 1)[1].split(end, 1)[0] == src.split(begin, 1)[1].split(end, 1)[0], (arm, begin)
        for token in ('if (wt == 0) *fc = 0;', 'bar_sync(1 + q, 128);', 'red_release_add(', 'mbar_arrive(', 'return;'):
            assert p.count(token) == src.count(token), (arm, token)
        if arm.startswith('SEG1'):
            assert PS_READ not in p and OLD_BLOCK not in p
        if arm.startswith('SEG2'):
            assert FENCE in p and all(s.replace('atom_acqrel_add','seg_relaxed_add') in p for s in ATOMIC_LINES)
        rendered[arm] = hashlib.sha256(p.encode()).hexdigest()
        assert env['seg_sources'](arm.removesuffix('_twin')) == (p, c, f)
    for count in (80, 1280, 1088):
        assert sorted(i for t in range(256) for i in range(t, count, 256)) == list(range(count))
    assert sorted(1024*m+4*t+k for m in range(4) for t in range(256) for k in range(4)) == list(range(4096))
    return {'status': 'PASS', 'base_prefix_sha256': hashlib.sha256(base.encode()).hexdigest(),
            'unchanged_string_sha256': hashes, 'rendered_cuda_sha256': rendered,
            'original_defines_equal': True, 'protocol_source_coverage': True,
            'split_count_model': protocol_model(),
            'live_readback': 'VSE_PS=1, base lines 5428-5435 (also skip inactive 5438-5443)',
            'gpu_compile_and_replay': 'NOT RUN'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--changes', action='store_true')
    args = parser.parse_args()
    base = BASE.read_text(encoding='utf-8')
    generated = generate(base)
    if args.check:
        assert DEST.read_bytes() == generated.encode(), 'regenerate mb_layer_seg.py'
    else:
        DEST.write_bytes(generated.encode())  # LF even on Windows.
    result = cpu_gates(base, generated)
    if args.changes:
        result['per_bit_changes'] = changes(base, generated)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

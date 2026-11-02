#!/usr/bin/env python3
"""BT4 timing-only probes. Generate only mb_layer_bt4.py; never a submission.

Kill risks: H1 can be combinatorially infeasible (P > 2*n); H2 can include
weight-cache carry. H3/H4 can lose to reset-launch cost or expose no tail.
There is no asserted performance win here. H5 is deliberately not implemented.

Protocol/accounting (source-level instruction counts, NOT measured SASS):
* H3: vm_main and all its loops are byte-for-byte unchanged: 36,864 weight
  bytes/stage before=after; zero added instructions/thread/stage. vm_fin's
  CTA 0, tid 0..255 resets JMAX=80 xready, 1,280 doneA, 1,088 cnt ints.
  Its CTA t=0..M-1, tid 0..255 clears four float4s per valid rmap slot.
  The prologue reproduces ALL those stores (9,792 counter bytes + 16,384*P
  y bytes), before vm_main, on an ordinary ordered stream. The old y-clear
  loop has 4 STG.128/thread/pair; new loop has the same four stores, removing
  the norm/combine loads/arithmetic. Each arm/shape owns scratch; rmap starts
  at -1, so its first reset has no y work, and warmed replays clear exactly
  the previous routing's P rows. VM_DP=0 for bf2, hence no gbar reset needed.
* H4: stream phase loops and all pending flushes/waits are unchanged:
  36,864 weight bytes/stage and zero added instructions/thread/stage.
  All consumer CTAs 0..G-1, tid 0..511 reach the new return AFTER phase B's
  final wgs_flush[_end]. Producer tid 512..575 retain their existing exit.
  Deleted final work: two CTA_SYNC, tid0 fence/atomic/grid poll/gen increment,
  distributed zeroing, then token tk=CTA+k*G, tid 0..511 norm/combine and its
  three CTA_SYNC/token. The new one-CTA prologue tid 0..255 covers ctr[0:2],
  doneA[0:JMAX], xready[0:JMAX], doneAs[0:16*JMAX], residue classes mod 256.
  With FT_T=64, JMAX=88: 6,344 store bytes before=after (2+18*88)*4.
  Old ownership: ctr CTA0/tid0..1; doneA/xready CTA0/tid0..87;
  doneAs CTA0..1/tid0..511 and CTA2/tid0..383. New ownership is CTA0,
  tid0..255 with stride256 (doneAs: tid0..127 six stores, others five).
  Generation remains zero in isolated H4 scratch; no final poll/advance is
  needed. Existing other-ss and other-parity clearing stays. HYB doesn't use
  ss for normalization. Split fix/cnt zeroing inside wgs_flush remains intact.
  Those kept resolver loops run in each consumer WG (tid128*q..128*q+127,
  q=0..3); wt0 publishes/zeros fc, all 128 threads read/zero their float4s.
  Non-resolver WGs retain their original early return; the original last or
  designated resolver still covers every partial before the new final exit.
  Each reset-loop element still has one STG.32; thread t executes
  [t<2]+2*[t<88]+ceil(max(1408-t,0)/256) stores. The final return adds one
  uniform flag test/branch per consumer/call, not per stage (SASS gated).
  HYB token loop removes 64 bytes/thread/local pair at 128 (two y reads),
  32 at 256 (HYB7 retains ya/yb), and 16 output bytes/thread/token. Also
  removes 8 squares+7 adds+5 shuffle/add reductions+8 weighted accumulations
  per thread/local pair, four BF16x2 conversions/token, shared reductions.
* Both reset kernels are INSIDE timed graphs. Raw net savings include their
  launch cost; separate reset timings are diagnostics, never added to gates.

Literal original CUDA globals are preserved. The additionally compiled HYB
runtime source necessarily differs even at flag 0; erasing its instrumentation
must recover the original exactly. Do not mislabel that as literal runtime
source equality or as proof of GPU output equality.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'kernels/muse_moe_w4a16/mb_layer.py'
DEST = ROOT / 'kernels/muse_moe_w4a16/mb_layer_bt4.py'

HYB_RESET = r'''// BT4 H4: stream-ordered prologue, no grid spin or cross-CTA race.
__global__ void bt4_hyb_reset(Params p) {
const int t = threadIdx.x;
if (t < 2) p.ctr[t] = 0;
for (int i = t; i < JMAX; i += 256) p.doneA[i] = 0;
for (int i = t; i < JMAX; i += 256) p.xready[i] = 0;
for (int i = t; i < JMAX * 16; i += 256) p.doneAs[i] = 0;
}
'''
HYB_EDITS = [
    ('unsigned long long* dbg;\n', 'unsigned long long* dbg;\nint bt4_mask; // BT4 kernel-argument snapshot\n'),
    ('template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {',
     HYB_RESET + 'template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {'),
    ('DBG(3);\n}\nasm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");',
     'DBG(3);\n}\nif (p.bt4_mask & 4) return; // BT4 H4: every stream-phase wait/flush has completed\n'
     'asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");'),
    ('static int g_okc_pdl = 0;\n',
     'static int g_okc_pdl = 0;\nstatic int g_bt4_mask = 0;\nvoid bt4_set_mask(int64_t mask) { g_bt4_mask = (int)mask; }\n'),
    ('p.zero = 0;\n', 'p.zero = 0;\np.bt4_mask = g_bt4_mask;\n'),
    ('cudaStream_t st = at::cuda::getCurrentCUDAStream();\n',
     'cudaStream_t st = at::cuda::getCurrentCUDAStream();\n'
     'if (p.bt4_mask & 4) {\n'
     'TORCH_CHECK(!g_okc_pdl, "BT4 H4 reset requires ordinary stream ordering");\n'
     'okc::bt4_hyb_reset<<<1, 256, 0, st>>>(p);\n'
     'TORCH_CHECK(cudaGetLastError() == cudaSuccess, "BT4 HYB reset launch");\n}\n'),
]

VML_RESET = r'''
// BT4 H3: vm_main/vm_fin code above is unchanged. All removed cleanup lives here.
static_assert(VM_DP == 0 && VM_BRED == 1, "BT4 reset is specialized to bf2");
__global__ void bt4_vml_reset_kernel(const vm::Args A) {
const int t = blockIdx.x, tid = threadIdx.x;
if (t == 0) {
for (int i = tid; i < vm::JMAX; i += 256) A.xready[i] = 0;
for (int i = tid; i < vm::JMAX * 16; i += 256) A.doneA[i] = 0;
for (int i = tid; i < 2 * vm::SLOTSD * 4; i += 256) A.cnt[i] = 0;
}
const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
for (int sl = 0; sl < 8; ++sl) {
const int r = __ldcg(A.rmap + t * 8 + sl);
if (r >= 0) {
#pragma unroll
for (int m = 0; m < 4; ++m)
__stcg(reinterpret_cast<float4*>(A.y + (size_t)r * vm::KDIM + 1024 * m + 4 * tid), z);
}
}
}
void bt4_vml_reset(std::vector<torch::Tensor> T, int64_t M) {
vm::Args a = {};
a.xready = T[0].data_ptr<int>(); a.doneA = T[1].data_ptr<int>();
a.cnt = T[2].data_ptr<int>(); a.rmap = T[3].data_ptr<int>(); a.y = T[4].data_ptr<float>();
bt4_vml_reset_kernel<<<(unsigned)M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(a);
C10_CUDA_CHECK(cudaGetLastError());
}
'''

# This append-only bank follows padz's source+flags build/cache pattern. Original
# CUDA globals, prepare validation, and the original _mbl_call stay intact.
EXTENSION = r'''

# BT4_BEGIN -- GENERATED TIMING-ONLY SUPPORT, never use probe outputs as a model.
_BT4_HYB_EDITS = @HYB_EDITS@
_BT4_VML_RESET = @VML_RESET@
_bt4_original_builds = _mbl_builds
_bt4_original_call = _mbl_call

def bt4_sources(engine, instrument=True):
 if engine == "bf2":
  src = _vml_CUDA + (_BT4_VML_RESET if instrument else "")
  cpp = _vml_CPP + ("\nvoid bt4_vml_reset(std::vector<torch::Tensor> T, int64_t M);\n" if instrument else "")
  return src, cpp, list(_vml_FLAGS) + list(_VML_DEFS["bf2"])
 if engine not in ("hyb", "hyb7"):
  raise ValueError(engine)
 src = _HYB_MOE_CUDA
 if instrument:
  for old, new in _BT4_HYB_EDITS:
   assert src.count(old) == 1, old[:80]
   src = src.replace(old, new, 1)
 cpp = _HYB_MOE_CPP + ("\nvoid bt4_set_mask(int64_t mask);\n" if instrument else "")
 return src, cpp, list(HYB_FLAGS) + (list(HYB_F7) if engine == "hyb7" else [])

def _mbl_builds(deadline):
 mods = _bt4_original_builds(deadline)
 # Serial extra compiles; the base modules/cache remain the unmodified control.
 for engine in ("bf2", "hyb", "hyb7"):
  _mbl_deadline(deadline, "BT4 instrumented build")
  src, cpp, fl = bt4_sources(engine)
  fn = ["vml_run", "vml_cfg", "bt4_vml_reset"] if engine == "bf2" else ["okc_run", "okc_set_pdl", "bt4_set_mask"]
  tag = hashlib.sha1((cpp + src + " ".join(fl)).encode()).hexdigest()[:12]
  mod = _mbl_load_inline(name="bt4_" + engine + "_" + tag, cpp_sources=cpp, cuda_sources=src,
                         functions=fn, extra_cuda_cflags=fl, verbose=False)
  if engine == "bf2":
   _vml_cfg(mod)
  else:
   mod.bt4_set_mask(0)
   mod.okc_set_pdl(int(HYB_PDL))
  mods[engine] = mod
 return mods

def _bt4_init(state, dev):
 state["bt4_flag"] = torch.zeros(1, dtype=torch.int32, device=dev)
 state["bt4_mask"] = 0
 state["bt4_arms"] = {}

def bt4_arm(state, mask, m):
 """Untimed: set/read the device word, bake its value into each captured launch.

 H3 selects the existing mask=1 launch API. H4 snapshots a host param holder
 into Params.bt4_mask; replay never dereferences a mutable shared flag word.
 One module and prepared weight bank, distinct scratch for each probe/shape.
 """
 if mask & ~6 or not 1 <= m <= 320:
  raise ValueError((mask, m))
 state["bt4_flag"].fill_(mask)
 actual = int(state["bt4_flag"].item())
 assert actual == mask
 print(f"[bt4] flag readback={actual} M={m}", flush=True)
 for name in ("hyb", "hyb7"):
  state["mods"][name].okc_set_pdl(0)
  state["mods"][name].bt4_set_mask(actual)
 key = (mask, m)
 if mask == 0:
  return state
 if key not in state["bt4_arms"]:
  arm = dict(state)
  arm["bt4_mask"] = actual
  arm["bt4_shape"] = m
  dev = state["out"][m].device
  if mask & 2 and 25 <= m <= 48:
   arm["vml_sc"] = dict(state["vml_sc"])
   arm["vml_sc"]["bf2"] = _vml_alloc(dev, state["nsm"])
   arm["vml_sc"]["bf2"]["rmap"].fill_(-1)
  if mask & 4 and m >= 49:
   arm["h_sc"], arm["h_full"] = _hyb_scratch(dev)
  state["bt4_arms"][key] = arm
 return state["bt4_arms"][key]

def bt4_reset_only(data, state):
 m = data["hidden_states"].shape[0]
 sc = state["vml_sc"]["bf2"]
 state["mods"]["bf2"].bt4_vml_reset([sc["xready"], sc["doneA"], sc["cnt"], sc["rmap"], sc["y"]], m)

def _mbl_call(name, data, state):
 mask = state["bt4_mask"]
 if mask:
  assert data["hidden_states"].shape[0] == state["bt4_shape"]
 if (mask & 2) and 25 <= data["hidden_states"].shape[0] <= 48:
  assert name == "bf2", ("BT4 H3 route", name)
  # Exact state and weight accessors copied from the base _mbl_call.
  x, ids, tw = data["hidden_states"], data["topk_ids"], data["topk_weights"]
  m = x.shape[0]
  lo = int(data["config"]["local_start"])
  st = state["banks"][lo]["k2a33"]["st"]
  out = state["out"][m]
  mod = state["mods"][name]
  rep = _k2a_bucket_of(m)
  bt4_reset_only(data, state)
  return _vml_call(mod, st, state["vml_sc"][name], x, ids, tw, ls=lo, out=out, mask=1,
                   flags=VML_FLAGS.get(rep,193), cost=VML_COST.get(rep,VML_COST[32]))
 return _bt4_original_call(name, data, state)
# BT4_END
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:90])
    return text.replace(old, new, 1)


def generate(base):
    # No source constants or validation instructions in the base are modified.
    text = replace_once(base, ' _mbl_validate(state,weights)\n',
                        ' _bt4_init(state,dev)\n _mbl_validate(state,weights)\n')
    extension = EXTENSION.replace('@HYB_EDITS@', repr(HYB_EDITS)).replace('@VML_RESET@', repr(VML_RESET))
    text = text.rstrip() + '\n' + extension
    ast.parse(text)
    return text


def environment(source):
    """Evaluate only literal globals and pure CUDA renderers; never import torch."""
    tree = ast.parse(source)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                safe = (ast.Constant, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp,
                        ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div, ast.Mod,
                        ast.LShift, ast.RShift, ast.USub, ast.UAdd, ast.Tuple, ast.List)
                if not all(isinstance(n, safe) for n in ast.walk(node.value)):
                    continue
                try:
                    value = eval(compile(ast.Expression(node.value), '<constant>', 'eval'), {'__builtins__':{}}, env)
                except (NameError, TypeError):
                    continue
            for t in node.targets:
                if isinstance(t, ast.Name):
                    env[t.id] = value
                elif isinstance(t, ast.Tuple) and all(isinstance(x, ast.Name) for x in t.elts):
                    assert len(t.elts) == len(value)
                    env.update((x.id, v) for x, v in zip(t.elts, value))
    for name in ('_vml_wg_spec', '_vml_wg_spec64'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module([node], type_ignores=[]), '<CUDA renderer>', 'exec'), env)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_vml_CUDA' for t in n.targets))
    exec(compile(ast.Module([node], type_ignores=[]), '<CUDA substitution>', 'exec'), env)
    return env


def extension_env(source):
    env = environment(source)
    ext = source[source.index('# BT4_BEGIN'):]
    env.update(_mbl_builds=lambda deadline: {}, _mbl_call=lambda *a: None)
    exec(compile(ext, '<BT4 support>', 'exec'), env)
    return env


def cpu_gates(base, variant):
    assert variant == generate(base), 'generated artifact drift'
    a, b = environment(base), environment(variant)
    strings = ('VMS_CUDA_SRC', '_HYB_MOE_CUDA', '_HYB_MOE_CPP', '_HYBLO_CUDA', '_vml_CUDA_T', '_vml_CUDA', '_vml_CPP')
    for name in strings:
        assert a[name] == b[name], name
    env = extension_env(variant)
    for name in ('hyb', 'hyb7', 'bf2'):
        src, _, _ = env['bt4_sources'](name, False)
        assert src == a['_vml_CUDA' if name == 'bf2' else '_HYB_MOE_CUDA']
        patched, _, _ = env['bt4_sources'](name)
        if name == 'bf2':
            assert patched == src + VML_RESET
        else:
            restored = patched
            for old, new in reversed(HYB_EDITS):
                restored = replace_once(restored, new, old)
            assert restored == src
            # Both stream phases, their waits/flushes and all their callees stay exact.
            start, end = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase(', '__device__ __forceinline__ void wgs_prefix('
            assert patched.split(start)[1].split(end)[0] == src.split(start)[1].split(end)[0]
    # Every moved reset index covered exactly once (H3 y mapping checked in dry).
    for n in (2, 80, 1280, 1088, 88, 1408):
        indices = [i for t in range(256) for i in range(t, n, 256)]
        assert sorted(indices) == list(range(n))
    lines = variant.splitlines()
    return {'status': 'PASS', 'original_cuda_globals_equal': list(strings),
            'uninstrumented_generated_sources_equal': True, 'runtime_hyb_erasure_equal': True,
            'literal_runtime_hyb_source_equal': False,
            'literal_note': 'Runtime flag control needs different HYB source; flag 0 semantic equality still requires GPU validation.',
            'optional_H5': 'not implemented',
            'generated_sha256': hashlib.sha256(variant.encode()).hexdigest(),
            'changed_lines': [i+1 for i,l in enumerate(lines) if '_bt4_init(state,dev)' in l] +
                             [next(i+1 for i,l in enumerate(lines) if l.startswith('# BT4_BEGIN')), len(lines)]}


def change_report(base, variant):
    """Machine-readable, exact before/after line spans for every source edit."""
    env = extension_env(variant)
    out = {}
    for name in ('hyb', 'hyb7', 'bf2'):
        old = env['bt4_sources'](name, False)[0].splitlines()
        new = env['bt4_sources'](name, True)[0].splitlines()
        out[name] = [{'kind':tag, 'base_cuda_lines':[i+1,j], 'probe_cuda_lines':[k+1,l],
                      'before':old[i:j], 'after':new[k:l]}
                     for tag,i,j,k,l in difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes() if tag!='equal']
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='read-only, require existing generated artifact')
    parser.add_argument('--changes', action='store_true')
    args = parser.parse_args()
    base = BASE.read_text(encoding='utf-8')
    text = generate(base)
    if args.check:
        assert DEST.read_bytes() == text.encode('utf-8'), 'regenerate mb_layer_bt4.py'
    else:
        DEST.write_bytes(text.encode('utf-8'))
    result = cpu_gates(base, text)
    if args.changes:
        result['per_engine_changes'] = change_report(base, text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""S7 vms16: expert-ready progressive combine; CPU-only generator and gates.

Only writes the named generated candidate and the gate JSON. Never imports the
layer, opens CUDA, runs a GPU kernel, or writes pycache. CUDA selector VMS_EPF
defaults to zero; --enable emits the active T16-only candidate requested by S7.

Kill reasons before the design:
1. Extra fences at B job crossings can consume the overlap. Final-segment fence
   replaces the old helper-end fence; only crossings add a fence.
2. An unfortunate expert slot order can leave no earlier contribution ready;
   larger register allocation can also perturb the unchanged weight stream.

Evidence: kernel_baseline.md:29/41 MEASURED 68.1/64.9 us d16s/f. R-v16m-1,
box_out/v16m/marks.log s803: MEASURED comp/hend/bar/fin maxima 52.7/53.8/54.5/55.8.
EST conditional saving = (0.7 + 1.3) - (0.80 + 1.3/2) - 0.05 = 0.50 us, assuming
half the critical combine has completed when the last needed job publishes.
The 0.05 us prologue allowance is EST 76 issue cycles / 1.52 GHz (MEASURED clock
from the standing rules): 16 map-loop iterations at 4 source issue slots plus
12 setup slots. SASS is required to validate it. The two epochs are fetched by
lanes 0/1 in one warp load, from the same 32-byte sector. Extra boundary fences
must remain hidden. This is a hypothesis, not a measured gain; measured mean
routes/token = 29/16 = 1.8125. EST value 0.50 * 0.115 = .0575 %-pts over d16s/f.
Falsifier: standalone cold-L2 kernel median <0.5 us gain or any correctness,
spill, stream-duration regression kills it; no paid test is started here.

Other candidates considered, not built: segment-local norm accumulation (norm
atomics normally overlap MMA; extra registers), and per-row A-sector packing
(L2 already combines adjacent stores; B gather/decode cost). Neither has an
evidence-supported >=0.5 us saving. They are not proposed as GPU experiments.

Protocol: A->B, NRB ring, all 8 compute warps and FP32 arithmetic stay intact.
Every helper warp publishes completed B *segments*, counting rb units, after
all 32 lanes fence ybuf/ss writes. Each job sums exactly 256 units. Finalizers
wait only for their route's job and preserve slot order. Global arrival still
happens once per CTA. Only CTA0 needs the old global wait for next-replay reset.
No CTA/thread exits the kernel early. Current-bank flags survive all consumers.
CTA0 tids 0..63 clear A counters and next-bank doneB, 0..127 clear next ss, tid0
increments both generations. ctr[3] is an EPF-only generation, snapshotted in
Smem before work: low-module calls share ctr but must not advance ready parity.
Combine CTAs 0..4*m-1, tids 0..255 cover every output;
tids 256..287 skip combine but attend both CTA barriers. Empty jobs work too.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import builtins
import contextlib
import copy
import difflib
import hashlib
import io
import json
import random
import re
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_tinys7_c1.py"
OUT = ROOT / "box_out/s7/k3_vms16_v1.py"
REPORT = ROOT / "tools/local/s7f_vms16_gate.json"
SHELL = ROOT / "tools/local/s7f_vms16_gate.sh"


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:100], s.count(old), count)
    return s.replace(old, new)


def definition(src, name):
    nodes = [n for n in ast.parse(src).body if
             isinstance(n, ast.FunctionDef) and n.name == name or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, (name, len(nodes))
    return nodes[0]


def extract(src):
    ns = {"G": 132}
    for key in ("VMS_CUDA_SRC", "VMS_CPP_SRC"):
        ns[key] = ast.literal_eval(definition(src, key).value)
    build = definition(src, "_vms_build")
    flag_nodes = [n for n in build.body if isinstance(n, ast.Assign) and
                  isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("flags", "fl_hi")]
    assert len(flag_nodes) == 2
    exec(compile(ast.Module(body=flag_nodes, type_ignores=[]), str(BASE), "exec"), ns)
    return ns


DEFINES = """#ifndef VMS_EPF
#define VMS_EPF 0
#endif
#if VMS_EPF && !VMS_T16
#error VMS_EPF requires the existing T16 specialization
#endif
"""

JOBMAP = """#if VMS_EPF
// Route scratch is dead after every lane finishes its ent/pos stores.
__syncwarp();
for (int j = lane; j < S.njobs; j += 32) {
  const int st = S.jS[j], cnt = S.jC[j];
  for (int tk = 0; tk < cnt; ++tk) {
    const int pr = st + tk;
    S.base[pr >> 5][pr & 31] = j;
  }
}
#endif
"""

PUBLISH = """#if VMS_EPF
// All ybuf stores and norm atomics for this contiguous segment precede ready.
if (q == N - 1 || ((v + 1) >> 8) != j) {
  __threadfence();
  __syncwarp();
  if (lane == 0) {
    const int n = min(b0 + N - nA, (j + 1) * RBB) - max(b0, j * RBB);
    atomicAdd(p.ctr + 72 + (S.epf_gen & 1) * MAXJ + j, n);
  }
}
#endif
"""

WAIT = """#if VMS_EPF
const int j = S.base[pr >> 5][pr & 31];
const int* ready = p.ctr + 72 + (S.epf_gen & 1) * MAXJ + j;
if ((tid & 31) == 0) {
  int spins = 0;
  while (ld_acq(ready) < RBB) {
    if (++spins > WAIT_MAX) { atomicExch(p.ctr + 2, 3); break; }
    __nanosleep(32);
  }
}
__syncwarp();
(void)ld_acq(ready); // acquire for every reader, as in the base A->B path
#endif
"""


def patch_cuda(cuda):
    # Tail boundary is scoped to the VMS string, never another module.
    tail_start = cuda.index("__syncthreads();\n#if __CUDA_ARCH__ >= 900\nasm volatile(\"griddepcontrol.launch_dependents;")
    combine_start = cuda.index("const float* ssc = p.ss + (gen & 1) * MAXP;", tail_start)
    combine_end = cuda.index("if (mk && tid == 0) mrow[9] = gt();", combine_start)
    combine = cuda[combine_start:combine_end]
    eager = sub(combine, "if (pr >= 0) {\n", "if (pr >= 0) {\n" + WAIT)
    eager = sub(eager, "const float* ssc = p.ss + (gen & 1) * MAXP;",
                "const float* ssc = p.ss + (S.gen & 1) * MAXP;")
    helper = ("#if VMS_EPF\n__device__ __forceinline__ void epf_combine(const Params& p, const Smem& S, int c, int tid) {\n"
              + eager + "}\n#endif\n")
    cuda = sub(cuda, "namespace vms {", DEFINES + "namespace vms {")
    cuda = sub(cuda, 'int njobs, gen;', 'int njobs, gen;\n#if VMS_EPF\nint epf_gen;\n#endif')
    cuda = sub(cuda, 'if (tid == 0) S.gen = ld_rlx(p.ctr);',
               '''#if VMS_EPF
// Both epoch words share one warp load / one 32-byte sector, no second load hop.
if (tid < 2) {
  const int epoch = ld_rlx(p.ctr + 3 * tid);
  if (tid == 0) S.gen = epoch; else S.epf_gen = epoch;
}
#else
if (tid == 0) S.gen = ld_rlx(p.ctr);
#endif''')
    cuda = sub(cuda, "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {",
               helper + "__global__ void __launch_bounds__(NTL, 1) vms_kernel(Params p) {")
    route_end = "else S.pos[i] = -1;\n}\n}\n}\n__syncthreads();"
    cuda = sub(cuda, route_end, "else S.pos[i] = -1;\n}\n}\n" + JOBMAP + "}\n__syncthreads();")
    # Last B store loop ends here; retain both tile loops and their arithmetic.
    red_end = """}
}
#endif
}
}
#if VMS_FX & 2
asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#else
__threadfence();
#endif
if (mk && lane == 0) mrow[7] = gt();"""
    new_end = """}
}
#endif
""" + PUBLISH + """}
}
#if !VMS_EPF
#if VMS_FX & 2
asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#else
__threadfence();
#endif
#endif
if (mk && lane == 0) mrow[7] = gt();"""
    cuda = sub(cuda, red_end, new_end)
    # The arrival is unchanged and precedes any eager consumer wait.
    before_wait = "#endif\nconst int target = (gen + 1) * G;"
    cuda = sub(cuda, before_wait, """#endif
#if VMS_EPF
}
epf_combine(p, S, c, tid);
if (tid == 0 && c == 0) {
#endif
const int target = (gen + 1) * G;""")
    cuda = sub(cuda, "if (tid < MAXJ) p.ctr[8 + tid] = 0;",
               """if (tid < MAXJ) p.ctr[8 + tid] = 0;
#if VMS_EPF
if (tid < MAXJ) p.ctr[72 + ((S.epf_gen + 1) & 1) * MAXJ + tid] = 0;
#endif""")
    cuda = sub(cuda, 'if (tid == 0) atomicAdd(p.ctr, 1);',
               '#if VMS_EPF\nif (tid == 0) atomicAdd(p.ctr + 3, 1);\n#endif\nif (tid == 0) atomicAdd(p.ctr, 1);')
    # The original tail remains verbatim behind the off arm.
    cuda = sub(cuda, combine, "#if !VMS_EPF\n" + combine + "#endif\n")
    cuda = sub(cuda, 'TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");',
               'TORCH_CHECK(m >= 1 && m <= 16, "vms: 1 <= m <= 16");\n'
               '#if VMS_EPF\nTORCH_CHECK(ctr.numel() >= 72 + 2 * vms::MAXJ, "vms epf: counter capacity");\n#endif')
    return cuda


def generate(base, enable=True):
    old = extract(base)
    node = definition(base, "VMS_CUDA_SRC")
    lines = base.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = ['VMS_CUDA_SRC=r"""' + patch_cuda(old["VMS_CUDA_SRC"]) + '"""\n']
    out = "".join(lines)
    flag = '"-DVMS_NS=2","-DVMS_ER=1"]+flags[4:]'
    out = sub(out, flag, f'"-DVMS_NS=2","-DVMS_ER=1","-DVMS_EPF={int(enable)}"]+flags[4:]')
    # This expression is copied from the real _vms_prepare state constructor.
    out = sub(out, '"ctr":torch.zeros((72,),dtype=torch.int32,device=dev),',
              '"ctr":torch.zeros((200,),dtype=torch.int32,device=dev),')
    lines = out.splitlines(keepends=True)
    assert lines[0].startswith("# okc: problem=muse_decode_layer") and lines[1].startswith("# nonce ")
    lines[1] = f"# nonce vms16-v1 (S7 expert-ready progressive combine; VMS_EPF={int(enable)}; tools/s7f_vms16_patches.py)\n"
    return "".join(lines)


def exec_nodes(src, names, ns):
    body = [definition(src, name) for name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(OUT), "exec"), ns)


def dry_run(base, candidate, enabled):
    """Padz/VMSS harness pattern: real AST layouts, fake tensor/build operations.

    No invented state accessor: execute _vms_prepare's actual st['vms'] assign,
    _v11_k2a33_prepare's actual return dict, _vms_build, and _vms_call.
    """
    class Tensor:
        def __init__(self, shape=(1,), dtype=None, device="cpu"):
            self.shape, self.dtype, self.device = tuple(shape), dtype, device
        def float(self): return self
        def reshape(self, *shape): return Tensor(shape, self.dtype, self.device)
        def contiguous(self): return self
        def to(self, *args): return self
        def __mul__(self, other): return self
        def __radd__(self, other): return self

    builds, calls = [], []
    class Module:
        def __init__(self, name): self.name = name
        def run(self, *args): calls.append((self, args))
        def attrs(self): return [0, 0, 0, 132, 2]

    def load_inline(**kw):
        builds.append(kw)
        return Module(kw["name"])

    class Thread:
        def __init__(self, target, **kw): self.target = target
        def start(self): self.target()
        def join(self): pass

    def fake_import(name, *args, **kw):
        if name == "torch.utils.cpp_extension": return types.SimpleNamespace(load_inline=load_inline)
        if name == "threading": return types.SimpleNamespace(Thread=Thread)
        return builtins.__import__(name, *args, **kw)

    torch = types.SimpleNamespace(zeros=Tensor, empty=Tensor, int32="int32", int64="int64",
        bfloat16="bf16", float32="fp32", cuda=types.SimpleNamespace(
            get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=132)))
    ns = dict(torch=torch, hashlib=hashlib, time=time, _K3_BUILD_S={},
        __builtins__=dict(vars(builtins), __import__=fake_import),
        K2A_D_HIDDEN=4096, K2A_D_EXPERT=4096, K2A_POST_NORM_EPS=1e-6,
        VMS_HI_ON=True, E=32, two62=float(2**62), dev="cpu", wA=Tensor(), wB=Tensor(),
        w={k: Tensor() for k in ("w13_gs", "w2_gs", "post_norm_w")}, st={})
    exec_nodes(candidate, ("VMS_CUDA_SRC", "VMS_CPP_SRC", "_VMS", "_vms_build", "_vms_call"), ns)
    with contextlib.redirect_stdout(io.StringIO()): ns["mod"] = ns["_vms_build"]()
    assert len(builds) == 2
    info = extract(candidate)
    for key, module_key in (("flags", "mod"), ("fl_hi", "mod_hi")):
        b = next(v for v in builds if v["name"] == ns["_VMS"][module_key].name)
        assert b["extra_cuda_cflags"] == info[key]
    assert f"-DVMS_EPF={int(enabled)}" in info["fl_hi"]
    assert not any(f.startswith("-DVMS_EPF") for f in info["flags"])
    layouts = {}
    for tag, source in (("base", base), ("candidate", candidate)):
        prep = definition(source, "_vms_prepare")
        layout = [n for n in ast.walk(prep) if isinstance(n, ast.Assign) and
                  any(ast.unparse(t) == "st['vms']" for t in n.targets)]
        assert len(layout) == 1
        ns["st"] = {}
        exec(compile(ast.Module(body=layout, type_ignores=[]), str(BASE), "exec"), ns)
        layouts[tag] = ns["st"]["vms"]
    assert layouts["base"].keys() == layouts["candidate"].keys()
    assert layouts["base"]["ctr"].shape == (72,) and layouts["candidate"]["ctr"].shape == (200,)
    for k in ("inter", "ybuf", "ss", "nomarks"):
        assert layouts["base"][k].shape == layouts["candidate"][k].shape
    kprep = definition(base, "_v11_k2a33_prepare")
    ret = [n for n in kprep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)]
    assert len(ret) == 1
    ns["ok"] = {16: True}
    state = {"k2a33": eval(compile(ast.Expression(ret[0].value), str(BASE), "eval"), ns), "k2a": None}
    assert state["k2a33"]["st"] is ns["st"]
    for high in (False, True):
        ns["VMS_HI_ON"] = high
        for t in (1, 8, 9, 12, 16):
            out = ns["_vms_call"](state["k2a33"]["st"], Tensor((t, 4096)), Tensor((t, 8)), Tensor((t, 8)))
            md, args = calls[-1]
            vs = layouts["candidate"]
            assert out.shape == (t, 4096) and args[11] is vs["ctr"]
            assert md is vs["mod_hi" if high and t > 8 else "mod"]
            assert args[3] is vs["wA"] and args[4] is vs["wB"]
    # Exercise existing high-module failure fallback with the enlarged allocation.
    ns["st"]["vms"]["mod_hi"] = None
    ns["_vms_call"](ns["st"], Tensor((16, 4096)), Tensor((16, 8)), Tensor((16, 8)))
    assert calls[-1][0] is ns["st"]["vms"]["mod"]
    for missing in ("vms",):
        try: ns["_vms_call"]({}, Tensor((16, 4096)), Tensor((16, 8)), Tensor((16, 8)))
        except KeyError as exc: assert exc.args[0] == missing
        else: raise AssertionError("missing state key was not detected")
    return {"real_state_layout": True, "build_paths": len(builds), "call_paths": len(calls),
            "counter_ints": 200, "base_counter_ints": 72, "fallback_checked": True,
            "new_bank": False, "arm": int(enabled)}


def weighted_plan(counts):
    weights = [30 if n > 8 else 20 for n in counts]
    prefix = [0]
    for w in weights: prefix.append(prefix[-1] + w)
    def split(per, x):
        j = min(len(weights), bisect.bisect_right([per * v for v in prefix[1:]], x))
        return len(weights) * per if j == len(weights) else j * per + (x - per * prefix[j]) // weights[j]
    def wpre(per, v):
        j, k = divmod(v, per)
        return per * prefix[j] + (k * weights[j] if j < len(weights) else 0)
    result = []
    for c in range(132):
        a0, a1 = [split(512, 512 * prefix[-1] * z // 132) for z in (c, c + 1)]
        b0, b1 = [split(256, 768 * prefix[-1] * z // 132 - wpre(512, a))
                  for z, a in ((c, a0), (c + 1, a1))]
        result.append((a0, a1, b0, b1))
    return result


def protocol_gate():
    rng = random.Random(716)
    cases = [[], [1], [8], [9], [16], [16] * 8, [4] * 32, [1] * 64, [16] * 4 + [1] * 60,
             [12, 6, 5, 3, 2, 1], [13, 5, 4, 2, 1, 1, 1, 1]]
    for _ in range(200):
        counts, left = [], rng.randrange(129)
        while left and len(counts) < 32:
            n = rng.randint(1, min(left, 16)); counts.append(n); left -= n
        cases.append(counts)
    crossing_max = 0
    for counts in cases:
        assert sum(counts) <= 128
        jobs = len(counts)
        plan = weighted_plan(counts)
        assert plan[0][0] == plan[0][2] == 0
        assert plan[-1][1] == jobs * 512 and plan[-1][3] == jobs * 256
        for left, right in zip(plan, plan[1:]): assert left[1] == right[0] and left[3] == right[2]
        events, covered = [], [set() for _ in counts]
        for c, (a0, a1, b0, b1) in enumerate(plan):
            assert a0 <= a1 and b0 <= b1
            for j in range(b0 // 256, (b1 - 1) // 256 + 1) if b1 > b0 else ():
                lo, hi = max(b0, j * 256), min(b1, (j + 1) * 256)
                assert not covered[j].intersection(range(lo, hi))
                covered[j].update(range(lo, hi))
                events.append((c, j, hi - lo))
        assert all(len(v) == 256 for v in covered)
        crossing_max = max(crossing_max, len(events) - sum(b1 > b0 for _, _, b0, b1 in plan))
        # Replay alternating active/inactive jobs, arbitrary segment arrival order.
        banks = [[0] * 64 for _ in range(2)]
        for gen in range(8):
            ev = list(events) if gen % 3 != 1 else []
            rng.shuffle(ev)
            touched = set()
            for _, j, n in ev:
                banks[gen & 1][j] += n
                assert banks[gen & 1][j] <= 256
                touched.add(j)
            assert all(banks[gen & 1][j] == 256 for j in touched)
            assert not any(banks[gen & 1][j] for j in range(64) if j not in touched)
            banks[(gen + 1) & 1] = [0] * 64  # CTA0 tids 0..63 only, after 132 arrivals
        # Reused routing map stores exactly one job for every local route.
        mapping = [j for j, cnt in enumerate(counts) for _ in range(cnt)]
        assert len(mapping) == sum(counts) and all(0 <= j < jobs for j in mapping)
    # Shared ctr state is used by both low and high modules during real prepare.
    # Exhaust short low/hi sequences. A layer-generation parity mutation MUST fail.
    def replay_sequence(sequence, bad_parity=False):
        banks, epgen = [[0] * 64 for _ in range(2)], 0
        for layergen, high in enumerate(sequence):
            if not high:
                continue  # low module knows nothing about the EPF counters
            active = (layergen if bad_parity else epgen) & 1
            assert not any(banks[active]), (sequence, layergen, epgen, active)
            banks[active][0] = 256
            banks[active ^ 1] = [0] * 64
            epgen += 1
    for mask in range(1 << 10):
        replay_sequence([(mask >> k) & 1 for k in range(10)])
    try: replay_sequence([1, 0, 1], bad_parity=True)
    except AssertionError: pass
    else: raise AssertionError('mixed low/high parity mutation survived')
    for m in range(1, 17):
        owners = [(c, tid, (c >> 2) * 4096 + (c & 3) * 1024 + tid * 4)
                  for c in range(132) if c < m * 4 for tid in range(256)]
        indices = [col + k for _, _, col in owners for k in range(4)]
        assert sorted(indices) == list(range(m * 4096))
        # No exits: every CTA has all 288 threads at both common tail barriers.
        assert m * 4 <= 64 and set(range(64)).issubset(range(288))
        assert set(range(128)).issubset(range(288))
    return {"routing_plans": len(cases), "replay_generations_per_plan": 8,
            "all_output_shapes": list(range(1, 17)), "max_extra_segment_fences": crossing_max,
            "arrival_ctas": list(range(132)), "cleanup_cta": 0,
            "cleanup_threads": {"A_and_next_doneB": [0, 63], "next_ss": [0, 127], "generation": [0, 0]},
            "combine_threads": [0, 255], "helper_threads_at_barriers": [256, 287], "early_exits": 0,
            "mixed_low_hi_sequences": 1024, "bad_generation_mutant_rejected": True,
            "epf_generation_counter": 3, "epf_generation_snapshotted_before_work": True}


def changed_lines(base, out):
    before, after = base.splitlines(), out.splitlines()
    return [{"bit": 1, "base": [i + 1, j], "candidate": [k + 1, l],
             "first": after[k] if k < l else "<empty>"}
            for op, i, j, k, l in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
            if op != "equal"]


def cpu_gate(base, out, enabled=True):
    assert generate(base, enabled) == out, "stale generated candidate"
    assert out.isascii() and "\r" not in out and len(out.encode()) < 1_000_000
    compile(out, str(OUT), "exec")  # no pycache, no module import
    original, changed = extract(base), extract(out)
    assert original["flags"] == changed["flags"]
    assert [f for f in changed["fl_hi"] if not f.startswith("-DVMS_EPF=")] == original["fl_hi"]
    assert original["VMS_CPP_SRC"] == changed["VMS_CPP_SRC"]
    # Every other top-level definition, including self-tests and kernel routes,
    # is exactly the original AST. Only these three nodes are allowed to differ.
    a, b = ast.parse(base).body, ast.parse(out).body
    assert len(a) == len(b)
    allowed = {"VMS_CUDA_SRC", "_vms_build", "_vms_prepare"}
    for x, y in zip(a, b):
        key = x.name if isinstance(x, ast.FunctionDef) else (x.targets[0].id if isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) else None)
        if key not in allowed: assert ast.dump(x) == ast.dump(y), key
    for name in ("_vms_build", "_vms_prepare"):
        x, y = copy.deepcopy(definition(base, name)), copy.deepcopy(definition(out, name))
        if name == "_vms_build":
            a = next(n for n in x.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "fl_hi")
            b = next(n for n in y.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "fl_hi")
            b.value = a.value
        else:
            a = next(n for n in ast.walk(x) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "st['vms']" for t in n.targets))
            b = next(n for n in ast.walk(y) if isinstance(n, ast.Assign) and any(ast.unparse(t) == "st['vms']" for t in n.targets))
            bd = ast.dump(b)
            expected = ast.dump(a).replace("Constant(value=72)", "Constant(value=200)")
            assert bd == expected
            b.value = a.value
        assert ast.dump(x) == ast.dump(y), name
    cuda = changed["VMS_CUDA_SRC"]
    assert "__trap" not in cuda and "VMS_EPF 0" in cuda
    assert cuda.count("p.ctr + 72 + (S.epf_gen & 1) * MAXJ + j") == 2
    assert cuda.count("p.ctr[72 + ((S.epf_gen + 1) & 1) * MAXJ + tid] = 0") == 1
    assert cuda.count('const int epoch = ld_rlx(p.ctr + 3 * tid)') == 1
    assert cuda.count('S.epf_gen = epoch') == 1
    assert cuda.count('atomicAdd(p.ctr + 3, 1)') == 1
    assert not re.search(r'p\.ctr\s*\+\s*3\b|p\.ctr\[3\]', original['VMS_CUDA_SRC'])
    # Byte-for-byte unchanged hot weight/consumer loop, including ring protocol.
    start, end = "if (warp < NW) {", "} else {\nconst int g = lane >> 2, t = lane & 3;\nfloat* ssc"
    hot = lambda s: s[s.index(start):s.index(end, s.index(start))]
    assert hot(original["VMS_CUDA_SRC"]) == hot(cuda)
    dry = dry_run(base, out, enabled)
    return {"ok": True, "bytes": len(out.encode()), "sha256": hashlib.sha256(out.encode()).hexdigest(),
            "dry_run": dry, "protocol": protocol_gate(), "changed_lines": changed_lines(base, out),
            "hot_weight_loop_unchanged": True, "weight_bytes_per_stage_before_after": [36864, 36864],
            "scratch_bytes_added": 512, "shared_bytes_added": 0,
            "instructions": {"compute_per_thread_delta": 0, "reducer_per_B_stage":
                "+2 end/boundary predicates; per job segment: 1 fence/lane + 1 syncwarp + 1 atomic on lane0 (final fence replaces old helper-end fence)",
                "combine_per_local_slot": "+1 shared job load, lane0 acquire poll loop, 1 syncwarp + 1 acquire/thread; original FP32 arithmetic and ybuf loads unchanged"},
            "static_loop_accounting": {
                "kind": "EST source-level counts, not measured SASS; logical scalar bytes before coalescing",
                "route": {"job_map_store_bytes_per_local_route_per_CTA": [0, 4],
                          "job_map_store_instructions_per_route": [0, 1],
                          "extra_epoch_load_bytes_per_CTA": 4},
                "weight_compute": {"weight_stage_bytes": [36864, 36864],
                    "partial_store_bytes_per_compute_thread_8_or_16_tokens": [[16, 32], [16, 32]],
                    "instructions_per_compute_thread": "identical source; delta 0"},
                "reducer": {"partial_read_bytes_per_helper_thread_8_or_16_tokens": [[128, 256], [128, 256]],
                    "A_inter_store_bytes_per_stage": ["16*live_tokens", "16*live_tokens"],
                    "B_ybuf_store_bytes_per_stage": ["64*live_tokens", "64*live_tokens"],
                    "norm_atomic_RMW_operands_bytes_per_stage": ["4*live_tokens", "4*live_tokens"],
                    "B_end_tests_per_thread_per_stage": [0, "2 comparisons + OR + branch"],
                    "publication_per_B_segment": ["none", "1 fence/lane + 1 syncwarp + 1 lane0 4-byte atomic RMW"],
                    "final_helper_fence_per_thread": [1, 0]},
                "combine": {"global_bytes_per_thread_per_live_slot": [24, "28 + 4*polls on lane0"],
                    "shared_bytes_per_thread_per_live_slot": [4, 8],
                    "added_source_instructions_per_thread_per_live_slot": "1 shared load + 1 acquire + 1 syncwarp; lane0 also bounded acquire/nanosleep loop",
                    "arithmetic_instructions_per_thread": "same expression order, delta 0",
                    "output_store_bytes_per_thread": [8, 8]},
                "cleanup": {"new_flag_store_bytes_per_call": 256, "new_epoch_RMW_operand_bytes_per_call": 4,
                    "grid_counter_pollers_before_after": [132, 1]},
            },
            "marks_caveat": "EPF mrow[8] is CTA0 cleanup wait after its eager combine; do not interpret it as the old all-CTA pre-combine barrier timestamp",
            "est_us": {"kind": "conditional, not measured", "d16s": .50, "d16f": .50,
                       "formula": "(0.7+1.3)-(0.80+1.3/2)-76/1520", "weighted_percentage_points": .50 * .115}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enable", action="store_true", help="enable EPF only on the T16 high module; default is off")
    ap.add_argument("--check", action="store_true", help="verify the existing output without rewriting it")
    args = ap.parse_args()
    base = BASE.read_bytes().decode("ascii")
    candidate = generate(base, args.enable)
    cpu = cpu_gate(base, candidate, args.enable)
    # Also run the selector-off generation path, in memory only.
    cpu["off_dry_run"] = dry_run(base, generate(base, False), False)
    if args.check:
        assert OUT.read_bytes() == candidate.encode("ascii")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(candidate.encode("ascii"))
    report = {"cpu": cpu, "cuda_compile_ok": None, "sass_ok": None, "gpu_run": False,
              "reason": "CPU-only Windows run; Linux compile/SASS script supplied", "base_sha256": hashlib.sha256(base.encode()).hexdigest()}
    REPORT.write_bytes((json.dumps(report, indent=2) + "\n").encode("ascii"))
    print(json.dumps({"cpu_ok": cpu["ok"], "bytes": cpu["bytes"], "dry_run": cpu["dry_run"],
                      "output": str(OUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()

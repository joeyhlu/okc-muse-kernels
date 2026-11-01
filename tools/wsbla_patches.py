"""WSBLK-A: exact A-tail delegation in the ship TINY/TINY2 bank.

Kill risks first: named-barrier/fence drain can consume the helper backlog;
tail epilogue register pressure can spill or slow the common consumer stream.
Neither is a measured outcome: compile/resource gates and paired timing decide.
Only the named lab output is written. --check / --dry-run are read-only.
"""
import argparse
import ast
import builtins
import contextlib
import difflib
import functools
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
OUT = ROOT / "kernels/lab/k3_wsbla_ab.py"


@functools.lru_cache(maxsize=4)
def parsed(source):
    return ast.parse(source)


def node(source, name):
    return next(n for n in parsed(source).body if
                getattr(n, "name", None) == name or isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def literal(source, name):
    return ast.literal_eval(node(source, name).value)


def _wsbla_sub(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"WSBLA anchor count {s.count(old)}: {old[:100]!r}")
    return s.replace(old, new)


_WSBLA_DEBUG_EDITS = (
    ('#define OKC_DEBUG 0', '#define OKC_DEBUG 1'),
    ('mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));',
     'mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\nif (!PB && i == 0) { DBG(6); } // TINYMK: first W13 slot observed full'),
    ('if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }',
     'if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64);\nif (i == 0) { DBG(7); } // TINYMK: first W2 doneA spin exit\n} '),
)
WSBLA_SLOTS = {
    0: "entry_before_PDL_wait", 1: "route_end_before_ss_reset_and_WS_JOIN",
    2: "W13_TIDX0_end_ctl_helper_may_drain_a2_own_tail_done_not_all_delegates",
    3: "W2_consumer_phase_end_before_WS_JOIN_and_combine_setup",
    4: "combine_grid_barrier_exit_after_CTA_SYNC", 5: "combine_end_TIDX0_not_all_thread_exit",
    6: "first_W13_ring_full_wait_passed_TIDX0",
    7: "first_W2_doneA_spin_exit_TIDX0_before_syncwarp",
}
WSBLA_ON = "ctl"
WSBLA_ARMS = ("ctl", "a2", "a4", "ctl_dbg", "a2_dbg")


def _wsbla_source(source, kind):
    if kind.endswith("_dbg"):
        s = _wsbla_source(source, kind[:-4])
        for old, new in _WSBLA_DEBUG_EDITS:
            s = _wsbla_sub(s, old, new)
        return s
    if kind == "ctl":
        return source
    if kind not in ("a2", "a4"):
        raise ValueError(kind)
    s = _wsbla_sub(source, "#define OKC_WSRF 0\n", "#define OKC_WSRF 0\n" +
        f"#define OKC_WSBLKA {int(kind[1:])}\n" +
        'static_assert(OKC_WS && OKC_WSB && OKC_WSBL && OKC_WSBLKA <= NRB && OKC_WSBLKA <= NW, "WSBLKA needs the shipped ring/refill protocol");\n')
    # Verbatim helper reduction and epilogue: preserve FP32 addition order,
    # SILU_DIV, probe_q, BF16 rounding and store policy.
    helper = source.split('__device__ void ws_helper(', 1)[1]
    reduction = helper.split('float4 sm =', 1)[1].split('__syncwarp();', 1)[0]
    reduction = 'float4 sm =' + reduction
    epilogue = helper.split('if (!PB) {\n', 1)[1].split('if (q == nA - 1)', 1)[0]
    credit = helper.split('if (q == nA - 1) {\n', 1)[1].split('\n} else {', 1)[0]
    credit = credit.rsplit('\n}', 1)[0]  # remove old q-condition, keep lane0/job loop
    s = _wsbla_sub(s,
        'if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK)) mbar_arrive1(S.wsrf + bb);',
        'if (!(OKC_WSBL && PB && i >= nU - OKC_WSBLK) && !(!PB && i >= nU - OKC_WSBLKA)) mbar_arrive1(S.wsrf + bb);')
    tail = '''// WSBLA: last <= NRB units occupy distinct red slots after the final CTA_SYNC.
if (!PB && warp < OKC_WSBLKA && warp < nU) {
const int ul = nU - 1 - warp;
const int q = seq0 + ul, bb = q % NRB;
const int v = v0 + ul, j = njF + v / RB_A, rb = v - (v / RB_A) * RB_A;
const int gg = lane >> 2, tt = lane & 3;
''' + reduction + '''__syncwarp();
// Advance BOTH ring phases once. Unlike B-tail, these slots are reused in B.
// All red reads are complete; helper skips these full notifications entirely.
if (lane == 0) {
mbar_arrive1(S.wsrf + bb);
mbar_arrive1(S.wsre + bb);
}
const int cnt = S.jC[j], st = S.jS[j], ce = S.jE[j];
''' + epilogue + '''// All 32 lanes arrive after their inter stores. No CTA-wide early exit.
const int participants = 32 * (1 + min(OKC_WSBLKA, nU));
asm volatile("bar.arrive 3, %0;\\n" :: "r"(participants) : "memory");
}
'''
    s = _wsbla_sub(s, 'if (PB && warp < OKC_WSBLK && warp < nU) {',
                  tail + 'if (PB && warp < OKC_WSBLK && warp < nU) {')
    old_credit = 'if (q == nA - 1) {\n' + credit + '\n}'
    s = _wsbla_sub(s, old_credit, '// WSBLA credits are published at the helper/delegate rendezvous.')
    # Jump over the delegated A interval, retaining GLOBAL q for B parity and
    # refill geometry. Do this even when Nh==nA (all B units are delegated).
    s = _wsbla_sub(s, 'for (int q = 0; q < Nh; ++q) {', '''for (int q = 0; q < Nh; ++q) {
if (nA > 0 && q == nA - min(OKC_WSBLKA, nA)) {
const int participants = 32 * (1 + min(OKC_WSBLKA, nA));
// Helper prefix stores and every delegate store precede this rendezvous.
asm volatile("bar.sync 3, %0;\\n" :: "r"(participants) : "memory");
''' + credit + '''
q = nA;
if (q >= Nh) break;
}
''')
    return s


def _wsbla_build_one(k, kind, var):
    from torch.utils.cpp_extension import load_inline
    cu = _wsbla_source(_HYBLO_CUDA, kind)
    flags = list(_HYBLO_FLAGS) + list(var) + ["-Xptxas=-v"]
    tag = hashlib.sha1((cu + " ".join(flags)).encode()).hexdigest()[:12]
    return load_inline(name=f"wsbla_{k}_{kind}_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=cu,
                       functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=flags, verbose=False)


def _wsbla_prepare(dev):
    _WSBLA["errors"].clear()
    _WSBLA["calls"].clear()
    if not HYBLO_ENABLE:
        return
    if _HYBLO["w"] is None or _HYBLO["sc"] is None:
        raise RuntimeError("WSBLA: ship HYBLO prepare unavailable")
    if _WSBLA["dbg"] is None:
        _WSBLA["dbg"] = torch.zeros(_HYB["nsm"] * 8, dtype=torch.int64, device=dev)
    assert len(_HYBLO["sc"]) == 3
    _WSBLA["sc_dbg"] = list(_HYBLO["sc"]) + [_WSBLA["dbg"]]
    for k, _, var in _HYBLO_KEYS:
        if not _HYBLO["ok"].get(k) or k not in _HYBLO["mods"]:
            raise RuntimeError(f"WSBLA: ship {k} unavailable")
        for kind in WSBLA_ARMS[1:]:
            key, t0 = (k, kind), time.time()
            try:
                if key not in _WSBLA["mods"]:
                    _WSBLA["mods"][key] = _wsbla_build_one(k, kind, var)
            except Exception as ex:
                _WSBLA["errors"][key] = repr(ex)
            _WSBLA["secs"][key] = time.time() - t0


def _wsbla_select(k, m):
    if WSBLA_ON not in WSBLA_ARMS:
        raise RuntimeError(f"invalid WSBLA arm {WSBLA_ON!r}")
    kind = WSBLA_ON if (k, m) in (("tiny", 1), ("tiny2", 8)) else "ctl"
    if kind == "ctl":
        mod = _HYBLO["mods"][k]
    else:
        if (k, kind) not in _WSBLA["accepted"]:
            raise RuntimeError(f"WSBLA {k}/{kind}: compile/SASS gate not accepted")
        if (k, kind) in _WSBLA["errors"] or (k, kind) not in _WSBLA["mods"]:
            raise RuntimeError(f"WSBLA {k}/{kind}: missing build")
        mod = _WSBLA["mods"][(k, kind)]
    key = f"{k}:{m}:{WSBLA_ON}:{kind}"
    _WSBLA["calls"][key] = _WSBLA["calls"].get(key, 0) + 1
    return mod, (k, kind)


def build(base):
    assert "WSBLA_ON" not in base
    runtime = "\n# WSBLK-A LAB: T=1/T=8 only; debug never enters paired timings.\n"
    for name in ("_WSBLA_DEBUG_EDITS", "WSBLA_SLOTS", "WSBLA_ON", "WSBLA_ARMS"):
        runtime += f"{name} = {globals()[name]!r}\n"
    runtime += '_WSBLA = {"mods": {}, "errors": {}, "secs": {}, "calls": {}, "accepted": set(), "dbg": None, "sc_dbg": None}\n'
    for fn in (_wsbla_sub, _wsbla_source, _wsbla_build_one, _wsbla_prepare, _wsbla_select):
        runtime += inspect.getsource(fn) + "\n"
    s = _wsbla_sub(base, "HYBLO_ENABLE = True\n", runtime + "HYBLO_ENABLE = True\n")
    run = ast.get_source_segment(s, node(s, "_hyblo_run"))
    new = _wsbla_sub(run, 'mod = _HYBLO["mods"][k]', 'mod, pkey = _wsbla_select(k, x.shape[0])')
    new = new.replace('_HYBLO["pdl"].get(k)', '_HYBLO["pdl"].get(pkey)').replace('_HYBLO["pdl"][k]', '_HYBLO["pdl"][pkey]')
    new = _wsbla_sub(new, 'mod.okc_run(_HYBLO["w"], _HYBLO["sc"],',
        'sc = _WSBLA["sc_dbg"] if pkey[1].endswith("_dbg") else _HYBLO["sc"]\n    mod.okc_run(_HYBLO["w"], sc,')
    s = _wsbla_sub(s, run, new)
    anchor = next(l for l in s.splitlines(True) if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    s = _wsbla_sub(s, anchor, anchor + '        _wsbla_prepare(weights["w13"].device)\n')
    ast.parse(s)
    cu = literal(base, "_HYBLO_CUDA")
    assert literal(s, "_HYBLO_CUDA") == cu == _wsbla_source(cu, "ctl")
    assert _WSBLA_DEBUG_EDITS == literal((ROOT / "kernels/lab/k3_tinymk.py").read_text(encoding="utf-8"), "_TINYMK_EDITS")
    def section(x, a, b): return x.split(a, 1)[1].split(b, 1)[0]
    for arm in WSBLA_ARMS:
        v = _wsbla_source(cu, arm)
        assert v.count('atomicAdd(p.doneA + jj, b - a);') == 1
        assert v.count('while (ld_acquire(p.doneA + j) < RB_A)') == 1
        assert v.count('return;') == cu.count('return;')
        assert v.count('__threadfence();') == cu.count('__threadfence();')
        for a, b in (('const unsigned char* slot = ring + sl * UNIT_BYTES;', 'const int bb = q % NRB;'),
                     ('if (PB && warp < OKC_WSBLK && warp < nU) {', 'seq = seq0 + nU;'),
                     ('WS_JOIN();\nasm volatile("griddepcontrol.launch_dependents;', '#include <torch/extension.h>')):
            assert section(cu, a, b) == section(v, a, b), (arm, a)
        for token in ('mbar_init(', 'CTA_SYNC();', 'WS_JOIN();', 'other[i]', 'p.doneA[i] = 0'):
            assert [l for l in cu.splitlines() if token in l] == [l for l in v.splitlines() if token in l], (arm, token)
    base_lines, new_lines = base.splitlines(), s.splitlines()
    for n in parsed(base).body:
        if isinstance(n, ast.FunctionDef) and n.name not in ('prepare', '_hyblo_run'):
            nn = node(s, n.name)
            assert base_lines[n.lineno-1:n.end_lineno] == new_lines[nn.lineno-1:nn.end_lineno], n.name
    assert not any('bar.sync 3' in l or 'bar.arrive 3' in l for l in cu.splitlines())
    assert '#define OKC_WSNR 4' in cu and '#define OKC_F1EARLY 1' in cu
    for _, _, var in literal(base, '_HYBLO_KEYS'):
        assert '-DOKC_SPLIT=0' in var
    return s


def protocol_gate():
    """Exhaust unit ownership, credit intersections and reused ring parity.

    Event model below asserts that helper prefix and tail stores precede credits;
    the source implements those dependencies with CTA_SYNC and named barrier 3.
    No CUDA scheduling/visibility claim is inferred from this CPU model alone.
    """
    cases = 0
    # Mirror the helper's actual q jump, including zero helper-owned A/B units.
    for nA in range(10):
        for nB in range(10):
            for k in (2,4):
                for bk in (1,2):
                    nh = nA+nB-min(bk,nB)
                    q, publishes, visited = 0, 0, []
                    while q < nh:
                        if nA > 0 and q == nA-min(k,nA):
                            publishes += 1
                            q = nA
                            if q >= nh: break
                        visited.append(q)
                        q += 1
                    assert publishes == bool(nA)
                    assert visited == list(range(nA-min(k,nA)))+list(range(nA,nh))
    for jobs in range(33):
        for k in (2, 4):
            total = [0] * jobs
            for c in range(132):
                a0, a1 = jobs*512*c//132, jobs*512*(c+1)//132
                b0, b1 = jobs*256*c//132, jobs*256*(c+1)//132
                nA, nB = a1-a0, b1-b0
                d = min(k, nA)
                prefix, delegated = list(range(nA-d)), [nA-1-w for w in range(d)]
                assert sorted(prefix+delegated) == list(range(nA))
                assert len({q % 4 for q in delegated}) == d
                # Both full+empty barriers see one arrival per A unit even
                # though the A-tail full arrives late (in the owning warp).
                full, empty = [0]*4, [0]*4
                for q in prefix:
                    assert full[q%4] == empty[q%4] == q//4
                    full[q%4] += 1; empty[q%4] += 1
                for q in delegated:
                    assert full[q%4] == empty[q%4] == q//4
                    full[q%4] += 1; empty[q%4] += 1
                for q in range(nA, nA+nB):
                    assert full[q%4] == empty[q%4] == q//4
                    full[q%4] += 1; empty[q%4] += 1
                # All helper/delegate lanes reach the same count, even nA<k.
                participants = set(range(256, 288)) | set(range(d*32))
                assert len(participants) == 32*(d+1)
                stores = {a0+q for q in prefix+delegated}
                if nA:
                    for j in range(a0//512, (a1-1)//512+1):
                        lo, hi = max(a0,j*512), min(a1,(j+1)*512)
                        assert all(u in stores for u in range(lo,hi))
                        total[j] += hi-lo
                cases += 1
            assert total == [512]*jobs
    return {'status':'PASS', 'cases':cases, 'jobs':list(range(33)), 'k':[2,4],
            'credit_per_job':512, 'new_CTA_exits':0, 'new_smem_bytes':0,
            'ring_full_empty_parity':'PASS; A-tail full advanced before empty release',
            'post_wait_coverage':[
                'All CTAs: helper physical 0..31/logical256..287 drains prefix, joins d=min(k,nA) compute warps, fences then lane0 credits each intersected job once. nA=0: no rendezvous/credit.',
                'Delegates logical0..32*d-1 (physical32..32*(d+1)-1) read final distinct slots after last compute CTA_SYNC; each lane stores its unchanged token pair; lane0 advances wsrf/wsre once; all lanes bar.arrive3 after stores.',
                'All CTAs logical0..255 retain refills and both phases. All 288 threads retain WS_JOIN2. Existing helper return remains after its final fence/join; no new returns.',
                'B tail unchanged: logical0..32*min(WSBLK,nB)-1 reduces/stores last B units; WSBLK2 at tiny,1 at tiny2; helper owns the disjoint B prefix.',
                'Before phases, other[i]=0: all CTAs logical0..255, i=c*256+tid, stride132*256, EMAX3072 => CTA0..11 cover all entries.',
                'CTA0 logical0 resets next-bank ctr[4+((gen+1)&1)]. Every CTA logical0 retains ctr[3] generation read and ss-bank selection. No new persistent counter or current-ss-bank reset.',
                'After final WS_JOIN, compute tid0 per CTA fences/increments bc and waits; last arrival old==131 increments ctr[3]; all256 compute threads CTA_SYNC before reset/combine.',
                'ctr[0:2] reset CTA0 logical0/1; doneA[0:JMAX=88] reset CTA0 logical0..87; helper excluded as in ship.',
                'Combine: every CTA logical0..255; it=c then c+132 strides to M*4. T1 CTA0..3; T8 CTA0..31 cover four1024-column segments per token; second loop has zero iterations at scored T1/8. No cleanup skipped for empty shares.'
            ]}


def accounting():
    return {'measured':{'source':'notes/lab.md:1222 R-tstag-2; box_out/tstag/stamps_ctl_dbg_d1s.json',
                       'W13_end_us':12.78, 'helper_publish_us':14.64, 'first_B_exit_us':15.38},
            'estimate':{'label':'EST; conditional target, not a benchmark result',
                        'us':'d1s headroom14.64-12.78=1.86; subtract0.80 measured flag-hop as a rendezvous allowance =>1.06us budget for delegated tail and net gain, NOT a measured or unconditional net saving',
                        'critical_path':'B doneA acquire cannot consume inter until all publishers complete; measured exit15.38 follows publish14.64. No launch/PDL edge removed.',
                        'kill_risks':['named barrier/fence drain consumes the backlog (R-dsm-0b/R-tstag-2)', 'register/occupancy loss from delegate epilogue (R-vmsw-1/2); no changed inner MMA stream']},
            'bytes_per_unit_before_after':{'HBM_weight':[36864,36864], 'red_write':[4096,4096],
                'red_read':[4096,4096], 'inter_store':'16*cnt B -> same (cnt1:16 B, cnt8:128 B)',
                'red_read_per_reducer_lane':[128,128], 'extra_global_bytes':0, 'extra_smem_capacity':0},
            'instructions_per_thread_EST_source_not_SASS':{
                'weight_MMA_loop':'before=after per compute lane/unit:1 scale uint4 LDS +8 weight uint4 LDS;16 SCDK +32 dq4 +128 bmul2 +32 mma_bf16 calls +12 FP32 accumulator ADD +1 float4 red STS. Arrival predicate adds one compare on logical tid0 in A; PB is compile-time. Builtins are source operations, not assumed one SASS instruction.',
                'reduction':'helper before:8 float4 LDS +32 FP32 ADD / lane / unit; after:identical on one delegate for each tail unit, zero on helper for those units',
                'epilogue':'unchanged per valid h:2 scale multiplies, negate/exp, add1, SILU_DIV, multiply, probe_q, BF16 cvt, BF16 store; h=0,1 mask unchanged. Actual exp/div SASS recorded by compile gate.',
                'sync':'tail before:compute lane0 full-arrive + helper full-wait + warp sync/empty-arrive; after:delegate warp sync + lane0 full/empty-arrive (same arrivals), helper d full-waits removed, plus32*(d+1) named-barrier participant instructions/CTA; one existing helper fence + credit loop retained. Helper q loop adds one tail-boundary equality test per visited unit; exact lowering pending SASS.',
                'hot_loop_SASS':'no manufactured dynamic instruction count; gate records full-function counts and actual SASS for both template instances'},
            'resources':'MEASURED existing box_out/tstag/gate.json tiny_ctl/tiny2_ctl:145 regs,0 stack/local/spills,both BIG instances. EST unchanged shared-memory allocation; tail may raise regs; gate rejects stack/local/spills and >224 regs. Candidate resources NOT MEASURED.',
            'routes':{
                'd1s/d1f':'k3_hh_5_sk2_pz_rt.py:7535 tiny,L2H0,CPF1,WSBLK2;M1<OKC_BIGM64(:6577) => moe_fused<false> PDL launch:7524;run_phase<false,false>:7385 => skinny_ring_ws<false>:7027',
                'v1s/v1f':'same:7535 tiny2,L2H1,CPF0,WSBLK1;M8<64 => same moe_fused<false>/A instance; B is skinny_ring_ws<true>. bank selector changes only (tiny,1)/(tiny2,8).'},
            'T8_measured':{'source':'box_out/tinymk/v1s.json,v1f.json pooled CTA median(dbg_ns[7]-dbg_ns[2])/1000',
                           'Aend_to_Bexit_us':[2.208,2.56],
                           'EST_budget_us':'subtract retained d1s acquire tail15.38-14.64=0.74 and rendezvous allowance0.80 =>0.668/1.020 budget for delegated tail plus net gain; acquire-tail transfer to T8 is an assumption'}}


def change_report(base):
    cu, start = literal(base, '_HYBLO_CUDA'), node(base, '_HYBLO_CUDA').lineno
    result = {}
    for arm in WSBLA_ARMS:
        parent = arm[:-4] if arm.endswith('_dbg') else 'ctl'
        before, after = _wsbla_source(cu,parent), _wsbla_source(cu,arm)
        result[arm] = {'relative_to':parent, 'changes':[
            {'before_cuda_lines':[a0+1,a1], 'after_cuda_lines':[b0+1,b1], 'kind':tag,
             'base_lines':([start+a0,start+a1-1] if parent=='ctl' and a1>a0 else []),
             'base_insert_after':(start+a0-1 if parent=='ctl' and a1==a0 else None),
             'before':'\n'.join(before.splitlines()[a0:a1]), 'after':'\n'.join(after.splitlines()[b0:b1])}
            for tag,a0,a1,b0,b1 in difflib.SequenceMatcher(None,before.splitlines(),after.splitlines(),autojunk=False).get_opcodes() if tag!='equal']}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    base = BASE.read_text(encoding='utf-8')
    source = build(base)
    with contextlib.redirect_stdout(io.StringIO()): dry_run(base,source)
    if a.check or a.dry_run:
        assert OUT.read_bytes() == source.encode(), 'regenerate WSBLA layer'
    else:
        OUT.write_bytes(source.encode())
    print(json.dumps({'status':'CPU_PASS', 'protocol':protocol_gate(), 'accounting':accounting(),
                      'changes':change_report(base), 'compile_SASS_GPU':'NOT RUN'}, indent=2))
    return 0


# CPU doubles copied from TSTAG/PADZ; real base AST/state layout retained.
def dry_run(base, patched, fixture=False):
    """Execute real _hyblo_prepare/_hyb_scratch and every new host branch.

    Only tensor arithmetic, prepack, reference values and compiler are doubles.
    State dictionaries, allocations' shapes, flags, build arguments and selectors
    are real AST nodes from the base and emitted layer; no permissive dicts.
    """
    class Tensor:
        device = "cpu-dry"
        def __init__(self, shape=(8, 4096)): self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        def __getitem__(self, k):
            if isinstance(k, slice):
                return Tensor((len(range(*k.indices(self.shape[0]))),) + self.shape[1:])
            return Tensor(self.shape)
        def __add__(self, x): return self
        __radd__ = __mul__ = __rmul__ = __eq__ = __add__
        def all(self): return True
        def numel(self):
            import math
            return math.prod(self.shape)
        def view(self, *s): return self if len(s) == 1 and isinstance(s[0], str) else Tensor(s)
        def reshape(self, *s): return Tensor(s)
        def to(self, *a): return self
        def contiguous(self): return self
        clone = float = contiguous
        def fill_(self, v): return self
        def data_ptr(self): return 16
    builds, calls, failures, build_names = [], [], set(), {}
    class Module:
        def __init__(self, key): self.key, self.flags, self.pdl = key, 0, None
        def okc_set_pdl(self, pdl): self.pdl = pdl
        def okc_run(self, *a):
            expected_sc = ns["_WSBLA"]["sc_dbg"] if self.key[1].endswith("_dbg") else ns["_HYBLO"]["sc"]
            assert a[0] is ns["_HYBLO"]["w"] and a[1] is expected_sc
            if self.key[1].endswith("_dbg"):
                assert all(a[1][i] is ns["_HYBLO"]["sc"][i] for i in range(3))
                assert a[1][3] is ns["_WSBLA"]["dbg"] and a[1][3].shape == (132*8,)
            assert a[-3:] == (0, 32, 132)
            calls.append((self.key, self.flags, self.pdl, a[2].shape[0]))
    torch = NS(**{s: s for s in ("int16", "int32", "int64", "float32", "bfloat16", "uint8")},
        empty=lambda s, **k: Tensor(s), zeros=lambda s, **k: Tensor(s),
        arange=lambda *a, **k: Tensor((a[0],)), stack=lambda *a, **k: Tensor(), cat=lambda *a, **k: Tensor(),
        isfinite=lambda t: t, cuda=NS(synchronize=lambda: None))
    ns = dict(torch=torch, time=time, hashlib=hashlib, _pt1_wait=lambda *a: None,
              _k2a_ref_fp32=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0., 0.))
    names = ("_HYB", "_HYB_MMAX", "_HYB_EMAX", "_HYB_FIXF", "_HYB_XROWS", "_HYB_GUARD",
             "_HYBLO", "_HYBLO_CUDA", "_HYB_MOE_CPP", "_HYBLO_FLAGS", "_HYBLO_KEYS", "HYBLO_ENABLE",
             "HYBLO_PDL", "HYBLO_T", "HYBLO_K", "HYBLO_SELF_TEST", "WSBLA_ON", "WSBLA_ARMS",
             "_WSBLA", "_WSBLA_DEBUG_EDITS", "WSBLA_SLOTS", "_wsbla_sub", "_wsbla_source",
             "_wsbla_build_one", "_wsbla_prepare", "_wsbla_select", "_hyb_scratch", "_hyb_guards_ok",
             "_hyblo_key", "_hyblo_build", "_hyblo_run", "_hyblo_route", "_hyblo_prepare")
    for name in names:
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), f"real:{name}", "exec"), ns)
    def load_inline(**kw):
        if kw["name"].startswith("hyblo_"):
            kind, k = "ctl", kw["name"].split("_")[1]
        else:
            _, k, rest = kw["name"].split("_", 2)
            kind, _ = rest.rsplit("_", 1)
        var = next(v for key, _, v in ns["_HYBLO_KEYS"] if key == k)
        expected = _wsbla_source(literal(base, "_HYBLO_CUDA"), kind)
        assert kw["cuda_sources"] == expected
        assert kw["extra_cuda_cflags"] == list(ns["_HYBLO_FLAGS"]) + list(var) + ([] if kind == "ctl" else ["-Xptxas=-v"])
        assert kw["functions"] == ["okc_run", "okc_set_pdl"]
        assert kw["cpp_sources"] == literal(base, "_HYB_MOE_CPP")
        builds.append((k, kind))
        if (k, kind) in failures: raise RuntimeError("injected build failure")
        return Module((k, kind))
    original_import = builtins.__import__
    ns["__builtins__"] = dict(vars(builtins), __import__=lambda n, *a, **k: NS(load_inline=load_inline) if n == "torch.utils.cpp_extension" else original_import(n, *a, **k))
    # Re-create functions with the restricted per-exec imports, never sys.modules.
    for name in ("_hyblo_build", "_wsbla_build_one"):
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), name, "exec"), ns)
    for name in ("_hyblo_pack_codes", "_hyblo_pack_scales", "_hyblo_fat_order", "_hyblo_unit_order"):
        ns[name] = lambda *a: Tensor()
    ns["_hyblo_inputs"] = lambda bs, *a: {k: Tensor((bs, 4096)) for k in ("hidden_states", "topk_ids", "topk_weights")}
    ns["_v11_k2a33_call"] = lambda *a: Tensor()
    ns["_HYB"].update(ok=True, w=[Tensor() for _ in range(14)], nsm=132, full=[])
    w = {k: Tensor() for k in ("w13", "w13_sf", "w2", "w2_sf")}
    ns["_hyblo_prepare"]({}, w, "cpu-dry")
    assert ns["_HYBLO"]["ok"] == {"tiny": True, "tiny2": True}
    assert len(ns["_HYBLO"]["sc"]) == 3 and len(ns["_HYBLO"]["w"]) == 14
    prepare_bank = lambda: ns["_wsbla_prepare"](w["w13"].device)
    prepare_bank(); prepare_bank()
    assert len(builds) == 10 and not ns["_WSBLA"]["errors"]
    def must_fail(f):
        try: f()
        except (RuntimeError, KeyError): return
        raise AssertionError("failure silently accepted")
    for key,t in (("tiny",1),("tiny2",8)):
        for arm in WSBLA_ARMS[1:]:
            ns["WSBLA_ON"] = arm
            must_fail(lambda: ns["_wsbla_select"](key,t))
    ns["_WSBLA"]["accepted"] = set(ns["_WSBLA"]["mods"])
    for t in (1, 2, 4, 5, 8):
        k = "tiny" if t <= 4 else "tiny2"
        for arm in WSBLA_ARMS:
            kind, flags = (arm if t in (1,8) else "ctl"), 0
            for pdl in (True, False, True):
                ns["WSBLA_ON"], ns["HYBLO_PDL"] = arm, pdl
                out = ns["_hyblo_route"](Tensor((t, 4096)), Tensor(), Tensor())
                assert out.shape == (t, 4096) and calls[-1] == ((k, kind), flags, int(pdl), t)
    ns["WSBLA_ON"] = "bad"; must_fail(lambda: ns["_wsbla_select"]("tiny", 1))
    for key,t in (("tiny",1),("tiny2",8)):
        for arm in WSBLA_ARMS[1:]:
            ns["WSBLA_ON"] = arm
            del ns["_WSBLA"]["mods"][(key,arm)]
            failures.add((key,arm)); prepare_bank()
            must_fail(lambda: ns["_wsbla_select"](key,t))
            failures.clear(); prepare_bank(); ns["_wsbla_select"](key,t)
    saved = ns["_HYBLO"]["sc"]; ns["_HYBLO"]["sc"] = None
    must_fail(prepare_bank); ns["_HYBLO"]["sc"] = saved
    ns["_HYBLO"]["ok"]["tiny2"] = False; must_fail(prepare_bank)
    ns["_HYBLO"]["ok"]["tiny2"] = True
    ns["HYBLO_ENABLE"] = False; prepare_bank()
    ns["HYBLO_ENABLE"] = True; ns["WSBLA_ON"] = "ctl"
    # Execute the new prepare call site against the REAL outer return layout.
    ns.update(st={}, state={}, weights={k: Tensor() for k in
        ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_expert_norm")}, t0=0)
    for name in ("K2A_M_MAX", "K2A_BUCKETS", "V11_K2A33_ROUTE"):
        exec(compile(ast.Module(body=[node(base, name)], type_ignores=[]), name, "exec"), ns)
    prep = node(base, "_v11_k2a33_prepare")
    ok_stmt = next(n for n in prep.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ok")
    exec(compile(ast.Module(body=[ok_stmt], type_ignores=[]), "real-ok", "exec"), ns)
    ret = next(n.value for n in prep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    k33 = eval(compile(ast.Expression(ret), "real-return", "eval"), ns)
    ns["_v11_k2a33_prepare"] = lambda *a: k33
    assignment = next(n for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_v11_k2a33_prepare")
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "real-state", "exec"), ns)
    site = next(l.strip() for l in patched.splitlines() if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    exec(site + '\n_wsbla_prepare(weights["w13"].device)', ns)
    # The actual layer dispatch must reach the HYBLO branch at both scored T.
    exec(compile(ast.Module(body=[node(base, "_v11_k2a33_call")], type_ignores=[]), "real-route", "exec"), ns)
    for t in (1, 8):
        for arm in WSBLA_ARMS:
            ns["WSBLA_ON"] = arm
            assert ns["_v11_k2a33_call"](Tensor((t, 4096)), Tensor(), Tensor(), k33["st"]).shape == (t, 4096)
    ns["WSBLA_ON"] = "ctl"
    print("[wsbla] CPU dry PASS: real prepare/scratch, ten builds, cache, all arms/PDL, missing gate/bank, independent failure/retry")
    if fixture:
        class ModuleProxy:
            def __getattr__(self, k):
                if k not in ns: raise AttributeError(k)
                return ns[k]
            def __setattr__(self, k, v): ns[k] = v
        return ModuleProxy(), ns["state"], Tensor

if __name__ == "__main__":
    raise SystemExit(main())

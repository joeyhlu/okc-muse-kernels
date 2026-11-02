"""HPF admission gate and explicitly disabled MOEXR lab vehicle.

The requested accumulator design is rejected by the task's byte-growth rule.
This generator deliberately does NOT implement or advertise a working HPF arm.
It leaves all CUDA, prepare bodies, and build flags identical to the real base.
Only the four task-named deliverables are written by the Windows workflow.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_moexr_ab.py"
LAB = ROOT / "kernels/lab/k3_hpf_ab.py"
REPORT = ROOT / "box_out/reset/hpf_report.md"
REASON = (
    "HPF dropped: progressive FP32 accumulation increases finalize traffic; "
    "T256 mod7 already retains its first-pass ybuf loads. "
    "The standing byte-growth rule prohibits this arm."
)
ANCHOR = 'def _moexr_bank():\n return _HYB if MOEXR else _MOEXR_SHIP\n'
INSERT = (
    '# HPF admission gate: no progressive-finalize CUDA build was admitted.\n'
    'HPF = False\n'
    f'_HPF = {{"ok": False, "reason": {REASON!r}}}\n'
    'def _moexr_bank():\n'
    ' if HPF:\n'
    '  raise RuntimeError(_HPF["reason"])\n'
    ' return _HYB if MOEXR else _MOEXR_SHIP\n'
)


def literal(source, name):
    hits = [n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(hits) == 1, (name, len(hits))
    return ast.literal_eval(hits[0])


def nodes(source, names):
    body = []
    for n in ast.parse(source).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            body.append(n)
    return compile(ast.Module(body=body, type_ignores=[]), '<hpf-cpu-ast>', 'exec')


def generate(base):
    assert base.count(ANCHOR) == 1, 'MOEXR selector anchor changed'
    assert 'HPF = ' not in base and '_HPF = ' not in base
    source = base.replace(ANCHOR, INSERT, 1)
    compile(source, str(LAB), 'exec')  # In memory: no .pyc or other output files.
    return source


def byte_gate():
    """Useful interface bytes, deliberately NOT assumed HBM transactions.

    R = real local routes, A = tokens with >=1 local route. Base BF16 output
    traffic cancels. A red payload counts only its 4-byte operand; any internal
    L2 read/modify/write traffic is additional, never credited as a saving.
    """
    b = 4096 * 4
    rows = []
    for t in (128, 256):
        for r, a, case in ((t, t, 'one local route/token'), (8*t, t, 'eight local routes/token')):
            reads = 2 if t == 128 else 1
            base = reads*b*r
            candidate = b*r + b*r + 2*b*a  # y read, red operand, acc read + replay reset
            delta = candidate-base
            assert delta > 0
            rows.append(dict(T=t, local_routes=r, active_tokens=a, case=case,
                             base_bytes=base, candidate_bytes=candidate, added_bytes=delta,
                             stream_equivalent_est_us=delta/4.64e6))
    return dict(verdict='DROP', bytes_per_fp32_row=b, rows=rows,
                scope='finalize payload only; weight-stage bytes unchanged',
                assumption='non-last/direct-output specializations are not part of the requested accumulator design')


def source_gates(base, source):
    bt, nt = ast.parse(base), ast.parse(source)
    bf = {n.name: ast.dump(n) for n in bt.body if isinstance(n, ast.FunctionDef)}
    nf = {n.name: ast.dump(n) for n in nt.body if isinstance(n, ast.FunctionDef)}
    assert set(bf) == set(nf)
    changed = [n for n in bf if bf[n] != nf[n]]
    assert changed == ['_moexr_bank'], changed
    # Stronger than checking a selected list: all original top-level nodes are
    # AST-identical except the explicitly guarded selector.
    added = {'HPF', '_HPF'}
    clean = [n for n in nt.body if not (isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id in added for t in n.targets))]
    assert len(bt.body) == len(clean)
    for before, after in zip(bt.body, clean):
        if isinstance(before, ast.FunctionDef) and before.name == '_moexr_bank':
            continue
        assert ast.dump(before) == ast.dump(after), getattr(before, 'lineno', None)
    assert literal(source, 'HPF') is False and literal(source, '_HPF')['ok'] is False
    assert source.replace(INSERT, ANCHOR, 1) == base
    assert literal(source, '_HYB_MOE_CUDA') == literal(base, '_HYB_MOE_CUDA')
    hunks = []
    sm = difflib.SequenceMatcher(a=base.splitlines(), b=source.splitlines(), autojunk=False)
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag != 'equal':
            hunks.append(dict(operation=tag, base_lines=[a0+1,a1] if a1>a0 else None,
                             before_base_line=a0+1, lab_lines=[b0+1,b1]))
    return dict(status='PASS', changed_functions=changed, changed_lines=hunks,
                cuda_and_build_flags_exact=True, new_cuda_builds=0,
                sha256=hashlib.sha256(source.encode()).hexdigest(), bytes=len(source.encode()))


def selector_dry_run(source):
    """Execute actual call/capture-time selectors with strict real bank schemas."""
    env = {}
    exec(nodes(source, {'HPF','_HPF','MOEXR','_HYB','_MOEXR_SHIP',
                        '_moexr_bank','_moexr_call_bank','_hyb_call','_hyb_on',
                        'HYB_ENABLE','HYB_T','HYB_T7','HYB_PDL'}), env)
    from types import SimpleNamespace
    seen = []
    class Mod:
        def __init__(self, name): self.name = name
        def okc_set_pdl(self, pdl): assert pdl in (0,1)
        def okc_run(self, w, sc, x, ids, wts, out, lo, nloc, grid):
            assert len(w) == 14 and len(sc) == 3 and (lo,nloc,grid) == (0,32,132)
            assert out.shape == (x.shape[0],4096)
            seen.append((env['MOEXR'],x.shape[0],self.name))
    env['torch'] = SimpleNamespace(bfloat16='bf16', empty=lambda shape,**kw:SimpleNamespace(shape=shape))
    for name in ('_HYB','_MOEXR_SHIP'):
        bank = env[name]
        assert set(bank) == {'mod','ok','w','sc','full','nsm','err','build_s','pdl','mod7','pdl7'}
        bank.update(ok=True, mod=Mod(name+'/mod'), mod7=Mod(name+'/mod7'),
                    w=list(range(14)), sc=list(range(3)), nsm=132)
    denied = 0
    for moexr in (False, True):
        env['MOEXR'] = moexr
        for t in (1,8,16,32,49,128,192,193,256,320,384):
            env['HPF'] = False
            assert env['_moexr_bank']() is env['_HYB' if moexr else '_MOEXR_SHIP']
            assert env['_hyb_on'](t) == (49 <= t <= 320)
            if env['_hyb_on'](t):
                x = SimpleNamespace(shape=(t,4096), device='cpu')
                assert env['_hyb_call'](x,None,None).shape == x.shape
                assert seen[-1][2].endswith('/mod7' if t >= 193 else '/mod')
            env['HPF'] = True
            try: env['_moexr_bank']()
            except RuntimeError as ex:
                assert str(ex) == REASON
                denied += 1
            else: raise AssertionError('rejected HPF must never masquerade as a live timing arm')
    return dict(status='PASS', baseline_launches=seen, denied_hpf_requests=denied,
                runtime_and_capture_use_same_selector=True)


def cpu_gates(base, source):
    # Reuse the already-proven MOEXR/padz-derived allocation-free harness.
    # It executes the real base _k2a33_prepare_st and prepare() hook; only the
    # tensor/CUDA leaves are doubles. No import of the actual GPU module.
    helpers = runpy.run_path(str(ROOT/'tools/moexr_patches.py'))
    from types import FunctionType
    original = helpers['dry_run']
    # Extend only this local harness's AST allowlist for the two new globals;
    # no imported module or base-layer object is patched.
    dry_fn = FunctionType(original.__code__, dict(original.__globals__,
                          nodes=lambda text,names: nodes(text,names | {'HPF','_HPF'})),
                          original.__name__, original.__defaults__)
    with contextlib.redirect_stdout(io.StringIO()):
        dry = dry_fn(base, source, False)
        integration = helpers['integration_dry_run'](base, source)
    return dict(source=source_gates(base,source), admission=byte_gate(),
                prepare_dry_run=dry, state_layout=integration,
                selectors=selector_dry_run(source))


def report(base, source, gates):
    def line(text, needle):
        hits = [i for i,v in enumerate(text.splitlines(),1) if needle in v]
        assert hits, needle
        return hits[0]
    hunks = '; '.join(f"insert before base line {h['before_base_line']} -> lab lines {h['lab_lines'][0]}-{h['lab_lines'][1]}"
                     for h in gates['source']['changed_lines'])
    text = f'''| item | shape(s) | MEASURED us | EST saving us | critical-path reason | closest DO_NOT_REPEAT row | verdict |
|---|---|---|---|---|---|---|
| Requested FP32 accumulator HPF | T=128 | LAD A=173.856, L=154.524, LF=150.716; paired L->LF=3.640 is explicitly VOID (box_out/lad/summary.md:7-13) | No admissible positive estimate; extra payload for R=A=128: 4,194,304/4.64e6=0.904 us stream-equivalent cost | LAD is a board diagnostic, not current smp timing; MFIN already removed the grid barrier | R-hybst-2; board-to-layer process trap | DROP: byte-growth rule; no enabled kernel |
| Requested FP32 accumulator HPF | T=256 | LAD A=208.420, L=192.216, LF=185.972; paired L->LF=6.244 (box_out/lad/summary.md:21-27) | No admissible positive estimate; extra payload for R=A=256: 12,582,912/4.64e6=2.712 us stream-equivalent cost | mod7 retains first-pass loads; no eliminated grid barrier or launch remains to credit; current MOEXR exposure unmeasured | R-b4m-0 stream geometry; conserved-coupling model | DROP: byte-growth rule; no enabled kernel |

Kill reasons first: (1) the proposed accumulator adds traffic even with perfect first-pass reuse; (2) last-expert ownership serializes all of a hot expert's rows on one CTA/WG and competes with unfinished B work. The first reason is sufficient under implementation rule 2. No performance gain is claimed.

Accounting (EST, exact source-derived payload counts, not measured HBM traffic): B=4096*4=16,384 bytes, R=real local routes, A=tokens with a local route. Excluding identical BF16 outputs (8192*T bytes), base128=2BR, base256=BR. Proposed HPF=BR y-read + BR red operand + BA accumulator read + BA replay reset. Delta128=2BA; delta256=BR+2BA, positive for any nonempty local routing. Internal atomic RMW traffic and counters are excluded, so this is generous to HPF. R=A=T above is a stated illustrative routing, not a measured profile. The measured 4.64 TB/s denominator gives stream-equivalent costs only: it does not prove these bytes reach HBM or lie on the critical path. No 1.9 us intercept is added because no launch is added/removed. No 0.84 us PDL edge or 0.80 us flag-hop saving is credited.

Before/after hot loops: no hot loop was changed. smp's SC6 weight stage remains 4*(8192+768)=35,840 bytes; A job 32*16 stages=18,350,080 bytes; B job 16*16 stages=9,175,040 bytes. Dynamic instructions/thread are IDENTICAL before/after because all CUDA and flags are byte-identical. No SASS instruction count is invented. Requested new finalizer instruction count is N/A (not admitted); rejected row traffic is accounted above.

Exact routes (base file kernels/lab/k3_moexr_ab.py): MOEXR=True selects _HYB at {line(base,'return _HYB if MOEXR else _MOEXR_SHIP')}; T128 uses mod, T256 uses mod7 at {line(base,'md,k=(H["mod7"]')}; smp flags SC6/MFIN/PLAN at {line(base,"fl=list(HYB_FLAGS)")}; HYB_F7=7/HN=1 at {line(base,'HYB_F7=')}. Both execute moe_fused<true> (BIG specialization, not the PDL flag), selected at {line(base,'if (p.M >= OKC_BIGM) cudaLaunchKernelEx')}; kernel definition at {line(base,'moe_fused(Params p)')}. wgs_phase<true> selects wgs_seg<true,NTT> via ntt=(jC+7)>>3, NTT=1..8 at {line(base,'switch (ntt)')}. NTT is routing-dependent, not fixed by T. Final pass starts at {line(base,'static_assert(HD == NT * 8')}; retained mod7 loads at {line(base,'ya[sl] = a; yb[sl] = b;')}. B's epilogue already multiplies (1+normw) at {line(base,'const float w0 = (1.f + __bfloat162float')}; doing it again would be incorrect.

Protocol audit (unchanged): every CTA's compute threads 0..511 participate in zeroing the other parity's 32 expert counters + 96 xready flags + JMAX*16 doneAs, i=CTA*512+tid, stride=G*512. JMAX=32+3072/64+8=88; the 1536 integers are covered by CTAs 0..2 when G=132. Helper threads 512..543 / 544..575 return after the unchanged producer routines and own none of that cleanup. Each WG's threads 128*w..128*w+127 resolve/zero its split buffer: thread wt owns floats 4*wt..4*wt+3 per nt at stride512, and wt=0 resets fc and publishes completion. After the MFIN wait, CTA threads 0..511 cover their tk=CTA+nG row, eight columns each; timeout path collectively writes NaNs then continues. Every CTA remains through its entire assigned cleanup and token loop; there are no new exits, counters, waits, reset paths, or combine loops.

Numerics: unchanged output algorithm; FP32/ship GPU comparisons NOT RUN. For the rejected design only, with fixed norm scales, finite normal arithmetic and no FTZ effect, u=2^-24 and gamma8=8u/(1-8u): different FP32 accumulation/product rounding gives absolute difference <=2*gamma8*sum(abs(r*y)) ~=9.54e-7*sum(abs(r*y)), before BF16 conversion. This is not a relative-error guarantee under cancellation, nor a bound for changed RMS reduction/underflow. No numerical gate is claimed passed for an unbuilt arm.

Deliverables: deterministic generator, disabled lab, Linux compile gate, this report. HPF=False preserves MOEXR exactly. HPF=True raises a clear rejection at call/capture selection, preventing accidental timing of an unchanged fallback as HPF. MOEXR defaults remain unchanged; an smp control still requires MOEXR=True. No HPF CUDA, extra scratch, prepare hook, or live A/B candidate exists.

Per-bit exact changes: {hunks}. HPF metadata starts at kernels/lab/k3_hpf_ab.py:{line(source,'HPF = False')}; the selector guard is at {line(source,' if HPF:')}. Every other original AST node, every CUDA literal and build flag is identical. Lab size is {len(source.encode()):,} bytes; this is a lab, not a submission.

CPU checks RUN/PASS: deterministic regeneration, Python compile in memory, AST/source identity, requested-form byte rejection; existing MOEXR real-state allocation/prepare dry run incl. failure paths; actual nested state constructor and hook (missing st key detected); call/capture selectors at both MOEXR settings and T=1,8,16,32,49,128,192,193,256,320,384. Numerical CUDA leaves are doubles, not accuracy tests. New prepare paths: none.

Not run: NVCC, cuobjdump, actual H200 numerics (normal/skewed/64+-token experts), graphs, timings. No GPU/pod/judge jobs, git writes, notes edits or secret reads. Rejection is a completed admission decision, NOT a completed HPF implementation.

Linux gate command (LF bash/Python polyglot; CPU/offline compile only):
```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" bash "<repo>/tools/local/hpf_compile.py"
```
The lead-run gate compiles unchanged HYB smp mod/mod7 from base and lab, records both BIG template instances' regs/spills/stack, requires equal SASS and resources, and writes tools/local/hpf_compile.json next to the script. Other kernels are certified source/flag-identical, not claimed SASS-tested. Temporary compile files are retained under /tmp; nothing is deleted. No new HYB builds exist to qualify. The JSON is a future requested gate result, not a file created in this implementer session.
'''
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    base = BASE.read_text(encoding='utf-8')
    source = generate(base)
    gates = cpu_gates(base,source)
    if args.check:
        assert LAB.read_bytes() == source.encode(), 'generated lab is stale'
    else:
        LAB.write_bytes(source.encode())
        REPORT.write_bytes(report(base,source,gates).encode())
    print(json.dumps({'status':'CPU PASS; HPF DROPPED', 'source':gates['source'],
                      'prepare':gates['prepare_dry_run']['status'],
                      'state_layout':gates['state_layout']['status'],
                      'denied_requests':gates['selectors']['denied_hpf_requests']}, indent=2))


if __name__ == '__main__':
    main()

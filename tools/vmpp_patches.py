"""R-vmpp-1 lab generator; CPU gates/dry run only, never imports the layer.

Kill risks: ptxas already overlaps decode/LDSM; a faster M family unbalances
the fitted plan. No measured speedup is claimed. RS/producers are untouched.
P1 deletes zero decode/LDSM/HMMA instructions. Same HBM stage bytes:
36,864 weights + 512*npad x bytes/CTA; per consumer lane 4 LDS.128 + 2
LDS.32 = 72 weight bytes, 8*(N/8) LDSM.x4 = 128*(N/8) x bytes,
8 dec_half2 and 16*(N/8) mma16816, before and after, all three variants.
P1: c 16->4 regs funds hh 8->16 and first-nt bb 4->8 (net 0 operands).
Rolling XOR adds one scalar; actual allocation/spills MUST pass ptxas gate.
P2: tail prefills c[4] only after last decode; hh/bb need only 12 regs then.
P3: rolled c[0] alone is prefetched (remaining codes still loaded once in
their groups); this avoids retaining 16 codes beside two live hh buffers.
P2/P3 add ready + token-folded prevx scalars; register reuse is not assumed.
The has-next predicate and tail index/pointer temporaries also need allocation;
the latter start after the current cp/i/ws operands die. Only ptxas decides fit.
P2/P3 per eligible stage add one try_wait/lane + warp vote + predicate;
hit moves 4/1 code loads (P2/P3), misses retain head wait, never extra loads.
All variants retain three syncwarp calls/stage; P1 moves the two enclosing
weight-empty release together. P2 adds the one all_sync vote at a tail peek.
All dynamic poll/address/control SASS counts remain UNMEASURED.

Protocol audit: every vm_main CTA; consumers tid 0..511, WG q's 128 lanes,
warp leaders tid 32*w (w=0..15), each owns one of NCW=16 empty arrivals.
P1 weight arrive moves after c3 LDS, after x-full wait and a warp join;
rolling XOR retains c0.x^c1.x^c2.x^c3.x^sw0^sw1 & zom dependency.
P2/P3 x arrive keeps tail join and folds xdep into saved slot offset;
next stage releases it after group 0 HMMA. First-stage flush before
pend_flush and final-stage flush drain it, including one-stage segments.
Tail peek only within a segment; a uniform warp hit acquires next weight
parity, misses use head wait. No early exit, slot advance or count change.
All zero-work CTAs keep init/phase exit; producer tid 512..575 unchanged.
Split resolve/clear: original last-arriving CTA/WG tid 128*q..+127 retains
vse_take and leader cnt reset; nonlast seg returns occur after both flushes.
vm_fin CTA0 tid 0..255 resets xready[0:80], doneA[0:1280], cnt[0:1088]
stride 256; token CTAs all 256 lanes combine/RMS/clear BRED/store 4096 cols.
No reset/combine loops or RS operands are touched. Same accumulator chain
order and final d2 addition; GPU bitwise/stale checks remain mandatory.
Existing bounded wait errors still continue through arrivals/cleanup and
set A.err; no new error exit is added, and self-tests reject their outputs.
"""
import ast
import contextlib
import io
import pathlib
import re
import sys
from types import SimpleNamespace as NS

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmpp_ab.py"
VARIANTS = tuple(f"bf2p{i}" for i in range(1, 4))


def sub(s, old, new, what):
    assert s.count(old) == 1, (what, s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    n, = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return ast.literal_eval(n.value)


GLOBALS = '''VMPP_ON = 0
VMPP_COST_SCALE = 1.0
VMPP_ARMS = None  # None = all live variants x (1.0, 0.93); runner batches <=7 arms.
_VMPP = {"calls": {}}
def vmpp_live(n):
    v = "bf2" if n == 0 else f"bf2p{n}"
    return bool(type(n) is int and 0 <= n <= 3 and _VML["mods"].get(v) is not None
                and _VML["ok"].get(32) and (n == 0 or _VML["okv"].get((32, v))))
'''

STAGE = '''#if VM_PP
static_assert(VM_PP >= 1 && VM_PP <= 3 && SFX && VM_SYNC7 && VM_DEFER
&& !VSE_HOLD && !VCP_LDA && !VCP_M1, "VM_PP requires bf2");
struct Pp {
uint4 c[(VM_PP & 1) ? 1 : 4];
int ready, prevx;
};
__device__ __forceinline__ void pp_flush(Pp& p, uint32_t bars, int lane) {
if (p.prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX) + p.prevx);
p.prevx = -1;
}
}
template <int N>
__device__ __forceinline__ void stage_m(const Args& A, uint8_t* smem, uint32_t bars, int& it,
float (&d2)[2][N / 2], int q, int wp, int lane, Pp& p, bool next) {
const int ws = it % NS, xs = it % NX;
#if VM_PP & 2
if (!p.ready) mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
#else
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
#endif
const int tile = (q & 1) * 4 + wp;
const uint8_t* st = smem + ws * US + (q >> 1) * SW;
const uint4* cp = reinterpret_cast<const uint4*>(st + tile * 2048) + lane;
const uint32_t* sp = reinterpret_cast<const uint32_t*>(st + 16384 + tile * 256) + lane;
#if VM_PP & 1
if (!p.ready) p.c[0] = cp[0];
#else
if (!p.ready) {
#pragma unroll
for (int i = 0; i < 4; ++i) p.c[i] = cp[32 * i];
}
#endif
p.ready = 0;
const uint32_t sw0 = sp[0], sw1 = sp[32], zom = (uint32_t)A.flags >> 30;
uint32_t wdep = p.c[0].x ^ sw0 ^ sw1;
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
#if !(VM_PP & 1)
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws) + ((wdep ^ p.c[1].x ^ p.c[2].x ^ p.c[3].x) & zom));
__syncwarp();
#endif
uint32_t xdep = 0u;
const uint32_t xb = sa(smem + NS * US + xs * XS) + (uint32_t)(((lane >> 3) * 8 + (lane & 7)) * 16);
const uint32_t k16q = 16u + zom;
uint32_t hh[(VM_PP & 1) ? 2 : 1][8], bb[(VM_PP & 1) ? 2 : 1][4];
#if VM_PP & 1
dec_half2(p.c[0].x, p.c[0].z, sw0, 0, hh[0], k16q, 64u, 4u);
ldsm4(xb, bb[0][0], bb[0][1], bb[0][2], bb[0][3]);
#endif
#pragma unroll
for (int g = 0; g < 8; ++g) {
const int i = g >> 1, hf = g & 1, b = (VM_PP & 1) ? (g & 1) : 0;
#if VM_PP & 1
if (g < 7) {
const int ng = g + 1, ni = ng >> 1, nh = ng & 1, nb = ng & 1;
if (!nh) {
p.c[0] = cp[32 * ni];
wdep ^= p.c[0].x;
if (ni == 3) {
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws) + (wdep & zom));
__syncwarp();
}
}
// Buffer nb was last consumed by g-1; no previous operands are overwritten.
dec_half2(nh ? p.c[0].y : p.c[0].x, nh ? p.c[0].w : p.c[0].z,
ni < 2 ? sw0 : sw1, ni & 1, hh[nb], k16q, 64u, 4u);
ldsm4(xb + (uint32_t)((ni * 8 + 4 * nh) * 128), bb[nb][0], bb[nb][1], bb[nb][2], bb[nb][3]);
}
#else
dec_half2(hf ? p.c[i].y : p.c[i].x, hf ? p.c[i].w : p.c[i].z,
i < 2 ? sw0 : sw1, i & 1, hh[0], k16q, 64u, 4u);
ldsm4(xb + (uint32_t)((i * 8 + 4 * hf) * 128), bb[0][0], bb[0][1], bb[0][2], bb[0][3]);
#endif
#if VM_PP & 2
if (g == 7 && next) {
const int ni = it + 1, nw = ni % NS;
int hit = __all_sync(0xffffffffu, mbar_try(bars + 8 * nw, (ni / NS) & 1));
if (hit) {
const uint4* np = reinterpret_cast<const uint4*>(smem + nw * US + (q >> 1) * SW + tile * 2048) + lane;
#pragma unroll
for (int k = 0; k < ((VM_PP & 1) ? 1 : 4); ++k) p.c[k] = np[32 * k];
}
p.ready = hit;
}
#endif
#pragma unroll
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
if (nt == 0) { b0 = bb[b][0]; b1 = bb[b][1]; b2 = bb[b][2]; b3 = bb[b][3]; }
else ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
mma16816(&d2[0][4 * nt], hh[b][0], hh[b][4], hh[b][1], hh[b][5], b0, b1);
mma16816(&d2[1][4 * nt], hh[b][2], hh[b][6], hh[b][3], hh[b][7], b2, b3);
if (i == 3) xdep ^= b0;
}
#if VM_PP & 2
if (g == 0) pp_flush(p, bars, lane);
#endif
}
__syncwarp();
#if VM_PP & 2
p.prevx = 8 * xs + (xdep & zom);
#else
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs) + (xdep & zom));
#endif
++it;
}
#else
'''

SEG = '''#if VM_PP
Pp pp; pp.ready = 0; pp.prevx = -1;
stage_m<N>(A, smem, bars, it, d2, q, wp, lane, pp, s0 + 1 < s1);
SEGMK(if (smk.on) smk.t1 = gtime();)
#if VM_PP & 2
pp_flush(pp, bars, lane);
#endif
if constexpr (VM_DEFER) pend_flush(A, pd, q, wp, lane);
#pragma unroll 1
for (int s = s0 + 1; s < s1; ++s) stage_m<N>(A, smem, bars, it, d2, q, wp, lane, pp, s + 1 < s1);
#if VM_PP & 2
pp_flush(pp, bars, lane);
#endif
#else
'''


def pp_zero(cu):
    """Project only our PP directives, preserving every original directive/byte."""
    cu = sub(cu, "#ifndef VM_PP\n#define VM_PP 0\n#endif\n", "", "default")
    lines, stack = [], []
    for line in cu.splitlines(True):
        t = line.strip()
        if t.startswith(("#if ", "#ifdef ", "#ifndef ")):
            own = "VM_PP" in t
            stack.append([own, False if own else True])
            if own:
                continue
        elif t == "#else":
            if stack[-1][0]:
                stack[-1][1] = not stack[-1][1]
                continue
        elif t == "#endif":
            own, _ = stack.pop()
            if own:
                continue
        if all(active for own, active in stack if own):
            lines.append(line)
    assert not stack
    return "".join(lines)


def build(base, report=False):
    assert "VM_PP" not in base
    cu = literal(base, "_vml_CUDA_T")
    edits = []
    def edit(a, b, name):
        edits.append((a, b, name))
    edit("namespace vm {\n", "namespace vm {\n#ifndef VM_PP\n#define VM_PP 0\n#endif\n", "PP default")
    a = cu.index("template <int N>\n__device__ __forceinline__ void stage_m(")
    b = cu.index("template <bool PB, int N>\n__device__ __forceinline__ void epilogue(", a)
    old = cu[a:b]
    edit(old, STAGE + old + "#endif\n", "P1/P2/P3 stage_m")
    a = cu.index("#if VSE_HOLD\nstage_m<N>")
    b = cu.index("SEGMK(if (smk.on) smk.t2 = smk.t3 = gtime();)\n", a)
    old = cu[a:b]
    edit(old, SEG + old + "#endif\n", "seg sync-path drains")
    edit("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "globals/live")
    edit("VML_ABVARS = ()\n", f"VML_ABVARS = {VARIANTS!r}\n", "build variants")
    line = next(l for l in base.splitlines(True) if l.startswith('             "bf2":'))
    add = ''.join(line.replace('"bf2":', f'"{v}":').replace('"-DVSE_PS=1"),', f'"-DVSE_PS=1", "-DVM_PP={n}"),')
                  for n, v in enumerate(VARIANTS, 1))
    edit(line, line + add, "exact bf2 define tuples")
    anchor = '    dp, sc = var in ("d", "d2"), var == "c"\n'
    edit(anchor, '''    if rep == 32 and _VML["force"] is None and VMPP_ON and vmpp_live(VMPP_ON):
        var = f"bf2p{VMPP_ON}"
    if var in ("bf2p1", "bf2p2", "bf2p3") and rep != 32:
        var = VML_VAR
    if rep == 32:
        _VMPP["calls"][var] = _VMPP["calls"].get(var, 0) + 1
    cost = VML_COST.get(rep)
    if rep == 32 and var in ("bf2p1", "bf2p2", "bf2p3") and VMPP_COST_SCALE != 1.0:
        cost = tuple(round(c * VMPP_COST_SCALE) if i % 8 < 2 else c for i, c in enumerate(cost))
''' + anchor, "route/counters/plan scale")
    anchor = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
    edit(anchor, anchor.replace('"bf2", "w72"', '"bf2", "bf2p1", "bf2p2", "bf2p3", "w72"'), "bf2 scratch selection")
    edit('flags=fl, cost=VML_COST.get(rep)', 'flags=fl, cost=cost', "cost argument")
    anchor = 'def _vml_prepare(st, w, dev):\n    _VML["ok"] = {}\n'
    edit(anchor, anchor + '    _VML["okv"] = {}\n', "invalidate variant validation")
    anchor = '    pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]\n'
    edit(anchor, anchor + '    pairs = [(r, v) for r, v in pairs if v not in ("bf2p1", "bf2p2", "bf2p3")] + [(32, v) for v in ("bf2p1", "bf2p2", "bf2p3")]\n', "rep32 self-test pairs")
    anchor = '                    _VML["force"] = var\n                    o = _vml_route(st, d)\n'
    edit(anchor, '''                    if var in ("bf2p1", "bf2p2", "bf2p3"):
                        _VML["scb"]["xB"].fill_(float("nan"))
''' + anchor, "poison xB before every variant test")
    s = base
    for a, b, name in edits:
        s = sub(s, a, b, name)
    ast.parse(s)
    assert pp_zero(literal(s, "_vml_CUDA_T")) == cu, "VM_PP=0 source identity"
    for n, v in enumerate(VARIANTS, 1):
        assert literal(s, "_VML_DEFS")[v] == literal(base, "_VML_DEFS")["bf2"] + (f"-DVM_PP={n}",)
    restored = s
    for a, b, name in reversed(edits):
        restored = sub(restored, b, a, "undo " + name)
    assert restored == base
    if report:
        for a, b, name in edits:
            line = s[:s.index(b)].count("\n") + 1
            print(f"[vmpp] {name}: {OUT.name}:{line}-{line + b.count(chr(10))}")
    return s


def dry_run(base, patched):
    """Execute real AST-selected prepare/build/route with strict tensor doubles."""
    tree = ast.parse(patched)
    def name(n):
        return n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None
    wanted = {"_VML", "_VMPP", "VMPP_ON", "VMPP_COST_SCALE", "VMPP_ARMS", "vmpp_live", "_vml_build_job", "_vml_prepare", "_m32e_vartest", "_vml_route"}
    wanted |= {name(n) for n in tree.body if isinstance(name(n), str) and name(n).startswith("VML_")}
    ns = {}
    exec(compile(ast.Module(body=[n for n in tree.body if name(n) in wanted], type_ignores=[]), "vmpp-dry", "exec"), ns)
    alloc = next(n for n in ast.parse(base).body if isinstance(n, ast.FunctionDef) and n.name == "_vml_alloc")
    keys = ast.literal_eval(ast.List(elts=next(n.value.keys for n in ast.walk(alloc) if isinstance(n, ast.Return)), ctx=ast.Load()))
    class Tensor:
        def __init__(self, rows=32): self.shape = (rows, 4096); self.poison = False
        def contiguous(self): return self
        def to(self, *args): return self
        def float(self): return self
        def clone(self): return Tensor(self.shape[0])
        def zero_(self): self.poison = False; return self
        def fill_(self, value): assert value != value; self.poison = True; return self
        def __getitem__(self, key): return 0
    def alloc_stub(*args): return {k: Tensor() for k in keys}
    poisoned = []
    def call(mod, st, sc, x, *args, **kw):
        assert st is state and sc is ns["_VML"]["scb"] or (st is state and mod == "w72" and sc is ns["_VML"]["scw"])
        if ns["_VML"]["force"] in VARIANTS:
            assert sc["xB"].poison
            poisoned.append((mod, x.shape[0]))
        sc["xB"].poison = False
        return Tensor(x.shape[0])
    def inputs(bs, *args): return {k: Tensor(bs) for k in ("hidden_states", "topk_ids", "topk_weights")}
    # Real st is passed opaquely by every new path; no invented nested state accessor.
    state = object()
    ns.update(torch=NS(int32=0, float32=1, zeros_like=lambda t: Tensor(),
              isfinite=lambda t: NS(all=lambda: True), cuda=NS(synchronize=lambda: None,
              get_device_properties=lambda dev: NS(multi_processor_count=132))),
              _pt1_wait=lambda name: None, _vml_SLOTS=136, _vml_alloc=alloc_stub,
              _k2a_self_test_inputs=inputs, K2A_E_LOC=80, _k2a_ref_fp32=lambda *a: Tensor(),
              _v11_k2a33_call=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0, 0),
              _k2a33_reset=lambda st: None, _vml_call=call, _vml_FLAGS_DEFAULT=193,
              _nb_vml_rep=lambda n: 32 if n <= 48 else 128 if n <= 160 else 256)
    defs = literal(patched, "_VML_DEFS")
    ns["_VML_DEFS"] = defs
    failed = set()
    def build_stub(defines):
        var = next(k for k, v in defs.items() if v == defines)
        if var in failed: raise RuntimeError("injected independent build failure")
        return var, 0
    ns["_vml_build"] = build_stub
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml_prepare"](state, {}, "cpu")
    assert ns["_VML"]["err"] is None and all(ns["vmpp_live"](n) for n in range(4))
    assert len(poisoned) == 18 and all(bs in (25, 32, 48) for _, bs in poisoned)
    def fault_call(*args, **kwargs):
        if ns["_VML"]["force"] == "bf2p2": raise ValueError("injected self-test failure")
        return call(*args, **kwargs)
    ns["_vml_call"] = fault_call
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml_prepare"](state, {}, "cpu")
    assert ns["vmpp_live"](1) and not ns["vmpp_live"](2) and ns["vmpp_live"](3)
    ns["_vml_call"] = call
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_m32e_vartest"](state, {}, "cpu")
    seen = []
    ns["_vml_call"] = lambda mod, st, sc, x, *a, **kw: seen.append((mod, sc, kw["cost"]))
    for n in range(4):
        for scale in (1.0, 0.93):
            ns.update(VMPP_ON=n, VMPP_COST_SCALE=scale)
            ns["_vml_route"](state, dict(inputs(32), config={"local_start": 0}))
            mod, sc, cost = seen[-1]
            assert mod == ("bf2" if n == 0 else f"bf2p{n}") and sc is ns["_VML"]["scb"]
            assert list(cost) == [round(c * scale) if n and i % 8 < 2 else c for i, c in enumerate(ns["VML_COST"][32])]
    ns["_vml_route"](state, dict(inputs(128), config={"local_start": 0}))
    assert seen[-1] == ("w72", ns["_VML"]["scw"], ns["VML_COST"][128])
    ns["_VML"]["force"] = "bf2p3"
    ns["_vml_route"](state, dict(inputs(128), config={"local_start": 0}))
    assert seen[-1][0] == "bf2"
    ns["_VML"]["force"] = None
    for n in (1, 2, 3):
        ns["_VML"]["okv"][(32, f"bf2p{n}")] = False
        ns["VMPP_ON"] = n
        ns["_vml_route"](state, dict(inputs(32), config={"local_start": 0}))
        assert seen[-1][0] == "bf2" and not ns["vmpp_live"](n)
    failed.add("bf2p2"); ns["_VML"]["mods"] = {}; ns["_vml_build_job"]()
    assert "bf2p2" not in ns["_VML"]["mods"] and all(v in ns["_VML"]["mods"] for v in ("bf2", "bf2p1", "bf2p3"))
    ns["VML_ENABLE"] = False; ns["_vml_prepare"](state, {}, "cpu")
    assert not ns["_VML"]["okv"] and not ns["vmpp_live"](0)
    print("[vmpp] CPU dry run PASS: actual build/prepare/vartest/route; 18 NaN poisons/full pass; scales; fallback; independent build/self-test failures")


def main():
    assert len(sys.argv) == 1, "python -B -I tools/vmpp_patches.py"
    raw = BASE.read_bytes()
    base = raw.decode("utf-8").replace("\r\n", "\n")
    s = build(base, report=True)
    dry_run(base, s)
    out = s.encode("utf-8")
    OUT.write_bytes(out)
    print(f"[vmpp] {len(out):,} B; headroom {1_000_000-len(out):,} B")
    if len(out) > 1_000_000:
        print(f"[vmpp] OVER by {len(out)-1_000_000} B: retained original PP=0 body plus three lab variants")
    print("[vmpp] anchors/AST/exact reversal/PP0 CUDA identity/defines PASS")


if __name__ == "__main__":
    main()

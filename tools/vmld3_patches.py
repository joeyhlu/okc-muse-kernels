"""R-vmld-3: compile-time stage_m doses; CPU-only generator and dry run.

Kill risks FIRST: compiler scheduling/allocation may absorb a cut; reducing one
stage family may worsen contention (R-vmld-1 / R-vmln-1). This is a null probe,
not a speedup proposal. No ship saving is inferred from guarded-binary timings.

Source accounting (EST source ops, not measured SASS), bf2 N=8/16, lane/stage:
HBM unchanged: 36,864 weight B + 512*npad x B/CTA; code/scale LDS 72 B/lane.
Class 0 with K bits: dec 8->8-K calls (216->27*(8-K) core integer/BF16 ops),
LDSM.x4 N->N*(8-K)/8, MMA 2*N->N*(8-K)/4; x LDS 16*N->2*N*(8-K) B/lane.
z1/z7 K=1; z4 K=4; z8 K=8. No new runtime tests/branches/synchronizations.
Class 1: 27 dec ops/group -> 8 register assignments, other work/bytes unchanged.
Class 2: N/8 LDSM and 2*N B/group -> 4*(N/8) register assignments, no new bytes.
Class 3: N/4 MMA/group -> 8 + 4*(N/8) XOR folds into xdep; all operands live
through the existing SFX arrival address. Classes 1/2/3 supported, not built.
Only machine SASS can resolve folded ops/moves, spills or actual instructions.

Protocol proof: no CTA/warp exits or moved waits. All main CTAs 0..G-1 keep
consumers tid 0..511 (q=0..3, 128 lanes/WG); tids 0,32,..480 keep all 16
weight-empty/x-empty arrivals, three joins, ++it. Last i==3 xdep XOR stays
per retained group; both g6/g7 absent (class 0) leaves xdep initialized to 0.
Producers tid 512..575, segment pend_flush, d2 combine k=0..N/2-1 and
epilogue/last-arriver split-resolve/cnt reset retain original CTA/WG coverage.
No cut encloses cleanup: vm_fin token CTAs 0..M-1, tid 0..255 combine/RMS,
clear BRED/y, store 4096 cols; CTA0 tid-stride-256 resets xready[0:80],
doneA[0:1280], cnt[0:1088]. bf2 VM_DP=0; no DP gbar cleanup exists.

Prepare budget: MEASURED box_out/vmld/run.log:474 136.0 s, postprepare 24 s,
setup 11 s. No vmpp log: EST 4*20=80 s extra -> 251 s runner (<270 s).
Six adds 120 s plus ~6.4 s energy -> ~297 s runner; omit optional zd/zm.
PT1 runs these builds in parallel; additive allowance is deliberately cautious.
"""
import ast
import contextlib
import hashlib
import io
import pathlib
import runpy
import sys
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmld3_ab.py"
SPECS = (("z1", 0x01, 0), ("z7", 0x80, 0), ("z4", 0x0F, 0), ("z8", 0xFF, 0))
VARIANTS = tuple("bf2" + name for name, _, _ in SPECS)
DEFAULTS = "#ifndef VM_SKIPM\n#define VM_SKIPM 0\n#endif\n#ifndef VM_SKIPC\n#define VM_SKIPC 0\n#endif\n"

GLOBALS = '''VMLD3_ON = 0
VMLD3_VARIANTS = ("bf2z1", "bf2z7", "bf2z4", "bf2z8")
_VMLD3 = {"calls": {}}
def vmld3_live(n):
    if type(n) is not int or not 0 <= n <= len(VMLD3_VARIANTS):
        return False
    v = "bf2" if n == 0 else VMLD3_VARIANTS[n - 1]
    return bool(_VML["mods"].get(v) is not None and _VML["ok"].get(32)
                and (n == 0 or _VML["okv"].get((32, v))))
'''

# Macro expansion gives literal constexpr i/hf, unlike a pragma-unrolled int
# loop (which cannot be used as an if-constexpr condition). No lambda/helper
# call or volatile scheduling fence is introduced between surviving groups.
GROUP = '''{
constexpr int i = I, hf = HF;
constexpr bool cut = (VM_SKIPM & (1 << (2 * i + hf))) != 0;
if constexpr (!cut || VM_SKIPC != 0) {
const uint32_t sw2 = (i < 2) ? sw0 : sw1, par = i & 1;
const uint32_t k16q = 16u + ((uint32_t)A.flags >> 30);
uint32_t hh[8];
if constexpr (cut && VM_SKIPC == 1) {
hh[0] = hh[4] = c[i].x; hh[1] = hh[5] = c[i].y;
hh[2] = hh[6] = c[i].z; hh[3] = hh[7] = c[i].w;
} else {
dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);
}
if constexpr (cut && VM_SKIPC == 3) {
_Pragma("unroll")
for (int j = 0; j < 8; ++j) xdep ^= hh[j];
}
_Pragma("unroll")
for (int nt = 0; nt < N / 8; ++nt) {
uint32_t b0, b1, b2, b3;
if constexpr (cut && VM_SKIPC == 2) {
b0 = c[i].x; b1 = c[i].y; b2 = c[i].z; b3 = c[i].w;
} else {
ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);
}
if constexpr (cut && VM_SKIPC == 3) {
// Add b1..b3 first; b0 uses the original i==3 token fold below.
xdep ^= b1 ^ b2 ^ b3;
if constexpr (i != 3) xdep ^= b0;
} else {
mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);
mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);
}
if (i == 3) xdep ^= b0;
}
}
}
'''


def sub(s, old, new, what):
    assert s.count(old) == 1, (what, s.count(old))
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, name
    return ast.literal_eval(nodes[0].value)


def skip_zero(cu):
    """Project ONLY our preprocessor branch; retain original source bytes."""
    cu = sub(cu, DEFAULTS, "", "skip defaults")
    a, b = cu.split("#if VM_SKIPM\n", 1)
    _, b = b.split("#else // VM_SKIPM\n", 1)
    old, tail = b.split("#endif // VM_SKIPM\n", 1)
    return a + old + tail


def build(base, report=False):
    assert "VM_SKIPM" not in base and "VMLD3" not in base
    cu = literal(base, "_vml_CUDA_T")
    a = cu.index("template <int N>\n__device__ __forceinline__ void stage_m(")
    b = cu.index("template <bool PB, int N>\n__device__ __forceinline__ void epilogue(", a)
    old_m = cu[a:b]
    begin = old_m.index("#pragma unroll\nfor (int i = 0; i < 4; ++i) {\n")
    end = old_m.index("__syncwarp();\n#if SFX", begin)
    old_loop = old_m[begin:end]
    macro = "#define VM_MGROUP(I, HF) " + " \\\n".join(GROUP.rstrip().splitlines()) + "\n"
    # A // comment would swallow the rest of a backslash-continued macro.
    macro = macro.replace("// Add b1..b3 first; b0 uses the original i==3 token fold below.",
                          "/* b0 uses the original i==3 token fold below. */")
    new_loop = ("#if VM_SKIPM\n"
                'static_assert(VM_SKIPM > 0 && VM_SKIPM <= 255 && VM_SKIPC >= 0 && VM_SKIPC <= 3, "dose range");\n'
                'static_assert(SFX && VM_SYNC7 && VM_DEFER && !VSE_HOLD && !VCP_LDA && !VCP_M1, "bf2 only");\n'
                + macro + "".join(f"VM_MGROUP({i}, {hf});\n" for i in range(4) for hf in range(2))
                + "#undef VM_MGROUP\n#else // VM_SKIPM\n" + old_loop + "#endif // VM_SKIPM\n")
    new_m = sub(old_m, old_loop, new_loop, "constexpr groups")
    assert new_m[:begin] == old_m[:begin] and new_m.endswith(old_m[end:])
    assert not any(x in GROUP for x in ("mbar_", "__syncwarp", "return", "break;", "continue;", "++it"))
    edits = [("namespace vm {\n", "namespace vm {\n" + DEFAULTS, "defaults"),
             (old_m, new_m, "stage_m compile-time groups"),
             ("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "globals/live"),
             ("VML_ABVARS = ()\n", f"VML_ABVARS = {VARIANTS!r}\n", "parallel builds")]
    line = next(l for l in base.splitlines(True) if l.startswith('             "bf2":'))
    added = "".join(line.replace('"bf2":', f'"bf2{name}":').replace('"-DVSE_PS=1"),',
                     f'"-DVSE_PS=1", "-DVM_SKIPM={mask}", "-DVM_SKIPC={cls}"),') for name, mask, cls in SPECS)
    edits.append((line, line + added, "variant defines"))
    anchor = '    dp, sc = var in ("d", "d2"), var == "c"\n'
    edits.append((anchor, '''    if rep == 32 and _VML["force"] is None and VMLD3_ON and vmld3_live(VMLD3_ON):
        var = VMLD3_VARIANTS[VMLD3_ON - 1]
    if var in VMLD3_VARIANTS and rep != 32:
        var = VML_VAR
    if rep == 32:
        _VMLD3["calls"][var] = _VMLD3["calls"].get(var, 0) + 1
''' + anchor, "route/counters"))
    anchor = '    scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]\n'
    edits.append((anchor, anchor.replace('"w72")', '"w72") + VMLD3_VARIANTS'), "bf2 scratch"))
    anchor = 'def _vml_prepare(st, w, dev):\n    _VML["ok"] = {}\n'
    edits.append((anchor, anchor + '    _VML["okv"] = {}\n', "invalidate liveness"))
    anchor = '    pairs = [(rep, v) for rep in VML_BVAR for v in dict.fromkeys((VML_BVAR[rep],) + tuple(VML_ABVARS))]\n'
    edits.append((anchor, anchor + '''    pairs = [(r, v) for r, v in pairs if v not in VMLD3_VARIANTS]
    # Null variants intentionally fail output checks; eager runner checks exceptions.
    for v in VMLD3_VARIANTS:
        _VML["okv"][(32, v)] = bool(_VML["ok"].get(32) and v in _VML["mods"])
''', "null liveness (not numerical validation)"))
    # Use the ORIGINAL raw source/hash/cache name for bf2 and other exact builds.
    # This is stronger than only preprocessing away our mask-zero branch.
    helper = '''def _vmld3_source(defines):
    if any(d.startswith("-DVM_SKIPM=") and int(d.split("=", 1)[1], 0) for d in defines):
        return _vml_CUDA
    a, b = _vml_CUDA.split("#if VM_SKIPM\\n", 1)
    _, b = b.split("#else // VM_SKIPM\\n", 1)
    old, tail = b.split("#endif // VM_SKIPM\\n", 1)
    return (a + old + tail).replace(DEFAULTS_LITERAL, "", 1)
'''.replace("DEFAULTS_LITERAL", repr(DEFAULTS))
    tree = ast.parse(base)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vml_build")
    original = ast.get_source_segment(base, fn) + "\n"
    changed = original.replace('    fl = _vml_FLAGS + list(defines)\n',
                               '    fl = _vml_FLAGS + list(defines)\n    cu = _vmld3_source(defines)\n')
    changed = changed.replace("_vml_CPP + _vml_CUDA", "_vml_CPP + cu").replace("cuda_sources=_vml_CUDA", "cuda_sources=cu")
    edits.append((original, helper + changed, "shipped raw source/cache identity"))
    s = base
    for old, new, what in edits:
        s = sub(s, old, new, what)
    ast.parse(s)
    assert skip_zero(literal(s, "_vml_CUDA_T")) == cu
    for (name, mask, cls), v in zip(SPECS, VARIANTS):
        assert literal(s, "_VML_DEFS")[v] == literal(base, "_VML_DEFS")["bf2"] + (f"-DVM_SKIPM={mask}", f"-DVM_SKIPC={cls}")
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base
    assert len(s.encode()) <= 1_000_000
    if report:
        for old, new, what in edits:
            line = s[:s.index(new)].count("\n") + 1
            print(f"[vmld3] {what}: {OUT.name}:{line}-{line + new.count(chr(10)) - 1}")
        for name, mask, cls in SPECS:
            print(f"[vmld3] {name}: mask={mask:#04x} class={cls}; g={[g for g in range(8) if mask >> g & 1]}")
    return s


def dry_run(base, patched):
    """Execute real build/prepare/bank/route/call AST on base-derived key layout."""
    model = runpy.run_path(str(ROOT / "tools/vmld_patches.py"))
    selected = model["selected"]
    class Tensor(model["Tensor"]):
        def float(self): return self
        def clone(self): return Tensor(self.shape, self.dtype)
        def zero_(self): return self
        def __getitem__(self, key): return 0
        def to(self, dtype): return Tensor(self.shape, dtype)
    torch = NS(int32="i32", int64="i64", float32="f32", bfloat16="bf16",
               zeros=Tensor, empty=Tensor, zeros_like=lambda t: Tensor(t.shape, t.dtype),
               isfinite=lambda t: NS(all=lambda: True),
               cuda=NS(synchronize=lambda: None, get_device_properties=lambda dev: NS(multi_processor_count=132)))
    funcs = {"_vml_build_job", "_vml_prepare", "_m32e_vartest", "_vml_alloc", "_vml_mag", "_vml_route", "_vml_call",
             "_vml_cfg", "_vml_cost_ext", "_vml_pool_ext", "_vml_seg_ext", "_vml_pdel_ext", "_nb_vml_rep",
             "_k2a_bucket_of", "vmld3_live"}
    ns = selected(patched, funcs, {"torch": torch})
    tree = ast.parse(base)
    prep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_k2a33_prepare_st")
    ctor = next(n.value for n in prep.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "st" for t in n.targets))
    state = {kw.arg: Tensor() for kw in ctor.keywords}
    seen, failed = [], set()
    cfg = (64, 8, 3, 64, 32, 212912, 0, 0, 1, 0, 0)
    def build_stub(defines):
        v = next(k for k, d in ns["_VML_DEFS"].items() if d == defines)
        if v in failed: raise RuntimeError("injected build failure")
        return NS(vml_cfg=lambda: cfg, vml_run=lambda tensors, values: seen.append((v, tensors, values))), 0
    def inputs(bs, *args):
        return {"hidden_states": Tensor((bs, 4096)), "topk_ids": Tensor((bs, 8)), "topk_weights": Tensor((bs, 8))}
    ns.update(_vml_build=build_stub, _pt1_wait=lambda name: None, _k2a_self_test_inputs=inputs,
              _k2a_ref_fp32=lambda *a: Tensor(), _v11_k2a33_call=lambda *a: Tensor(),
              _k2a_rel_err=lambda *a: (0., 0.), _k2a33_reset=lambda st: None)
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml_prepare"](state, {}, "cpu")
    assert ns["_VML"]["err"] is None and all(ns["vmld3_live"](n) for n in range(5))
    assert all(v not in VARIANTS for v, _, _ in seen), "nulls must not enter exact prepare tests"
    scb, scw = ns["_VML"]["scb"], ns["_VML"]["scw"]
    for n in range(5):
        ns["VMLD3_ON"] = n
        for bs in (25, 32, 48, 128, 256):
            ns["_vml_route"](state, dict(inputs(bs), config={"local_start": 0}))
            v, tensors, values = seen[-1]
            expect = ("bf2" if n == 0 else VARIANTS[n - 1]) if bs <= 48 else "w72"
            assert v == expect and any(t is (scb if bs <= 48 else scw)["xB"] for t in tensors)
            assert values[5] == 193, "no launch-flag injection"
    assert all(ns["_VMLD3"]["calls"].get(v, 0) for v in ("bf2",) + VARIANTS)
    # Forced self-test arms still obey rep32-only restriction.
    ns["_VML"]["force"] = VARIANTS[0]
    ns["_vml_route"](state, dict(inputs(128), config={"local_start": 0}))
    assert seen[-1][0] == "bf2"
    ns["_VML"]["force"] = None
    for n, v in enumerate(VARIANTS, 1):
        ns["_VML"]["okv"][(32, v)] = False
        ns["VMLD3_ON"] = n
        ns["_vml_route"](state, dict(inputs(32), config={"local_start": 0}))
        assert seen[-1][0] == "bf2" and not ns["vmld3_live"](n)
    ns["VMLD3_ON"] = 0
    failed.add(VARIANTS[1]); ns["_VML"]["mods"] = {}; ns["_VML"]["mod"] = None
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml_prepare"](state, {}, "cpu")
    assert not ns["vmld3_live"](2) and all(ns["vmld3_live"](n) for n in (0, 1, 3, 4))
    ns["VML_ENABLE"] = False; ns["_vml_prepare"](state, {}, "cpu")
    assert not ns["_VML"]["okv"] and not ns["vmld3_live"](0)
    # Original raw CUDA/CPP/flags/hash identity, plus every source-select path.
    funcs = {"_vml_wg_spec", "_vml_wg_spec64", "_vmld3_source", "_vml_CUDA", "_vml_CPP", "_vml_FLAGS", "_VML_DEFS"}
    new = selected(patched, funcs)
    old = selected(base, funcs - {"_vmld3_source"})
    for v in ("bf2", "w72"):
        cu = new["_vmld3_source"](new["_VML_DEFS"][v])
        assert cu == old["_vml_CUDA"]
        def tag(env, cuda):
            return hashlib.sha1((env["_vml_CPP"] + cuda + " ".join(env["_vml_FLAGS"] + list(env["_VML_DEFS"][v]))).encode()).hexdigest()[:10]
        assert tag(new, cu) == tag(old, old["_vml_CUDA"])
        print(f"[vmld3] {v} shipped source/flags/cache identity PASS: vml_moe_{tag(new, cu)}")
    assert new["_vmld3_source"](("-DVM_SKIPM=0", "-DVM_SKIPC=3")) == old["_vml_CUDA"]
    for v in VARIANTS:
        assert new["_vmld3_source"](new["_VML_DEFS"][v]) == new["_vml_CUDA"]
    # Execute the edited builder itself, replacing only its CUDA-library import
    # by a CPU call recorder; no sys.modules edits or layer imports.
    fn = next(n for n in ast.parse(patched).body if isinstance(n, ast.FunctionDef) and n.name == "_vml_build")
    assert isinstance(fn.body[0], ast.ImportFrom) and fn.body[0].module == "torch.utils.cpp_extension"
    fn.body = fn.body[1:]
    builds = []
    new.update(hashlib=hashlib, time=NS(time=lambda: 0), load_inline=lambda **kw: builds.append(kw) or NS())
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<actual builder with CPU compiler>", "exec"), new)
    for v in ("bf2", "w72") + VARIANTS:
        new["_vml_build"](defines=new["_VML_DEFS"][v])
        rec = builds[-1]
        assert rec["cuda_sources"] == new["_vmld3_source"](new["_VML_DEFS"][v])
        assert rec["extra_cuda_cflags"] == new["_vml_FLAGS"] + list(new["_VML_DEFS"][v])
        if v == "bf2": assert rec["name"] == "vml_moe_f06956a044"
    # Leave a fully prepared CPU bank for the runner's real eager/timing dry run.
    failed.clear(); ns["VML_ENABLE"] = True; ns["_VML"]["mod"] = None
    with contextlib.redirect_stdout(io.StringIO()):
        ns["_vml_prepare"](state, {}, "cpu")
    print("[vmld3] CPU dry run PASS: real base state/banks/prepare/vartest/call, all arms, force/fallback/build failure/disabled")
    return ns, state


def main():
    assert len(sys.argv) == 1, "python -B -I tools/vmld3_patches.py"
    base = BASE.read_text(encoding="utf-8")
    source = build(base, report=True)
    dry_run(base, source)
    OUT.write_bytes(source.encode("utf-8"))
    print(f"[vmld3] {len(source.encode()):,} B; AST/reversal/VM_SKIPM=0 CUDA source identity PASS")


if __name__ == "__main__":
    main()

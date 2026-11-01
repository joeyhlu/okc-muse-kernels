"""R-vmld-2: stage_m position/class scan; lab only, copied from vmld.

Run: python -B -I tools/vmld2_patches.py (generator + CPU gates + dry run).
No layer import, CUDA call, files beyond the fixed output, or GPU job.

Kill risks: stage_m-only removal can worsen contention/order (R-vmln-1 nm);
runtime guards can cost registers/occupancy. This probe measures the former;
vmld2_compile.py gates the latter. No predicted speedup is asserted.

Source accounting, bf2 N=8/16, per consumer lane per stage:
  Weight LDS = 4*16 + 2*4 = 72 B, unchanged; CTA weight stream = 36,864 B.
  For C cut groups (popcount of mask), classes all/dec/ldsm/mma:
  X LDS payload: 16*N -> (8-C)*2*N / 16*N / (8-C)*2*N / 16*N B per lane.
  LDSM.x4: N -> (8-C)*N/8 / N / (8-C)*N/8 / N instructions per thread.
  MMA: 2*N -> (8-C)*N/4 / 2*N / 2*N / (8-C)*N/4 instructions per thread.
  dec_half2: 8 -> 8-C / 8-C / 8 / 8 calls; each has 2 deq4 (8 integer ops each),
  scale PRMT + multiply + add, and 8 packed BF16 multiplies: EST 27 core ops,
  or 216 -> (8-C)*27 / (8-C)*27 / 216 / 216 core ops per stage.
  EST extra source ops: 2 flag extracts, 8 mask tests and 8 group guards;
  each kept group adds 1 dec test, N/8 ldsm tests, 1 hh-fold test, N/8 mma tests.
  dec stand-in: 8 assignments/group (two raw code words, each repeated four times).
  ldsm stand-in: 4 register assignments per N/8; mma sink: 8+4*N/8 XORs/group.
  Excludes scale broadcasts, address arithmetic/moves and existing SFX XORs; not SASS counts.
  These are source-ISA/operation counts, not measured SASS instruction totals;
  compiler folding, hoisting and register moves require the compile/SASS gate.
  Weight/scale LDS, all waits/arrives, producer bytes and launches are unchanged.

Protocol coverage (bf2, all main CTAs 0..G-1): no early exit is added.
  Consumer tids 0..511, q=0..3/wp=0..3/lane=0..31: both waits and joins stay;
  tids 0,32,..480 still provide all 16 weight-empty/x-empty arrivals per stage.
  seg d2 combine k=0..N/2-1, deferred flush and epilogue/resolve remain on
  every original consumer thread, including mask=255/class=0 (zero d2).
  Producer tids 512..575 and all subsequent cleanup/handshake paths unchanged.
  vm_fin: every token CTA 0..M-1 retains tids 0..255 for route combine,
  norm/output and y reset; CTA 0 retains tid-stride-256 xready[0..79],
  doneA[0..1279], cnt[0..1087] resets. VM_DP=0 (no gbar reset in bf2).
  No new predicate encloses any cleanup, reset, combine, barrier or arrival.
"""
import ast
import copy
import json
import math
import pathlib
import re
import statistics
import sys
from contextlib import nullcontext
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmld2_ab.py"
LIMIT = 1_000_000
GLOBALS = '''# VMLD2 lab only: half-group mask and class flags, fixed within a launch.
VMLD2_MASK = 0
VMLD2_CLASS = 0
_VMLD2 = {"calls": {}}
'''
ROUTE = '''    if rep == 32:
        fl |= ((VMLD2_MASK & 255) << 18) | (VMLD2_CLASS & (7 << 26))
        key = f"M{VMLD2_MASK:02x}C{VMLD2_CLASS >> 26}"
        _VMLD2["calls"][key] = _VMLD2["calls"].get(key, 0) + 1
'''


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, f"unique assignment: {name}"
    return ast.literal_eval(nodes[0].value)


def choose_branch(source, condition, take):
    """CPU source gate: constant-fold just the new uniform C++ conditions."""
    head = "if (" + condition + ") {"
    assert source.count(head) == 1, condition
    start = source.index(head)

    def body_at(brace):
        depth, end = 1, brace + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        return source[brace + 1:end - 1], end

    yes, end = body_at(start + len(head) - 1)
    no = ""
    tail = source[end:]
    if tail.lstrip().startswith("else {"):
        brace = end + len(tail) - len(tail.lstrip()) + len("else ")
        no, end = body_at(brace)
    return source[:start] + (yes if take else no) + source[end:]


def build(base):
    cu = literal(base, "_vml_CUDA_T")
    assert "VMLD2" not in base and "VMLN" not in base, "base already patched"
    reserved = (255 << 18) | (7 << 26)
    masks = [int(x) for x in re.findall(r"\bflags\s*&\s*(\d+)", cu)]
    assert masks and max(masks) == 32768 and all(not (x & reserved) for x in masks)
    for bit in range(18, 29):
        assert not re.search(rf"\b(?:{1 << bit}|0[xX]0*{1 << bit:x})[uUlL]*\b|1[uUlL]*\s*<<\s*{bit}\b", cu)
    for anchor in ("unsigned long long magG; int flags;", "a.flags = (int)V[5];",
                   "flags = _vml_FLAGS_DEFAULT if flags is None else flags",
                   '[m, ls, sc["G"], sc["magG"], mask, flags]', "flags=fl, cost=VML_COST.get(rep)"):
        assert base.count(anchor) == 1, f"flags transport: {anchor}"
    assert all(0 <= v < (1 << 16) for v in literal(base, "VML_FLAGS").values())
    for name in ("_vml_FLAGS_DEFAULT", "VML_DPFLAGS", "VML_SCFLAGS"):
        assert 0 <= literal(base, name) < (1 << 16), name
    assert literal(base, "VML_FLAGS")[32] == 193
    assert literal(base, "VML_VAR") == "bf2" and literal(base, "VML_ABVARS") == ()
    assert literal(base, "VML_BVAR") == {128: "w72", 256: "w72"}
    for variant in ("bf2", "w72"):
        defines = literal(base, "_VML_DEFS")[variant]
        assert "-DSFX=1" in defines
        assert not any(d.startswith(("-DVCP_LDA=", "-DVCP_M1=")) for d in defines)
    assert "#define VCP_LDA 0" in cu and "#define VCP_M1 0" in cu
    start = base.index("template <int N>\n__device__ __forceinline__ void stage_m(")
    end = base.index("template <bool PB, int N>\n__device__ __forceinline__ void epilogue(", start)
    old_m = base[start:end]
    m = sub(old_m, "#endif\nconst int ws = it % NS, xs = it % NX;",
            "#endif\nconst unsigned mask = ((unsigned)A.flags >> 18) & 255u;\n"
            "const unsigned cls = ((unsigned)A.flags >> 26) & 7u;\n"
            "const int ws = it % NS, xs = it % NX;", "mask/class")
    m = sub(m, "for (int hf = 0; hf < 2; ++hf) {\nuint32_t hh[8];",
            "for (int hf = 0; hf < 2; ++hf) {\nconst int gi = 2 * i + hf;\n"
            "const bool cut = (mask & (1u << gi)) != 0;\nif (!cut || cls != 0) {\nuint32_t hh[8];", "guard")
    dec = "dec_half2(hf ? c[i].y : c[i].x, hf ? c[i].w : c[i].z, sw2, par, hh, k16q, 64u, 4u);"
    m = sub(m, dec, '''if (cut && (cls & 1u)) {
hh[0] = hh[1] = hh[2] = hh[3] = hf ? c[i].y : c[i].x;
hh[4] = hh[5] = hh[6] = hh[7] = hf ? c[i].w : c[i].z;
} else {
''' + dec + '''
}
#if SFX
if (cut && (cls & 4u)) {
// Volatile operands retain every lane; fold hh once (not twice at N=16).
asm volatile("xor.b32 %0, %0, %1; xor.b32 %0, %0, %2; xor.b32 %0, %0, %3; xor.b32 %0, %0, %4;"
             "xor.b32 %0, %0, %5; xor.b32 %0, %0, %6; xor.b32 %0, %0, %7; xor.b32 %0, %0, %8;"
             : "+r"(xdep) : "r"(hh[0]), "r"(hh[1]), "r"(hh[2]), "r"(hh[3]),
               "r"(hh[4]), "r"(hh[5]), "r"(hh[6]), "r"(hh[7]));
}
#endif''', "dec stand-in and live hh sink")
    ldsm = "ldsm4(xb + (uint32_t)(((nt * 4 + i) * 8 + 4 * hf) * 128), b0, b1, b2, b3);"
    m = sub(m, ldsm, '''if (cut && (cls & 2u)) {
b0 = c[i].x; b1 = c[i].y; b2 = c[i].z; b3 = c[i].w;
} else {
''' + ldsm + "\n}", "ldsm stand-in")
    mma = "mma16816(&d2[0][4 * nt], hh[0], hh[4], hh[1], hh[5], b0, b1);\n" + \
          "mma16816(&d2[1][4 * nt], hh[2], hh[6], hh[3], hh[7], b2, b3);"
    m = sub(m, mma, '''if (!(cut && (cls & 4u))) {
''' + mma + '''
} else {
#if SFX
asm volatile("xor.b32 %0, %0, %1; xor.b32 %0, %0, %2; xor.b32 %0, %0, %3; xor.b32 %0, %0, %4;"
             : "+r"(xdep) : "r"(b0), "r"(b1), "r"(b2), "r"(b3));
#endif
}''', "mma cut and live ldsm sink")
    m = sub(m, "#endif\n}\n}\n__syncwarp();\n#if SFX",
            "#endif\n} // VMLD2: end uniform half-group guard\n}\n}\n__syncwarp();\n#if SFX", "guard end")
    # The guard encloses only the existing half-group; never the protocol.
    guarded = m.split("if (!cut || cls != 0) {\n", 1)[1].split("} // VMLD2:", 1)[0]
    assert not any(x in guarded for x in ("mbar_", "__syncwarp", "return", "break;", "continue;", "++it"))
    for event in ("mbar_wait", "mbar_arrive", "__syncwarp", "++it"):
        assert [x for x in old_m.splitlines() if event in x] == [x for x in m.splitlines() if event in x]
    off = m
    for condition, take in (("cut && (cls & 1u)", False), ("cut && (cls & 4u)", False),
                            ("cut && (cls & 2u)", False), ("!(cut && (cls & 4u))", True),
                            ("!cut || cls != 0", True)):
        off = choose_branch(off, condition, take)
    off = off.replace("#if SFX\n\n#endif", "")
    added_lines = ("const unsigned mask =", "const unsigned cls =", "const int gi =", "const bool cut =",
                   "// VMLD2: end uniform half-group guard")
    lines = lambda s: [line.strip() for line in s.splitlines() if line.strip()]
    off = "\n".join(line for line in off.splitlines() if not line.strip().startswith(added_lines))
    assert lines(off) == lines(old_m), "flag-zero folded stage differs from base"
    route_anchor = "    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)\n"
    edits = ((old_m, m, "stage_m-only guard"),
             ("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "module globals"),
             (route_anchor, route_anchor + ROUTE, "T=32 flags/counter"))
    s = base
    for old, new, what in edits:
        s = sub(s, old, new, what)
    ast.parse(s)
    assert literal(s, "VMLD2_MASK") == literal(s, "VMLD2_CLASS") == 0
    assert literal(s, "_VMLD2") == {"calls": {}}
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base, "edits are not exactly reversible"
    code = compile("if True:\n" + ROUTE, "<VMLD2 route injection>", "exec")
    for rep in (1, 8, 16, 32, 128, 256):
        for mask in range(256):
            for cls in range(8):
                ns = dict(rep=rep, fl=193, VMLD2_MASK=mask, VMLD2_CLASS=cls << 26, _VMLD2={"calls": {}})
                exec(code, ns)
                assert ns["fl"] == 193 | (((mask << 18) | (cls << 26)) if rep == 32 else 0)
                assert ns["fl"] >> 29 == 0  # Includes unchanged SFX bits 30/31.
                assert ns["_VMLD2"]["calls"] == ({f"M{mask:02x}C{cls}": 1} if rep == 32 else {})
    for fl in range(1 << 16):
        assert fl | ((0 & 255) << 18) | (0 & (7 << 26)) == fl
    for k in range(9):
        assert [g for g in range(8) if not (((1 << k) - 1) & (1 << g))] == list(range(k, 8))
    return s


def selected(source, names, ns=None):
    """Execute explicitly selected AST nodes plus their assignment dependencies."""
    tree = ast.parse(source)
    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        defs[n.id] = node
    chosen = {id(defs[name]): defs[name] for name in names}
    while True:
        deps = {n.id for node in chosen.values() for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        extra = {id(defs[n]): defs[n] for n in deps if n in defs and n not in (ns or {})
                 and isinstance(defs[n], ast.Assign)}
        if set(extra) <= set(chosen):
            break
        chosen.update(extra)
    env = {} if ns is None else ns
    nodes = sorted(chosen.values(), key=lambda n: n.lineno)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<CPU selected AST>", "exec"), env)
    return env


class Tensor:
    """Shape-only CPU stand-in: cannot allocate or launch CUDA."""
    def __init__(self, shape=(1,), dtype=None, device="cpu"):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype, self.device = dtype, device

    def contiguous(self):
        return self

    def to(self, dtype):
        return Tensor(self.shape, dtype)


def dry_run(base, patched):
    """Real base scratch constructors + real route/call; all arms and buckets."""
    specs = literal((ROOT / "tools/box/vmld2_run.py").read_text(encoding="utf-8"), "ARM_SPECS")
    torch = SimpleNamespace(int32="i32", int64="i64", float32="f32", bfloat16="bf16",
                            zeros=Tensor, empty=Tensor, zeros_like=lambda t: Tensor(t.shape, t.dtype))
    funcs = {"_vml_alloc", "_vml_mag", "_vml_route", "_vml_call", "_vml_cfg", "_vml_cost_ext",
             "_vml_pool_ext", "_vml_seg_ext", "_vml_pdel_ext", "_nb_vml_rep", "_k2a_bucket_of"}
    old = selected(base, funcs, {"torch": torch})
    new = selected(patched, funcs, {"torch": torch})
    # Derive the weight-state keys from the real base constructor, not readers.
    tree = ast.parse(base)
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_k2a33_prepare_st")
    st_assign = next(n for n in prepare.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "st" for t in n.targets))
    assert isinstance(st_assign.value, ast.Call) and st_assign.value.func.id == "dict"
    st = {kw.arg: Tensor() for kw in st_assign.value.keywords}
    # Copy the exact scratch-bank construction statements in _vml_prepare.
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vml_prepare")
    bank = [n for n in ast.walk(prepare) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Subscript) and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id == "_VML" and isinstance(n.targets[0].slice, ast.Constant)
            and n.targets[0].slice.value in ("sc", "scb", "scw")]
    assert len(bank) == 3
    records = []
    for ns in (old, new):
        ns.update(dev="cpu", sms=132)
        exec(compile(ast.Module(body=sorted(bank, key=lambda n: n.lineno), type_ignores=[]), "<base banks>", "exec"), ns)
        seen = []
        # CPU cfg ABI from vml_cfg; only fields consumed by the real Python call.
        assert 'return {VM_JN, VM_NTM, VM_NX, VM_XSN, VM_MKW, (int64_t)vm::SMEM, VM_DP, VM_SC, VM_BRED, VM_DEFER2, VM_PS};' in base
        mod = SimpleNamespace(vml_cfg=lambda: (64, 8, 3, 64, 32, 212912, 0, 0, 1, 0, 0),
                              vml_run=lambda tensors, values: seen.append((tensors, values)))
        ns["_VML"]["mod"] = mod
        ns["_VML"]["mods"] = {"bf2": mod, "w72": mod}
        ns["_VML"]["okv"] = {(r, "w72"): True for r in (128, 256)}
        for tokens in (1, 8, 16, 25, 32, 48, 128, 256):
            rep = ns["_nb_vml_rep"](tokens)
            for _, mask, cls in specs:
                ns["VMLD2_MASK"], ns["VMLD2_CLASS"] = mask, cls
                data = {"hidden_states": Tensor((tokens, ns["_vml_KDIM"])), "topk_ids": Tensor((tokens, 8)),
                        "topk_weights": Tensor((tokens, 8)), "config": {"local_start": 0}}
                out = ns["_vml_route"](st, data)
                tensors, values = seen[-1]
                assert out.shape == (tokens, ns["_vml_KDIM"]) and len(tensors) == 20
                assert tensors[3] is st["ws13"] and tensors[4] is st["ws2"]
                bank_name = "scb" if rep <= 32 else "scw"
                assert tensors[10] is ns["_VML"][bank_name]["y"]
                extra = ((mask << 18) | cls) if ns is new and rep == 32 else 0
                assert values[5] == ns["VML_FLAGS"].get(rep, ns["_vml_FLAGS_DEFAULT"]) | extra
                assert values[5] >> 30 == 0
        records.append([[t.shape for t in ts] + [vs[:5], vs[6:]] for ts, vs in seen])
    assert records[0] == records[1], "non-flag arguments changed"
    expected = {}
    for _, mask, cls in specs:
        key = f"M{mask:02x}C{cls >> 26}"
        expected[key] = expected.get(key, 0) + 3
    assert new["_VMLD2"]["calls"] == expected
    new["VMLD2_MASK"] = new["VMLD2_CLASS"] = 0
    return new, st


def cpu_gates(base, source):
    """Runner/energy/threshold tests use in-memory files; never load judge/CUDA."""
    env, st = dry_run(base, source)
    run = (ROOT / "tools/box/vmld2_run.py").read_text(encoding="utf-8")
    rat = (ROOT / "tools/box/rat_run.py").read_text(encoding="utf-8")
    names = literal(rat, "NAMES")
    r = selected(run, {"calls", "require_live", "ProbeFailure", "energy_schedule", "run_ab_nocheck",
                       "energy_report", "assess"},
                 dict(copy=copy, json=json, math=math, statistics=statistics, sys=sys, NAMES=names, RS=123))
    arms, shapes = r["ARMS"], r["SHAPES"]
    specs = [("mid1", 0, 0), ("g0", 1, 0), ("g7", 128, 0), ("g3", 8, 0), ("g67", 192, 0),
             ("c0dec", 1, 1 << 26), ("c0ldsm", 1, 1 << 27), ("c0mma", 1, 1 << 28),
             ("k2ref", 3, 0), ("g0b", 1, 0), ("mid2", 0, 0)]
    assert arms == [(n, {"VMLD2_MASK": m, "VMLD2_CLASS": c}) for n, m, c in specs]
    assert shapes == [4, 5] and [names[i] for i in shapes] == ["d32s", "d32f"]
    module = SimpleNamespace(**env)
    r["calls"](module)
    r["require_live"](module._VMLD2["calls"], "CPU routes")
    for field, bads in (("VMLD2_MASK", (-1, 256, True, "1")),
                        ("VMLD2_CLASS", (-1, 1, 8 << 26, True, "0"))):
        for bad in bads:
            setattr(module, field, bad)
            try:
                r["calls"](module)
            except r["ProbeFailure"]:
                pass
            else:
                raise AssertionError("bad mask/class accepted")
        setattr(module, field, 0)
    for key in [k for k in module._VMLD2["calls"] if k != "M00C0"]:
        counts = dict(module._VMLD2["calls"], **{key: 0})
        try:
            r["require_live"](counts, "negative test")
        except r["ProbeFailure"]:
            pass
        else:
            raise AssertionError("dead arm accepted")

    class MemoryPath:
        def __init__(self):
            self.text, self.child = "", None

        def write_text(self, text):
            self.text = text

        def read_text(self):
            return self.text

        def with_suffix(self, suffix):
            self.child = MemoryPath()
            return self.child

    # Exercise the proven eager arm selection/restoration with real route code.
    # Mock check_output validates control-vs-null handling, not GPU numerics.
    def kernel_fn(data, state):
        env["VMLD2_MASK"], env["VMLD2_CLASS"] = module.VMLD2_MASK, module.VMLD2_CLASS
        return env["_vml_route"](state, data)
    module.kernel_fn = kernel_fn
    check_calls = []
    def check_output(*args):
        check_calls.append((module.VMLD2_MASK, module.VMLD2_CLASS))
        return module.VMLD2_MASK == 0, "CPU mock output gate"
    judge = SimpleNamespace(SEED_STEP=100, clone_data=copy.deepcopy, check_output=check_output,
                            make_input=lambda *a: {"hidden_states": Tensor((32, env["_vml_KDIM"])),
                                                   "topk_ids": Tensor((32, 8)), "topk_weights": Tensor((32, 8)),
                                                   "config": {"local_start": 0}})
    eager = selected(rat, {"eager_trace"}, dict(copy=copy, json=json, J=judge, NAMES=names,
                     log=lambda _: None, install_trace=lambda *a: {},
                     torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))))["eager_trace"]
    prob = SimpleNamespace(benches=[{"seed": i} for i in range(6)])
    module._VMLD2["calls"].clear()
    tr = eager(prob, None, module, st, None, arms, shapes, MemoryPath())
    assert check_calls == [(s["VMLD2_MASK"], s["VMLD2_CLASS"]) for _, s in arms] * 2
    assert module.VMLD2_MASK == module.VMLD2_CLASS == 0
    assert all(tr[name][i]["ok"] == (settings["VMLD2_MASK"] == 0) for name, settings in arms for i in shapes)
    r["require_live"](r["calls"](module), "CPU eager")

    out, samples = MemoryPath(), []
    original = lambda v: v
    fake = SimpleNamespace(keyfix=original, _rat_keyfix_orig=original)
    last = [0]
    def measure(nv, step, secs, torch):
        step()  # Same graph for null and control arms; no output validation.
        return {"calls": 16, "us": 150.0, "mJ": 60.0 - last[0], "W": 400.0,
                "mhz": 1500 + last[0] * 10, "mhz_p10": 1490, "cap": 0.5, "n_smp": 37,
                "future_field": {"preserved": True}}
    fake.measure = measure
    rev = (ROOT / "tools/box/rev_ab3.py").read_text(encoding="utf-8")
    main = next(n for n in ast.parse(rev).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    en_if = next(n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test).startswith("esecs > 0"))
    en_code = compile(ast.Module(body=[en_if], type_ignores=[]), "<real rev_ab3 energy loop>", "exec")
    def fake_main():
        assert all(x in sys.argv for x in ("--esecs=0.4", "--erounds=4", "--check=0", "--stale=0", "--router-mode=fixed"))
        variants = json.loads(out.child.read_text())
        variants = [{"name": v["name"], "set": fake.keyfix(v["set"])} for v in variants]
        assert [(v["name"], v["set"]) for v in variants] == arms
        fl = fake.measure(None, lambda: None, 0.4, None)
        raw = {"shapes": []}
        for idx in shapes:
            graphs = []
            for v in variants:
                settings = v["set"]
                def replay(settings=settings):
                    last[0] = settings["VMLD2_MASK"].bit_count() + (settings["VMLD2_CLASS"] >> 26)
                    env.update(settings)
                    env["_vml_route"](st, {"hidden_states": Tensor((32, env["_vml_KDIM"])), "topk_ids": Tensor((32, 8)),
                                           "topk_weights": Tensor((32, 8)), "config": {"local_start": 0}})
                graphs.append((SimpleNamespace(replay=replay),))
            scope = dict(esecs=0.4, eshapes="all", capped=True, erounds=4, variants=variants, graphs=graphs,
                         torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext())),
                         timer=SimpleNamespace(stream=None, flush=lambda: None), measure=fake.measure,
                         nv=None, fl=fl, en={}, statistics=statistics)
            exec(en_code, scope)
            raw["shapes"].append({"index": idx, "energy": scope["en"], "failures": {}, "variants": [
                {"name": n, "times_us": [150.0 - (settings["VMLD2_MASK"].bit_count() + (settings["VMLD2_CLASS"] >> 26)) * 0.8] * 24} for n, settings in arms]})
        out.write_text(json.dumps(raw))
    fake.main = fake_main
    argv_before = sys.argv
    r["run_ab_nocheck"](fake, "unused", arms, shapes, 3, 8, out, samples)
    assert sys.argv is argv_before and fake.measure is measure and fake.keyfix is original
    assert len(samples) == 91
    raw = json.loads(out.read_text())
    energy = r["energy_report"](raw, samples)
    for shape in energy["energy"].values():
        for arm, e in shape.items():
            settings = dict(arms)[arm]
            k = settings["VMLD2_MASK"].bit_count() + (settings["VMLD2_CLASS"] >> 26)
            assert e["mean_round_median_sm_mhz"] == 1500 + k * 10
            assert all(v["future_field"] == {"preserved": True} for v in e["rounds"])
    assert all(c["over_2pct"] for c in energy["clock_comparison"].values())
    missing = copy.deepcopy(samples)
    for sample in missing:
        sample["measurement"]["mhz"] = None
    assert all(c["over_2pct"] is None for c in r["energy_report"](raw, missing)["clock_comparison"].values())
    pair = selected((ROOT / "tools/box/sk2_run.py").read_text(encoding="utf-8"), {"paired"}, dict(json=json, NAMES=names))["paired"]
    p = pair(out, shapes)
    got = r["assess"](raw, p)
    assert got["verdict"] == "REPORT ONLY" and got["reading"]["pooled_H"][0].startswith("NO EFFECT")
    assert [row["arm"] for row in got["scan_table"]] == [n for n, _ in arms]
    assert all(row["abs_S_diff_us"] == 0 and row["ci_us"] == [0, 0] for row in got["repeat"].values())
    assert all(lo <= got["H_us"][a] <= hi for a, (lo, hi) in got["H_ci_us"].items())
    # Exercise unrounded VOID CIs, including positive/negative controls, a CI
    # straddling zero, touching zero, and the strict 0.6 boundary.
    for ctl, spread, void in ((0.62, 2, False), (0.7, 0, True), (-0.7, 0, True),
                              (0.7, 0.7, False), (0.6, 0, False)):
        q = copy.deepcopy(raw)
        for sh in q["shapes"]:
            vs = {v["name"]: v for v in sh["variants"]}
            cs = [ctl - spread] * 12 + [ctl + spread] * 12
            vs["mid1"]["times_us"] = [150 + c / 2 for c in cs]
            vs["mid2"]["times_us"] = [150 - c / 2 for c in cs]
        result = r["assess"](q, p)
        assert (result["verdict"] == "VOID") == void
        assert len(result["scan_table"]) == 11 and len(result["repeat"]) == 2
    cases = ((2.0, 1.0, (1.4, 0, 0), ["FIRST-GROUP / start chain", "CLASS dec"]),
             (2.0, 2.0, (1.4, 1.4, 1.4), ["ANY-CUT", "CLASS dec / ldsm / mma"]),
             (0.99, 2.0, (0, 0, 0), ["NO EFFECT (R-vmld-1 K1 not reproduced)"]),
             (0.5, 0, (0.35, 0, 0), ["NO EFFECT (R-vmld-1 K1 not reproduced)", "CLASS dec"]),
             (1.0, 1.5, (0, 0, 0), ["INCONCLUSIVE"]),
             (2.0, 1.5, (0, 0, 0), ["INCONCLUSIVE"]))
    for g0, g7, classes, expected in cases:
        q = copy.deepcopy(raw)
        values = dict(g0=g0, g7=g7, g0b=g0-0.25, **dict(zip(("c0dec", "c0ldsm", "c0mma"), classes)))
        for sh in q["shapes"]:
            for v in sh["variants"]:
                if v["name"] in values:
                    v["times_us"] = [150 - values[v["name"]]] * 24
        result = r["assess"](q, p)
        assert all(v == expected for v in result["reading"].values()), result["reading"]
        assert all(math.isclose(v["abs_S_diff_us"], 0.25) for v in result["repeat"].values())
        assert math.isclose(result["H_us"]["g0"], g0)
    q = copy.deepcopy(raw)
    for sh, gain in zip(q["shapes"], (2.0, 4.0)):
        next(v for v in sh["variants"] if v["name"] == "g0")["times_us"] = [150 - gain] * 24
    result = r["assess"](q, p)
    assert result["H_us"]["g0"] == 3.0 and result["H_ci_us"]["g0"] == [3.0, 3.0]
    # Failure restoration and stale-output rejection: no previous timing can pass.
    def broken_main():
        raise SystemExit(3)
    for bad_main in (broken_main, lambda: None):
        fake.main = bad_main
        try:
            r["run_ab_nocheck"](fake, "unused", arms, shapes, 3, 8, out, [])
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed/incomplete timing accepted")
        assert sys.argv is argv_before and fake.measure is measure and fake.keyfix is original
        assert json.loads(out.read_text()) == {"shapes": []}
    compile_gates()
    print("[vmld2] CPU dry run PASS: real base banks/route/call, 11 arms; real eager_trace and rev_ab3 energy loop")
    print("[vmld2] CPU gates PASS: exact scope/flags/protocol, live counters, position/class/repeat/VOID boundaries, all energy fields, resource limits")


def compile_gates():
    src = (ROOT / "tools/local/vmld2_compile.py").read_text(encoding="utf-8")
    ns = selected(src, {"compare", "parse_ptxas"}, dict(re=re))
    text = "\n".join(f"ptxas info : Compiling entry function '{name}' for 'sm_90a'\n"
                     f"ptxas info : Function properties for {name}\n"
                     " 16 bytes stack frame, 20 bytes spill stores, 80 bytes spill loads\n"
                     "ptxas info : Used 96 registers, 0 bytes smem" for name in ("vm_main", "vm_fin"))
    records = ns["parse_ptxas"](text)
    records["vm_fin"].update(regs=126, stack=0, stores=0, loads=0, smem=388)
    cfg = (64, 8, 3, 64, 32, 212912, 0, 0, 1, 0, 0)
    base = (cfg, records)
    assert not ns["compare"]("bf2", base, copy.deepcopy(base))
    for key, value in (("regs", 97), ("stores", 21), ("loads", 81), ("stack", 17), ("smem", 1)):
        changed = copy.deepcopy(base)
        changed[1]["vm_main"][key] = value
        assert ns["compare"]("bf2", base, changed), key
    assert ns["compare"]("bf2", base, (cfg[:5] + (212913,) + cfg[6:], copy.deepcopy(records)))
    for key in ("regs", "stores", "loads", "stack", "smem"):
        changed = copy.deepcopy(base)
        changed[1]["vm_fin"][key] += 1
        assert ns["compare"]("bf2", base, changed), f"vm_fin drift accepted: {key}"
    for broken in ("", text.replace("80 bytes spill loads", "missing loads")):
        try:
            ns["parse_ptxas"](broken)
        except ValueError:
            pass
        else:
            raise AssertionError("missing compiler metrics accepted")
    for path in (ROOT / "tools/local/vmld2_compile.py", ROOT / "tools/box/vmld2_run.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    shell = (ROOT / "tools/box/vmld2_batch.sh").read_bytes()
    assert b"\r" not in shell and b"timeout -k 10 245" in shell and b"self_stop.sh 340" in shell
    assert b"timeout 12 bash tools/box/sass_dump.sh box_out/vmld2 || true\nfi\nexit" in shell
    assert b"if [ -f tools/box/sass_dump.sh ]; then" in shell
    assert b"R-vmld-2 pre-registered, one pod, cap 0.50 USD" in shell
    runner = (ROOT / "tools/box/vmld2_run.py").read_text(encoding="utf-8")
    assert "threading.Timer(110.0, expired)" in runner
    assert 'if (seeds, pairs) != (3, 8):' in runner


def main():
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmld2_patches.py (lab only)")
    raw = BASE.read_bytes()
    base = raw.decode("utf-8")
    crlf = b"\r\n" in raw
    assert not crlf or base.count("\n") == base.count("\r\n"), "mixed newlines"
    base = base.replace("\r\n", "\n")
    source = build(base)
    cpu_gates(base, source)
    out = source.replace("\n", "\r\n").encode("utf-8") if crlf else source.encode("utf-8")
    assert len(out) <= LIMIT, f"lab file over cap: {len(out)} > {LIMIT}"
    OUT.write_bytes(out)
    print(f"[vmld2] {OUT.relative_to(ROOT)}: {len(out):,} bytes; headroom {LIMIT - len(out):,}")


if __name__ == "__main__":
    main()

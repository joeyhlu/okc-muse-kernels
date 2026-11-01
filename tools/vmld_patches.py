"""R-vmld-1: stage_m-only half-group dose; lab only, copied from vmln's anchors.

Run: python -B -I tools/vmld_patches.py (generator + CPU gates + dry run).
No layer import, CUDA call, files beyond the fixed output, or GPU job.

Kill risks: stage_m-only removal can worsen contention/order (R-vmln-1 nm);
runtime guards can cost registers/occupancy. This probe measures the former;
vmld_compile.py gates the latter. No predicted speedup is asserted.

Source accounting, bf2 N=8/16, per consumer lane per stage:
  Weight LDS = 4*16 + 2*4 = 72 B, unchanged; CTA weight stream = 36,864 B.
  X LDS payload = 8*(N/8)*16 = 16*N B -> (8-K)*2*N B per lane.
  Inline-ISA LDSM.x4: N -> (8-K)*N/8; MMA: 2*N -> (8-K)*N/4 instructions.
  dec_half2: 8 -> 8-K calls; each has 2 deq4 (8 elementary integer ops each),
  scale PRMT + multiply + add, and 8 packed BF16 multiplies: EST 27 core ops,
  or 216 -> (8-K)*27 per stage, excluding scale broadcasts/address/move/XOR.
  Added: one flag extraction and eight uniform tests (EST 1 BFE + 8 SETP/BRA).
  These are source-ISA/operation counts, not measured SASS instruction totals;
  compiler folding, hoisting and register moves require the compile/SASS gate.
  Weight/scale LDS, all waits/arrives, producer bytes and launches are unchanged.

Protocol coverage (bf2, all main CTAs 0..G-1): no early exit is added.
  Consumer tids 0..511, q=0..3/wp=0..3/lane=0..31: both waits and joins stay;
  tids 0,32,..480 still provide all 16 weight-empty/x-empty arrivals per stage.
  seg d2 combine k=0..N/2-1, deferred flush and epilogue/resolve remain on
  every original consumer thread, including K=8 (zero-initialized d2).
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
OUT = ROOT / "kernels/lab/k3_vmld_ab.py"
LIMIT = 1_000_000
GLOBALS = '''# VMLD lab only: stage_m half-groups skipped (0..8), fixed within a launch.
VMLD_K = 0
_VMLD = {"calls": {}}
'''
ROUTE = '''    if rep == 32:
        fl |= (VMLD_K & 15) << 26
        key = f"K{VMLD_K}"
        _VMLD["calls"][key] = _VMLD["calls"].get(key, 0) + 1
'''


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, f"unique assignment: {name}"
    return ast.literal_eval(nodes[0].value)


def build(base):
    cu = literal(base, "_vml_CUDA_T")
    assert "VMLD" not in base and "VMLN" not in base, "base already patched"
    reserved = 15 << 26
    masks = [int(x) for x in re.findall(r"\bflags\s*&\s*(\d+)", cu)]
    assert masks and max(masks) == 32768 and all(not (x & reserved) for x in masks)
    for bit in range(26, 30):
        assert not re.search(rf"\b(?:{1 << bit}|0[xX]0*{1 << bit:x})[uUlL]*\b|1[uUlL]*\s*<<\s*{bit}\b", cu)
    for anchor in ("unsigned long long magG; int flags;", "a.flags = (int)V[5];",
                   "flags = _vml_FLAGS_DEFAULT if flags is None else flags",
                   '[m, ls, sc["G"], sc["magG"], mask, flags]', "flags=fl, cost=VML_COST.get(rep)"):
        assert base.count(anchor) == 1, f"flags transport: {anchor}"
    assert all(0 <= v < (1 << 16) for v in literal(base, "VML_FLAGS").values())
    for name in ("_vml_FLAGS_DEFAULT", "VML_DPFLAGS", "VML_SCFLAGS"):
        assert 0 <= literal(base, name) < (1 << 16), name
    start = base.index("template <int N>\n__device__ __forceinline__ void stage_m(")
    end = base.index("template <bool PB, int N>\n__device__ __forceinline__ void epilogue(", start)
    old_m = base[start:end]
    m = sub(old_m, "#endif\nconst int ws = it % NS, xs = it % NX;",
            "#endif\nconst int K = (A.flags >> 26) & 15;\nconst int ws = it % NS, xs = it % NX;", "K")
    m = sub(m, "for (int hf = 0; hf < 2; ++hf) {\nuint32_t hh[8];",
            "for (int hf = 0; hf < 2; ++hf) {\nconst int gi = 2 * i + hf;\nif (gi >= K) {\nuint32_t hh[8];", "guard")
    m = sub(m, "#endif\n}\n}\n__syncwarp();\n#if SFX",
            "#endif\n} // VMLD: end uniform half-group guard\n}\n}\n__syncwarp();\n#if SFX", "guard end")
    # The guard encloses only the existing half-group; never the protocol.
    guarded = m.split("if (gi >= K) {\n", 1)[1].split("} // VMLD:", 1)[0]
    assert not any(x in guarded for x in ("mbar_", "__syncwarp", "return", "break;", "continue;", "++it"))
    for event in ("mbar_wait", "mbar_arrive", "__syncwarp", "++it"):
        assert [x for x in old_m.splitlines() if event in x] == [x for x in m.splitlines() if event in x]
    route_anchor = "    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)\n"
    edits = ((old_m, m, "stage_m-only guard"),
             ("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "module globals"),
             (route_anchor, route_anchor + ROUTE, "T=32 flags/counter"))
    s = base
    for old, new, what in edits:
        s = sub(s, old, new, what)
    ast.parse(s)
    assert literal(s, "VMLD_K") == 0 and literal(s, "_VMLD") == {"calls": {}}
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base, "edits are not exactly reversible"
    code = compile("if True:\n" + ROUTE, "<VMLD route injection>", "exec")
    for rep in (1, 8, 16, 32, 128, 256):
        for k in range(9):
            ns = dict(rep=rep, fl=193, VMLD_K=k, _VMLD={"calls": {}})
            exec(code, ns)
            assert ns["fl"] == 193 | ((k << 26) if rep == 32 else 0)
            assert ns["fl"] >> 30 == 0 and ns["fl"] & (255 << 18) == 0
            assert ns["_VMLD"]["calls"] == ({f"K{k}": 1} if rep == 32 else {})
    for fl in range(1 << 16):
        assert fl | ((0 & 15) << 26) == fl
    for k in range(9):
        assert [2 * i + hf for i in range(4) for hf in range(2) if 2 * i + hf >= k] == list(range(k, 8))
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
    """Real base scratch constructors + real route/call; all doses and buckets."""
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
            for k in range(9):
                ns["VMLD_K"] = k
                data = {"hidden_states": Tensor((tokens, ns["_vml_KDIM"])), "topk_ids": Tensor((tokens, 8)),
                        "topk_weights": Tensor((tokens, 8)), "config": {"local_start": 0}}
                out = ns["_vml_route"](st, data)
                tensors, values = seen[-1]
                assert out.shape == (tokens, ns["_vml_KDIM"]) and len(tensors) == 20
                assert tensors[3] is st["ws13"] and tensors[4] is st["ws2"]
                bank_name = "scb" if rep <= 32 else "scw"
                assert tensors[10] is ns["_VML"][bank_name]["y"]
                extra = (k << 26) if ns is new and rep == 32 else 0
                assert values[5] == ns["VML_FLAGS"].get(rep, ns["_vml_FLAGS_DEFAULT"]) | extra
                assert values[5] >> 30 == 0
        records.append([[t.shape for t in ts] + [vs[:5], vs[6:]] for ts, vs in seen])
    assert records[0] == records[1], "non-flag arguments changed"
    assert new["_VMLD"]["calls"] == {f"K{k}": 3 for k in range(9)}
    return new, st


def cpu_gates(base, source):
    """Runner/energy/threshold tests use in-memory files; never load judge/CUDA."""
    env, st = dry_run(base, source)
    run = (ROOT / "tools/box/vmld_run.py").read_text(encoding="utf-8")
    rat = (ROOT / "tools/box/rat_run.py").read_text(encoding="utf-8")
    names = literal(rat, "NAMES")
    r = selected(run, {"calls", "require_live", "ProbeFailure", "energy_schedule", "run_ab_nocheck",
                       "energy_report", "assess"},
                 dict(copy=copy, json=json, math=math, statistics=statistics, sys=sys, NAMES=names, RS=123))
    arms, shapes = r["ARMS"], r["SHAPES"]
    assert arms == [(n, {"VMLD_K": k}) for n, k in zip(("mid1", "k1", "k2", "k4", "k8", "mid2"), (0, 1, 2, 4, 8, 0))]
    assert shapes == [4, 5] and [names[i] for i in shapes] == ["d32s", "d32f"]
    module = SimpleNamespace(**env)
    r["calls"](module)
    r["require_live"](module._VMLD["calls"], "CPU routes")
    for bad in (-1, 9, True, "1"):
        module.VMLD_K = bad
        try:
            r["calls"](module)
        except r["ProbeFailure"]:
            pass
        else:
            raise AssertionError("bad K accepted")
    module.VMLD_K = 0
    for key in ("K1", "K2", "K4", "K8"):
        counts = dict(module._VMLD["calls"], **{key: 0})
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
        env["VMLD_K"] = module.VMLD_K
        return env["_vml_route"](state, data)
    module.kernel_fn = kernel_fn
    check_calls = []
    def check_output(*args):
        check_calls.append(module.VMLD_K)
        return module.VMLD_K == 0, "CPU mock output gate"
    judge = SimpleNamespace(SEED_STEP=100, clone_data=copy.deepcopy, check_output=check_output,
                            make_input=lambda *a: {"hidden_states": Tensor((32, env["_vml_KDIM"])),
                                                   "topk_ids": Tensor((32, 8)), "topk_weights": Tensor((32, 8)),
                                                   "config": {"local_start": 0}})
    eager = selected(rat, {"eager_trace"}, dict(copy=copy, json=json, J=judge, NAMES=names,
                     log=lambda _: None, install_trace=lambda *a: {},
                     torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))))["eager_trace"]
    prob = SimpleNamespace(benches=[{"seed": i} for i in range(6)])
    module._VMLD["calls"].clear()
    tr = eager(prob, None, module, st, None, arms, shapes, MemoryPath())
    assert check_calls == [s["VMLD_K"] for _, s in arms] * 2
    assert module.VMLD_K == 0
    assert all(tr[name][i]["ok"] == (settings["VMLD_K"] == 0) for name, settings in arms for i in shapes)
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
                k = v["set"]["VMLD_K"]
                def replay(k=k):
                    last[0] = k
                    env["VMLD_K"] = k
                    env["_vml_route"](st, {"hidden_states": Tensor((32, env["_vml_KDIM"])), "topk_ids": Tensor((32, 8)),
                                           "topk_weights": Tensor((32, 8)), "config": {"local_start": 0}})
                graphs.append((SimpleNamespace(replay=replay),))
            scope = dict(esecs=0.4, eshapes="all", capped=True, erounds=4, variants=variants, graphs=graphs,
                         torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext())),
                         timer=SimpleNamespace(stream=None, flush=lambda: None), measure=fake.measure,
                         nv=None, fl=fl, en={}, statistics=statistics)
            exec(en_code, scope)
            raw["shapes"].append({"index": idx, "energy": scope["en"], "failures": {}, "variants": [
                {"name": n, "times_us": [150.0 - settings["VMLD_K"] * 0.8] * 24} for n, settings in arms]})
        out.write_text(json.dumps(raw))
    fake.main = fake_main
    argv_before = sys.argv
    r["run_ab_nocheck"](fake, "unused", arms, shapes, 3, 8, out, samples)
    assert sys.argv is argv_before and fake.measure is measure and fake.keyfix is original
    assert len(samples) == 51
    raw = json.loads(out.read_text())
    energy = r["energy_report"](raw, samples)
    for shape in energy["energy"].values():
        for arm, e in shape.items():
            k = dict(arms)[arm]["VMLD_K"]
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
    assert got["family_verdict"] == "BUILD P1/P2"
    assert all(math.isclose(row["r_K"], 1) for row in got["dose_response"])
    # Strict magnitude AND CI excludes 0: test positive/negative, touching zero,
    # exact boundary, and the old vmln 0.62 [-0.16, 1.18] result.
    for ctl, ci, void in ((0.62, [-0.16, 1.18], False), (0.7, [0.1, 1], True),
                          (-0.7, [-1, -0.1], True), (0.7, [0, 1], False), (0.6, [0.1, 1], False)):
        q = copy.deepcopy(raw)
        for sh in q["shapes"]:
            vs = {v["name"]: v for v in sh["variants"]}
            vs["mid1"]["times_us"] = [150 + ctl / 2] * 24
            vs["mid2"]["times_us"] = [150 - ctl / 2] * 24
        pc = copy.deepcopy(p)
        for block in pc.values():
            block["ctl_ci"] = ci
        assert (r["assess"](q, pc)["family_verdict"] == "VOID") == void
    for saving, expected in ((0.8, "BUILD P1/P2"), (0.3, "KILL stage_m family"),
                              (-1, "KILL stage_m family"), (0.5, "INCONCLUSIVE")):
        q = copy.deepcopy(raw)
        for sh in q["shapes"]:
            next(v for v in sh["variants"] if v["name"] == "k1")["times_us"] = [150 - saving] * 24
        assert r["assess"](q, p)["family_verdict"].startswith(expected)
    q = copy.deepcopy(raw)
    for sh in q["shapes"]:
        next(v for v in sh["variants"] if v["name"] == "k8")["times_us"] = [150.0] * 24
    assert all(row["r_K"] is None for row in r["assess"](q, p)["dose_response"])
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
    print("[vmld] CPU dry run PASS: real base banks/route/call, K=0..8; real eager_trace and rev_ab3 energy loop")
    print("[vmld] CPU gates PASS: exact scope/flags/protocol, live counters, dose/VOID boundaries, all energy fields, resource limits")


def compile_gates():
    src = (ROOT / "tools/local/vmld_compile.py").read_text(encoding="utf-8")
    ns = selected(src, {"compare", "parse_ptxas"}, dict(re=re))
    text = "\n".join(f"ptxas info : Compiling entry function '{name}' for 'sm_90a'\n"
                     f"ptxas info : Function properties for {name}\n"
                     " 16 bytes stack frame, 20 bytes spill stores, 80 bytes spill loads\n"
                     "ptxas info : Used 96 registers, 0 bytes smem" for name in ("vm_main", "vm_fin"))
    records = ns["parse_ptxas"](text)
    cfg = (64, 8, 3, 64, 32, 212912, 0, 0, 1, 0, 0)
    base = (cfg, records)
    assert not ns["compare"]("bf2", base, copy.deepcopy(base))
    for key, value in (("regs", 97), ("stores", 21), ("loads", 81), ("stack", 17), ("smem", 1)):
        changed = copy.deepcopy(base)
        changed[1]["vm_main"][key] = value
        assert ns["compare"]("bf2", base, changed), key
    assert ns["compare"]("bf2", base, (cfg[:5] + (212913,) + cfg[6:], copy.deepcopy(records)))
    for path in (ROOT / "tools/local/vmld_compile.py", ROOT / "tools/box/vmld_run.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    shell = (ROOT / "tools/box/vmld_batch.sh").read_bytes()
    assert b"\r" not in shell and b"timeout -k 10 215" in shell and b"self_stop.sh 300" in shell
    assert b"timeout 12 bash tools/box/sass_dump.sh box_out/vmld || true\nexit" in shell


def main():
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmld_patches.py (lab only)")
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
    print(f"[vmld] {OUT.relative_to(ROOT)}: {len(out):,} bytes; headroom {LIMIT - len(out):,}")


if __name__ == "__main__":
    main()

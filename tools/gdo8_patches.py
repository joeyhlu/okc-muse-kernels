#!/bin/bash
""":"
set -euo pipefail
export PATH="$PATH:/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
exec python3 -B -I "$0" --linux-gates "$@"
":"""
"""GDO8: exact-anchor, default-off o/bucket-8 XP6 bank; no new CUDA source.

python -B -I tools/gdo8_patches.py              generate + CPU gates/dry run
python -B -I tools/gdo8_patches.py --cpu-only   verify existing output, no writes
bash tools/gdo8_patches.py                    Linux compile/SASS + CPU gates

The bash/Python polyglot keeps the Linux runner in the sole authorized tools
file. Linux mode requires tools/local/cudaenv.sh, never launches a kernel,
retains compiler scratch in /tmp, and writes gdo8_patches.json beside this file.
GPU boundaries in the CPU dry run are explicit doubles; the actual extracted
base/lab _gd_prepare functions construct the state and packed-route bank.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_diet_rq128.py"
OUT = ROOT / "kernels/lab/k3_gdo8_ab.py"

GLOBAL_A = "GD_ENABLE=True\n"
GLOBAL_N = "GDO8=False # lab: o/bucket 8 uses the already-built o/1 XP6 instance\n" + GLOBAL_A
PREP_A = ' state["gd"]={"mod":mod,"route":routes}if routes else None\n'
PREP_N = PREP_A + ' _gdo8_prepare(state)\n'
LIVE_A = ''' r=st["route"].get((proj,b))
 if r is None or tokens>r["T"]or(proj=="unpooler" and r["slice"]!=(tokens<=SLICED_UNPOOL_MAX_TOKENS)):
'''
LIVE_N = ''' if GDO8 and proj=="o" and b==8:
  r=st["gdo8"]
  if r is None:
   raise RuntimeError("[gdo8] XP6 arm did not pass prepare; refusing an inert A/B")
 else:
  r=st["route"].get((proj,b))
 if r is None or tokens>r["T"]or(proj=="unpooler" and r["slice"]!=(tokens<=SLICED_UNPOOL_MAX_TOKENS)):
'''
HELPER_A = 'def _gd_live(state,proj,tokens):\n'
HELPER = '''def _gdo8_prepare(state):
 t1=time.time()
 st=state.get("gd")
 ok=False
 shared=False
 msg="GD unavailable"
 if st is not None:
  st["gdo8"]=None
  r8=st["route"].get(("o",8))
  r6=st["route"].get(("o",1))
  msg="o/8 or o/1 route unavailable"
  if r8 is not None and r6 is not None:
   # _gd_prepare's pk_key omits XP; these are the SAME packed allocations.
   shared=r6["w"]is r8["w"]and r6["esc"]is r8["esc"]
   if not(shared and all(r6[k]==r8[k]for k in("tb","nbx","N","K","T","slice"))):
    raise RuntimeError("[gdo8] base packing/geometry changed")
   r=dict(r6,desc=r6["desc"]+" GDO8")
   try:
    ok,msg=_gd_selftest(r,state["dense"][GD_WEIGHT["o"]][:r["N"]].contiguous(),8)
   except Exception as ex:
    ok,msg=False,f"EXC {ex!r}"
   if ok:
    st["gdo8"]=r
 print(f"[gdo8] o/8 XP6 ready={ok}; shared_pack={shared}; extra_device_bytes=0; extra_prepare_s={time.time()-t1:.6f}; {msg}",flush=True)
'''
EDITS = (
    (GLOBAL_A, GLOBAL_N, "global"),
    (PREP_A, PREP_N, "prepare bank call"),
    (HELPER_A, HELPER + HELPER_A, "prepare bank helper"),
    (LIVE_A, LIVE_N, "live arm selection"),
)


def build(source):
    for old, new, label in EDITS:
        assert source.count(old) == 1, f"{label}: expected one exact anchor"
        source = source.replace(old, new, 1)
    compile(source, str(OUT), "exec")  # in memory: no __pycache__
    return source


def selected(source, names, env=None):
    env = {} if env is None else env
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            ids = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            ids = {node.name} if isinstance(node, ast.FunctionDef) else set()
        if ids & set(names):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<actual base/lab nodes>", "exec"), env)
            found |= ids
    assert set(names) <= found, f"missing real definitions: {set(names)-found}"
    return env


NAMES = (
    "HIDDEN", "D_EXPERT", "N_EXPERTS", "UNPOOL_SLICE", "SLICED_UNPOOL_MAX_TOKENS",
    "NB64_LO", "NB64_HI", "NB64_USE", "NB64_BUILD", "NB64_GD_B", "_nb64",
    "GD_ENABLE", "GD_BUILD", "GD_REQUIRE", "GD_PREPARE_BY_S", "GD_SMS", "GD_ESC_CAP",
    "GD_ROUTE", "GD_DIMS", "GD_WEIGHT", "GD_SELF_TEST_T", "GD_SELF_TEST_PDL",
    "GD_PDL", "GD_TRIG", "GD_ONE_WAVE", "_GD", "GD_CPP_SRC", "GD_CUDA_SRC",
    "_gd_bucket", "_gd_nk", "_gd_key", "_gd_keys", "_gd_prepare", "_gd_require", "_gd_live",
    "GD_CAT_ROUTE", "GD_CAT_UNITS", "_gd_cat_key", "_gd_cat_keys",
)


class TensorDouble:
    """Only the unchanged prepare's CUDA tensor boundary is mocked."""
    dtype = "bf16"
    device = "cpu-double"

    def __init__(self, shape):
        self.shape = shape

    def dim(self):
        return len(self.shape)

    def __getitem__(self, index):
        assert isinstance(index, slice) and index.start is None and index.step is None
        return TensorDouble((min(index.stop, self.shape[0]), self.shape[1]))

    def contiguous(self):
        return self

    def view(self, dtype):
        return self

    def numel(self):
        return self.shape[0] * self.shape[1]


def fixture(source, lab):
    ns = selected(source, NAMES + (("GDO8", "_gdo8_prepare") if lab else ()))
    records = {"packs": [], "tests": [], "builds": 0}
    mod = NS(gd_info=lambda i: (58, 0, 8, 4096), gd_unpack=lambda *args: None)
    ns["time"] = NS(time=lambda: 0.0)
    ns["torch"] = NS(bfloat16="bf16", int16="int16", empty_like=lambda w: TensorDouble(w.shape),
                     equal=lambda a, b: True, cuda=NS(synchronize=lambda: None, empty_cache=lambda: None,
                     get_device_properties=lambda dev: NS(multi_processor_count=132)))

    def fake_build():
        records["builds"] += 1
        ns["_GD"].update(mod=mod, keys=ns["_gd_keys"]())
        return mod

    def fake_pack(w, wr, wk, ku, s, base):
        records["packs"].append((w.shape, wr, wk, ku, s, base))
        n, k = w.shape
        return dict(w=TensorDouble((n, k)), esc=object(), tb=base * 0x01010101,
                    nbx=n // 16 // wr, G=n // 16 * s * wk, nesc=0)

    def fake_test(r, w, b):
        records["tests"].append((r, w.shape, b))
        assert w.shape == (r["N"], r["K"])
        assert r["T"] >= max(ns["GD_SELF_TEST_T"][b])
        return True, "CPU boundary double; no numerical claim"

    ns.update(_gd_build=fake_build, _gd_pack=fake_pack, _gd_window_base=lambda w: 42, _gd_selftest=fake_test)
    # Names/shapes come from the base's actual dictionaries, not a guessed layer layout.
    state = {"dense": {ns["GD_WEIGHT"][p]: TensorDouble(dims) for p, dims in ns["GD_DIMS"].items()}}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        ns["_gd_prepare"](state, 0.0)
    return ns, state, records, output.getvalue()


def cpu_gate(base, lab):
    assert build(base) == lab, "stale generated output"
    restored = lab
    for old, new, label in reversed(EDITS):
        assert restored.count(new) == 1, label
        restored = restored.replace(new, old, 1)
    assert restored == base, "unscoped edits"
    assert lab.isascii() and len(lab.encode()) < 1_000_000
    assert lab.splitlines()[0] == "# okc: problem=muse_decode_layer"
    b, bs, br, _ = fixture(base, False)
    n, st, nr, log = fixture(lab, True)
    assert not n["GDO8"]
    assert n["GD_ROUTE"] == b["GD_ROUTE"] and n["_gd_keys"]() == b["_gd_keys"]()
    assert n["GD_CPP_SRC"] == b["GD_CPP_SRC"] and n["GD_CUDA_SRC"] == b["GD_CUDA_SRC"]
    assert nr["packs"] == br["packs"] and nr["builds"] == br["builds"] == 1
    assert len(nr["tests"]) == len(br["tests"]) + 1
    gd = st["gd"]
    stock, xp6, alt = gd["route"][("o", 8)], gd["route"][("o", 1)], gd["gdo8"]
    assert n["GD_ROUTE"][("o", 8)] == (1, 1, 8, 2, 2, 1, 0)
    assert n["GD_ROUTE"][("o", 1)] == (1, 1, 8, 2, 2, 1, 6)
    keys = n["_gd_keys"]()
    assert keys[stock["inst"]] == (0, 1, 1, 8, 2, 2, 1, 4, 0)
    assert keys[alt["inst"]] == (0, 1, 1, 8, 2, 2, 1, 4, 6)
    assert alt is not xp6 and alt["inst"] == xp6["inst"]
    assert alt["mod"] is xp6["mod"] is stock["mod"] is gd["mod"]
    assert alt["w"] is stock["w"] is xp6["w"] and alt["esc"] is stock["esc"] is xp6["esc"]
    assert nr["tests"][-1][0] is alt and nr["tests"][-1][2] == 8
    assert len([s for s in log.splitlines() if s.startswith("[gdo8]")]) == 1
    selections = 0
    for arm in (False, True):
        n["GDO8"] = arm
        for proj in (*n["GD_DIMS"], "unknown"):
            for t in range(1, 321):
                got = n["_gd_live"](st, proj, t)
                expected = alt if arm and proj == "o" and 2 <= t <= 8 else b["_gd_live"](st, proj, t)
                assert got is expected, (arm, proj, t)
                selections += 1
    # A GD_ROUTE mutation cannot switch a frozen instance.
    n["GDO8"] = False
    n["GD_ROUTE"][("o", 8)] = n["GD_ROUTE"][("o", 1)]
    assert n["_gd_live"](st, "o", 8) is stock
    n["GD_ROUTE"][("o", 8)] = b["GD_ROUTE"][("o", 8)]
    failures = []
    for case in ("missing_o1", "missing_o8", "test_false", "test_exception", "packing_mismatch", "no_gd"):
        ns, state, _, _ = fixture(lab, True)
        if case.startswith("missing_"):
            del state["gd"]["route"][("o", 1 if case == "missing_o1" else 8)]
        elif case == "test_false":
            ns["_gd_selftest"] = lambda *a: (False, "injected failure")
        elif case == "test_exception":
            def fail(*a):
                raise ValueError("injected failure")
            ns["_gd_selftest"] = fail
        elif case == "packing_mismatch":
            state["gd"]["route"][("o", 1)]["w"] = object()
        else:
            state["gd"] = None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                ns["_gdo8_prepare"](state)
            except RuntimeError:
                assert case == "packing_mismatch"
            else:
                assert case != "packing_mismatch"
        if state["gd"] is not None:
            assert state["gd"]["gdo8"] is None
            ns["GDO8"] = True
            try:
                ns["_gd_live"](state, "o", 8)
            except RuntimeError:
                pass
            else:
                raise AssertionError("failed arm silently selected stock")
            ns["GDO8"] = False
            assert ns["_gd_live"](state, "o", 8) is b["_gd_live"](state, "o", 8)
        failures.append(case)
    for state in ({}, {"gd": None}):
        for arm in (False, True):
            n["GDO8"] = arm
            assert n["_gd_live"](state, "o", 8) is None
    n["GD_ENABLE"] = False
    assert n["_gd_live"](st, "o", 8) is None
    parser_gate()
    return {"status": "PASS", "dry_run": "actual base/lab _gd_prepare, GPU boundaries doubled",
            "selection_checks": selections, "failure_cases": failures, "extra_builds": 0, "extra_packs": 0,
            "extra_device_storage_B": 0, "extra_selftests": 1, "shared_module": True,
            "stock_instance": stock["inst"], "xp6_instance": alt["inst"],
            "host_route_dict_B_this_python": sys.getsizeof(alt), "lab_bytes": len(lab.encode()),
            "parser_negative_tests": "PASS (hoisted x, missing wait, malformed ptxas)",
            "changed_lines": changed_lines(base, lab)}


def changed_lines(base, lab):
    import difflib
    a, b = base.splitlines(), lab.splitlines()
    return [{"base": [i + 1, j], "lab": [k + 1, l], "kind": op}
            for op, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal"]


def parse_ptxas(log):
    rows, current = {}, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            current = rows.setdefault(m[1], {})
        if current is not None:
            m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
            if m:
                current.update(zip(("stack", "spill_stores", "spill_loads"), map(int, m.groups())))
            m = re.search(r"Used (\d+) registers", line)
            if m:
                current["regs"] = int(m[1])
    return rows


def sass_metrics(ins):
    ops = [re.sub(r"^@!?U?P\w+\s+", "", s).split()[0] for _, s, _ in ins]
    wait = [i for i, op in enumerate(ops) if op.startswith("ACQBULK")]
    assert len(wait) == 1, "missing/ambiguous PDL wait"
    # Reuse the established CFG walk, including branches, from RPO's audit.
    helpers = runpy.run_path(str(ROOT / "tools/rpo_sass.py"))
    pre = helpers["_prewait"](ins)
    weight = lambda op: op.startswith("LDG.") and ".NA." in op and ".128" in op
    escape = lambda op: op.startswith("LDG.") and ".CONSTANT" in op and ".128" not in op
    unexpected = [ins[i][1] for i in sorted(pre) if ops[i].startswith("LDG") and not(weight(ops[i]) or escape(ops[i]))]
    assert not unexpected, f"possible pre-wait activation load: {unexpected}"
    return {"instructions": len(ins), "prewait_weight_loads": sum(weight(ops[i]) for i in pre),
            "prewait_escape_loads": sum(escape(ops[i]) for i in pre),
            "postwait_x_loads": sum(op.startswith("LDG.") and ".128" in op and not weight(op)
                                    for i, op in enumerate(ops) if i not in pre),
            "HMMA": sum(op.startswith("HMMA") for op in ops),
            "local_loads_stores": sum(op.startswith(("LDL", "STL")) for op in ops),
            "wait_index": wait[0]}


def parser_gate():
    # Synthetic parser fixtures only; never report these as compiled evidence.
    ops = ["LDG.E.NA.128 R4, [R8]"] * 6 + ["LDG.E.CONSTANT R2, [R12]", "ACQBULK"]
    ops += ["LDG.E.128 R4, [R8]"] * 4 + ["HMMA.16816.F32 R0, R4, R8, R0"] * 8 + ["EXIT"]
    ins = [(16 * i, op, None) for i, op in enumerate(ops)]
    r = sass_metrics(ins)
    assert (r["prewait_weight_loads"], r["postwait_x_loads"], r["HMMA"]) == (6, 4, 8)
    for replacement in ("LDG.E.128 R4, [R8]", "@P0 LDG.E.128.CONSTANT R4, [R8]"):
        bad = list(ins)
        bad[0] = (0, replacement, None)
        try:
            sass_metrics(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("hoisted x was accepted")
    try:
        sass_metrics([row for row in ins if row[1] != "ACQBULK"])
    except AssertionError:
        pass
    else:
        raise AssertionError("missing wait was accepted")
    sample = ("ptxas info : Compiling entry function 'gdo8_fixture' for 'sm_90a'\n"
              "0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
              "ptxas info : Used 58 registers\n")
    assert parse_ptxas(sample)["gdo8_fixture"] == dict(stack=0, spill_stores=0, spill_loads=0, regs=58)
    assert parse_ptxas("") == {} and "stack" not in parse_ptxas(sample.replace("bytes stack frame", "absent"))["gdo8_fixture"]


def linux_gates(base, lab):
    result = {"status": "FAIL", "GPU_launched": False}
    report = Path(__file__).with_suffix(".json")
    try:
        assert os.name != "nt", "Linux runner requires tools/local/cudaenv.sh"
        result["cpu"] = cpu_gate(base, lab)
        lead = subprocess.run([sys.executable, "-B", "-I", str(ROOT / "tools/lead_gate.py"),
                               "file", str(OUT), f"--raw={BASE}"], text=True, capture_output=True, timeout=60)
        result["lead_gate"] = {"returncode": lead.returncode, "output": lead.stdout + lead.stderr}
        assert lead.returncode == 0, "lead file gate failed"
        env = selected(lab, NAMES + ("GDO8", "_gdo8_prepare"))
        keys = env["_gd_keys"]()
        src = env["GD_CUDA_SRC"]
        for marker, macro in (("@INST@", "GD_INST"), ("@INFO@", "GD_INFO")):
            src = src.replace(marker, "\n".join(f"    {macro}({i}, {', '.join(map(str, k))})" for i, k in enumerate(keys)))
        scratch = Path(tempfile.mkdtemp(prefix="gdo8_compile_"))  # intentionally retained; never delete
        result["scratch"] = str(scratch)
        cu, cub = scratch / "gd.cu", scratch / "gd.cubin"
        cu.write_bytes(src.encode("ascii"))
        command = ["nvcc", "-O3", "-std=c++17", "-lineinfo", "-cubin", "-gencode=arch=compute_90a,code=sm_90a",
                   "-Xptxas=-v", str(cu), "-o", str(cub)]
        result["compile_command"] = command
        cp = subprocess.run(command, text=True, capture_output=True, timeout=240)
        (scratch / "ptxas.log").write_bytes((cp.stdout + cp.stderr).encode())
        assert cp.returncode == 0, cp.stderr[-4000:]
        resource = parse_ptxas(cp.stderr)
        obj = subprocess.run(["cuobjdump", "-sass", str(cub)], text=True, capture_output=True, check=True, timeout=60)
        (scratch / "gd.sass").write_bytes(obj.stdout.encode())
        nvd = shutil.which("nvdisasm")
        assert nvd, "nvdisasm missing from cudaenv/triton PATH"
        dis = subprocess.run([nvd, "-c", "-g", "-gi", str(cub)], text=True, capture_output=True, check=True, timeout=60)
        (scratch / "gd.nvdisasm").write_bytes(dis.stdout.encode())
        helper = runpy.run_path(str(ROOT / "tools/rpo_sass.py"))
        funcs = helper["_functions"](dis.stdout)
        result["instances"] = {}
        for xp in (0, 6):
            key = (0, 1, 1, 8, 2, 2, 1, 4, xp, 0, 0)
            matches = [fn for fn in funcs if "gd_kernelI" in fn and tuple(map(int, re.findall(r"Li(\d+)E", fn))) == key]
            assert len(matches) == 1, f"missing/duplicate exact XP{xp} instance: {matches}"
            fn = matches[0]
            r = resource[fn]
            assert all(k in r for k in ("regs", "stack", "spill_stores", "spill_loads")), r
            assert r["stack"] == r["spill_stores"] == r["spill_loads"] == 0, r
            metric = sass_metrics(funcs[fn])
            assert metric["local_loads_stores"] == 0 and metric["HMMA"] == 8, metric
            assert metric["postwait_x_loads"] == 4, metric
            result["instances"][str(xp)] = dict(key=key, resources=r, sass=metric)
        control, alt = (result["instances"][str(x)] for x in (0, 6))
        assert alt["sass"]["prewait_weight_loads"] == 6, "XP6 failed to keep all six weight loads above wait"
        assert alt["resources"]["regs"] <= 64, "XP6 loses required four-CTA residency"
        result["mechanism"] = ("prefetch window changed" if control["sass"]["prewait_weight_loads"] < 6
                               else "XP0 already prefetches: no prefetch saving established")
        assert control["sass"]["prewait_weight_loads"] < 6, result["mechanism"]
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = repr(exc)
    report.write_bytes((json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def main():
    base = BASE.read_text(encoding="utf-8")
    args = sys.argv[1:]
    assert args in ([], ["--cpu-only"], ["--linux-gates"]), "unknown arguments"
    if not args:
        lab = build(base)
        result = cpu_gate(base, lab)  # fail before touching generated output
        OUT.write_bytes(lab.encode("ascii"))
        print(json.dumps(result, indent=2))
        return 0
    lab = OUT.read_text(encoding="utf-8")
    if args == ["--cpu-only"]:
        print(json.dumps(cpu_gate(base, lab), indent=2))
        return 0
    return linux_gates(base, lab)


if __name__ == "__main__":
    raise SystemExit(main())

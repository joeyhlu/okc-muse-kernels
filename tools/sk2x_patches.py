"""SK2X generator and CPU/compile gates. No GPU launches.

Default uses the original SK2 compilation, not the dg=0 control flow of a
recompiled experimental kernel. Enabled arms share one second module and the
original packed weights/table. This makes the off-path SASS proof exact.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUT = ROOT / "kernels/lab/k3_sk2x_ab.py"
Q = "(dg & 16) && NT == 3 && NWR == 1 && WSK == 4 && (st & 1)"
PF = "(dg & 15) && NT == 4 && NWR == 2 && WSK == 2 && pdl && COMP && CL == 1"
PTAB = '''if (COMP) {
if constexpr (CL == 1) {
mbar_init(pbar, 1);
asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
mbar_expect_tx(pbar, pmax * 8);
bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
}
'''
PREFETCH = '''const int PF = dg & 15;
for (int st = P; st < min(nst, P + PF); ++st)
asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(wsrc + (size_t)(gs0 + st) * wstride), "r"((uint32_t)WBY) : "memory");
'''


def replace(s, a, b):
    assert s.count(a) == 1, (a[:100], s.count(a))
    return s.replace(a, b)


def literal(s, key):
    nodes = [n.value for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)]
    assert len(nodes) == 1, key
    return ast.literal_eval(nodes[0])


def cuda_edits():
    edits = []
    for slot in ("st", "slot"):
        old = f"mbar_expect_tx(full0 + 8 * {slot}, XBYr + WBY);\n"
        new = f"if ({Q}) {{\nmbar_expect_tx(full0 + 8 * {slot}, WBY);\n}} else {{\n{old}}}\n"
        edits.append((old, new))
    old = "for (int st = 0; st < P; ++st) tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n"
    new = (f"if ((dg & 16) && NT == 3 && NWR == 1 && WSK == 4) {{\n"
           "for (int st = 0; st < P; ++st) {\n"
           f"if ({Q}) {{ /* TIMING ONLY: odd x stage deliberately stale. */ }} else {{\n"
           "tma_load_3d(sbase + st * SST, &xmap, full0 + 8 * st, 0, 0, (gs0 + st) * UPS);\n"
           f"}}\n}}\n}} else {{\n{old}}}\n")
    edits.append((old, new))
    old = "tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);\n"
    edits.append((old, f"if ({Q}) {{ /* TIMING ONLY: no x transaction expected. */ }} else {{\n{old}}}\n"))
    # Anchor includes the loop start: there are two PDL waits in this kernel.
    old = 'if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\nfor (int st = 0; st < P; ++st)'
    new = f"if ({PF}) {{\n{PREFETCH}{PTAB}}}\n" + old
    edits.insert(0, (old, new))
    # Only wrap the existing post-wait block, before inserting the hoisted copy.
    edits.insert(0, (PTAB, f"if ({PF}) {{ /* ptab already issued before the wait. */ }} else {{\n{PTAB}}}\n"))
    return edits


def patched_cuda(cu):
    for old, new in cuda_edits():
        cu = replace(cu, old, new)
    return cu


def fold_off(cu):
    # Every new guard starts with dg & mask. Reverse the guarded edits and
    # require byte identity, including the original transaction/wait ordering.
    for old, new in reversed(cuda_edits()):
        cu = replace(cu, new, old)
    return cu


BANK = '''SK2_DG={} # call-time only; qkv bit16 is TIMING-ONLY, never submit it
_SK2X={"mod":None,"err":None,"secs":0.0}
def _sk2x_build():
 from torch.utils.cpp_extension import load_inline
 t0=time.time()
 try:
  tag=hashlib.sha1((_SK2_CPP+_SK2_CUDA+" ".join(_SK2_FLAGS)).encode()).hexdigest()[:12]
  kw=dict(cpp_sources=_SK2_CPP,cuda_sources=_SK2_CUDA,functions=["sk2","sk2_set_pdl"],extra_cuda_cflags=list(_SK2_FLAGS),verbose=False)
  try:
   _SK2X["mod"]=load_inline(name=f"sk2_{tag}",no_implicit_headers=True,**kw)
  except TypeError:
   _SK2X["mod"]=load_inline(name=f"sk2_{tag}i",**kw)
 except Exception as ex:
  _SK2X["err"]=repr(ex)[-600:]
 _SK2X["secs"]=time.time()-t0

'''


def generate(s):
    cu = literal(s, "_SK2_CUDA")
    s = replace(s, '_SK2_CUDA=r"""' + cu + '"""',
                '_SK2X_BASE_CUDA=' + repr(cu) + '\n_SK2_CUDA=r"""' + patched_cuda(cu) + '"""')
    start, end = s.index("def _sk2_build():"), s.index("def _sk2_run(")
    build = s[start:end]
    s = s[:start] + BANK + build.replace("_SK2_CUDA", "_SK2X_BASE_CUDA") + s[end:]
    s = replace(s, '  jobs.append(("sk2",_sk2_build))\n',
                '  jobs.append(("sk2",_sk2_build))\n  jobs.append(("sk2x",_sk2x_build))\n')
    s = replace(s, "def _sk2_run(name,x,n,k,pdl):", "def _sk2_run(name,x,n,k,pdl,proj=None):")
    s = replace(s, ' mod=_SK2["mod"]\n mod.sk2_set_pdl(pdl)\n',
                ' mod=_SK2["mod"]\n if SK2_DG.get(proj,0):\n'
                '  mod=_SK2X["mod"]\n  assert mod is not None, _SK2X["err"]\n mod.sk2_set_pdl(pdl)\n')
    s = replace(s, 'ci,32,0,_SK2["dummy"])', 'ci,32,SK2_DG.get(proj,0),_SK2["dummy"])')
    s = replace(s, 'return _sk2_run(p[0],x,p[1],p[2],SK2_PDL.get(proj,0))',
                'return _sk2_run(p[0],x,p[1],p[2],SK2_PDL.get(proj,0),proj)')
    # Join the existing PT1 job, no new state accessor and no prepare-time arm.
    s = replace(s, '  _SK2["ok"]=res\n', '  _SK2["ok"]=res\n  _pt1_wait("sk2x")\n'
                '  if _SK2X["mod"]is None and _SK2X["err"]is None:\n   _sk2x_build()\n')
    return s


def pycompile_memory(path):
    # Run py_compile including bytecode serialization, without creating pyc files.
    saved = []
    with patch("importlib._bootstrap_external._write_atomic", lambda p, data, mode=0o666: saved.append(data)):
        py_compile.compile(str(path), cfile=str(path), doraise=True)
    assert saved and saved[0][:4] == __import__("importlib.util", fromlist=["MAGIC_NUMBER"]).MAGIC_NUMBER


def functions(s, names):
    tree = ast.parse(s)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), names
    return ast.Module(body=nodes, type_ignores=[])


def dry_run(base, generated):
    """Execute real prepare/glue with tensor/extension doubles, never import CUDA.

    State's dense comprehension is extracted from the real base AST, not a
    hand-invented accessor. Packing math is unchanged and mocked, not certified.
    """
    class Tensor:
        dtype, device = "bf16", "cpu"
        def __init__(self, shape=(32, 8192)): self.shape = shape
        def contiguous(self): return self
        def float(self): return self
        def t(self): return Tensor(self.shape[::-1])
        def to(self, *a): return self
        def dim(self): return len(self.shape)
        def is_contiguous(self): return True
        def data_ptr(self): return 16
        def __mul__(self, v): return self
        def __matmul__(self, v): return Tensor((self.shape[0], v.shape[1]))
        def __sub__(self, v): return self
        def norm(self): return self
        def clamp_min(self, v): return self
        def __truediv__(self, v): return 0.0
        def all(self): return True
    calls, builds, waits = [], [], []
    build_mode = ["ok"]
    class Module:
        def __init__(self, label): self.label = label
        def sk2_set_pdl(self, pdl): self.pdl = pdl
        def sk2(self, *args): calls.append((self.label, self.pdl, args))
    def load_inline(**kw):
        if build_mode[0] == "error":
            raise RuntimeError("CPU gate injected compile failure")
        if build_mode[0] == "legacy" and "no_implicit_headers" in kw:
            raise TypeError("CPU gate injected legacy API")
        assert kw["functions"] == ["sk2", "sk2_set_pdl"]
        assert kw["extra_cuda_cflags"] == list(literal(base, "_SK2_FLAGS"))
        builds.append(kw)
        return Module("ship" if kw["cuda_sources"] == literal(base, "_SK2_CUDA") else "experimental")
    fake = SimpleNamespace(bfloat16="bf16", int64="i64", cuda=SimpleNamespace(synchronize=lambda: None),
                           empty=lambda shape, **kw: Tensor(shape), zeros=lambda *a, **kw: Tensor(),
                           randn=lambda shape, **kw: Tensor(shape), equal=lambda *a: True,
                           isfinite=lambda a: a, Generator=lambda **kw: SimpleNamespace(manual_seed=lambda n: None))
    env = dict(torch=fake, time=time, hashlib=hashlib, _pt1_wait=waits.append, HIDDEN=8192, D_EXPERT=4096)
    for key in ("_SK2", "_SK2X", "SK2_DG", "_SK2_CUDA", "_SK2X_BASE_CUDA", "_SK2_CPP", "_SK2_FLAGS",
                "SK2_ENABLE", "SK2_ON", "SK2_PDL", "SK2_T", "_SK2_CFGS", "_SK2_K", "_SK2_PROJ"):
        node = next(n for n in ast.parse(generated).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == key for t in n.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<real SK2 globals>", "exec"), env)
    env.update(_sk2_compress=lambda w: {"packed": w, "base4": 0},
               _sk2_stream=lambda packed, *a: packed, _sk2_table=lambda *a: (Tensor(), 2))
    exec(compile(functions(generated, {"_sk2_build", "_sk2x_build", "_sk2_prepare", "_sk2_run", "_sk2"}),
                 "<real SK2 Python>", "exec"), env)
    # Same dense object the base's state constructor creates.
    dense_expr = [v for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Dict)
                  for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == "dense"
                  and isinstance(v, ast.DictComp)]
    assert len(dense_expr) == 1
    weights = {k: Tensor((n, kdim)) for k, n, kdim in
               (("qkv_gate", 2304, 8192), ("o", 8192, 2048), ("pooler", 4096, 8192), ("unpooler", 8192, 4096))}
    dense = eval(compile(ast.Expression(dense_expr[0]), "<base dense state>", "eval"), {"weights": weights})
    state = {"dense": dense}
    fake_ce = SimpleNamespace(load_inline=load_inline)
    with patch.dict(sys.modules, {"torch": fake, "torch.utils": SimpleNamespace(cpp_extension=fake_ce),
                                  "torch.utils.cpp_extension": fake_ce}), contextlib.redirect_stdout(io.StringIO()):
        env["_sk2_build"]()
        env["_sk2x_build"]()
        env["_sk2_prepare"](state, 0)
    assert len(builds) == 2 and env["_SK2"]["ok"] == {k: True for k in env["_SK2_CFGS"]}
    assert waits == ["sk2", "sk2x"]
    assert len(calls) == 6 and all(label == "ship" and args[-2] == 0 for label, _, args in calls)
    exercised = []
    for arm in ({}, {"qkv": 16}, {"unpooler": 2}, {"unpooler": 3}, {"unpooler": 4}, {}):
        env["SK2_DG"] = arm
        for proj, (name, n, k) in env["_SK2_PROJ"].items():
            result = env["_sk2"](state, proj, Tensor((32, k)))
            label, pdl, args = calls[-1]
            assert result is not None and args[-2] == arm.get(proj, 0)
            assert label == ("experimental" if arm.get(proj, 0) else "ship")
            assert pdl == env["SK2_PDL"][proj] and args[9] == env["_SK2_CFGS"][name][0]
            before = len(calls)
            assert env["_sk2"](state, proj, Tensor((16, k))) is None and len(calls) == before
        exercised.append(arm)
    # New prepare fallback (PT1 disabled/not started), legacy load_inline API,
    # and failure recording all execute locally with the same real function AST.
    env["SK2_DG"] = {}
    env["_SK2X"].update(mod=None, err=None)
    with patch.dict(sys.modules, {"torch": fake, "torch.utils": SimpleNamespace(cpp_extension=fake_ce),
                                  "torch.utils.cpp_extension": fake_ce}), contextlib.redirect_stdout(io.StringIO()):
        env["_sk2_prepare"](state, 0)
        assert env["_SK2X"]["mod"] is not None
        build_mode[0] = "legacy"
        env["_SK2X"].update(mod=None, err=None)
        env["_sk2x_build"]()
        assert env["_SK2X"]["mod"] is not None
        build_mode[0] = "error"
        env["_SK2X"].update(mod=None, err=None)
        env["_sk2x_build"]()
        assert env["_SK2X"]["mod"] is None and "injected compile failure" in env["_SK2X"]["err"]
    return {"prepare": "PASS (real AST, mocked tensor arithmetic/build)", "arms": exercised,
            "T16": "no SK2 launch", "successful_build_calls": len(builds),
            "fallbacks": "PT1-off, legacy API, compile failure all exercised",
            "state": 'state["dense"]', "bank": '_SK2 / _SK2X'}


def cpu_gates(base, generated):
    assert generated == generate(base), "generated file stale"
    cu, original = literal(generated, "_SK2_CUDA"), literal(base, "_SK2_CUDA")
    assert fold_off(cu) == original == literal(generated, "_SK2X_BASE_CUDA")
    assert literal(generated, "SK2_DG") == {}
    # Consumers, barrier arrivals, returns, and epilogue are literally untouched.
    tail = "if (warp >= NW + 1) {"
    assert cu[cu.index(tail):] == original[original.index(tail):]
    for path in (Path(__file__), OUT, ROOT / "tools/box/sk2x_check.py"):
        pycompile_memory(path)
    assert b"\r" not in (ROOT / "tools/local/sk2x_gate.sh").read_bytes()
    # Both real geometries have sixteen stages: precisely eight x issues removed.
    for ups, k, wby, xby, grid in ((8, 8192, 18432, 32768, 96), (4, 4096, 24576, 16384, 128)):
        assert k // (ups * 64) == 16
        for pdl in (0, 2, 3, 4):
            stages = 4 if ups == 8 else 3
            p = min(pdl, stages, 16) if pdl else 1
            issued = list(range(p)) + list(range(p, 16))
            assert issued == list(range(16))
            assert sum(xby for st in issued if not st & 1) == 8 * xby
            for pf in (2, 3, 4):
                assert all(0 <= st < 16 for st in range(p, min(16, p + pf)))
    protocol = []
    for name, grid, rows, nesc in (("qkv", 96, 24, 4), ("unpooler", 128, 64, 3)):
        nc, cp = 128, rows + 1
        # No CTA retires early. Every original warp role still exists, including
        # the skipped-x stages: eight/seven empty arrivals and 128/96 cbar arrivals.
        zrow = [tid - 32 for tid in range(32, 64)]
        assert zrow == list(range(32))
        correction = sorted(i for lane in range(32) for i in range(lane, 32 * cp, 32))
        assert correction == list(range(32 * cp))
        output = sorted(i for tid in range(nc) for i in range(tid, rows // 4 * 32, nc))
        assert output == list(range(rows // 4 * 32))
        nt, wsk = (3, 4) if name == "qkv" else (4, 2)
        rp = rows + ((40 - rows % 32) % 32)
        staging = []
        for tid in range(128):
            warp, lane = tid // 32, tid % 32
            wr, wk = warp // wsk, warp % wsk
            for mt in range(2):
                for tile in range(nt):
                    row, tok = (wr * nt + tile) * 8 + 2 * (lane & 3), mt * 16 + (lane >> 2)
                    r0 = (wk * 32 + tok) * rp + row
                    staging.extend((r0, r0 + 1, r0 + 8 * rp, r0 + 8 * rp + 1))
        assert sorted(staging) == sorted((wk * 32 + tok) * rp + row
                                         for wk in range(wsk) for tok in range(32) for row in range(rows))
        assert len(range(0, 128, 32)) + nesc == 4 + nesc
        protocol.append({"projection": name, "CTAs": [0, grid - 1], "tid0": "init/expect, prefetch+ptab once",
                         "zero_row_tids": [32, 63], "zero_row_bytes": 128,
                         "producer_tids": [128, 159], "producer_issue_tid": 128,
                         "escape_tids": [160, 160 + 32 * nesc - 1],
                         "corr_zero_elements_per_escape_warp": 32 * cp,
                         "empty_arrivals_per_stage": 4 + nesc, "cbar_arrivals": 32 * nesc,
                         "postwait_barrier_initializers": "tid0: empty[0:NS], full[P:NS], cbar, qbar; pbar only if not hoisted",
                         "red_staging_tids": [0, 127], "red_staging_float_elements": len(staging),
                         "combine_tids": [0, 127], "combine_float4_items": len(output),
                         "coverage": "unchanged; producer/escape returns only after their original loops; no rcnt in CL=1"})
    # Exercise the arm-list code without importing torch or starting a GPU path.
    arm_ns = {"__name__": "sk2x_arm_gate", "__file__": str(ROOT / "tools/box/sk2x_check.py")}
    exec(compile((ROOT / "tools/box/sk2x_check.py").read_text(), arm_ns["__file__"], "exec"), arm_ns)
    assert [a["name"] for a in arm_ns["arms"]("all")] == ["ship", "ship2", "xprobe", "pf2", "pf3", "pf4", "ship3"]
    assert all(a["name"] != "xprobe" for a in arm_ns["arms"]("correct"))
    assert [a["name"] for a in arm_ns["arms"]("probe")] == ["ship", "ship2", "xprobe", "ship3"]
    return {"source_default_identical": True, "consumer_protocol_identical": True,
            "py_compile": "PASS (bytecode held in memory)", "dry_run": dry_run(base, generated), "protocol": protocol,
            "accounting": {
                "qkv_stage_bytes_before": {"weight": 18432, "x": 32768},
                "qkv_stage_bytes_after": {"weight": 18432, "x_even": 32768, "x_odd": 0},
                "qkv_issue_inline_PTX_before_after": "5 -> 5 even / 3 odd; expect+policy/bulk+policy/TMA, plus C++ guards",
                "unpooler_stage_bytes_before_after": "weight 24576 -> 24576; x 16384 -> 16384",
                "WEIGHT_REQUEST_FLAG": "each PF target adds one 24576-byte L2 prefetch request, possibly redundant after eviction",
                "PF_added_inline_PTX_per_tid0": "PF instructions before wait, plus address/loop control (SASS pending)",
                "ptab_bytes_before_after": "pmax*8 -> pmax*8 per CTA; same static table, moved before wait",
                "ptab_inline_PTX_before_after": "5 -> 5 per tid0 (init/fence/expect/policy/bulk), moved not duplicated",
                "consumer_instructions": "unchanged byte-for-byte; machine instruction counts require Linux gate"}}


def linux_gate(report):
    import torch.utils.cpp_extension as ce
    import torch
    import sysconfig
    base, lab = BASE.read_text(), OUT.read_text()
    report.update(cpu_gates(base, lab))
    # Retain compile artifacts for inspection; never delete files in this task.
    work = Path(tempfile.mkdtemp(prefix="sk2x_gate_"))
    report["artifacts"] = str(work)
    report["builds"] = {}
    flags = list(literal(base, "_SK2_FLAGS"))
    sources = {"base": literal(base, "_SK2_CUDA"), "off": literal(lab, "_SK2X_BASE_CUDA"),
               "experimental": literal(lab, "_SK2_CUDA")}
    sass = {}
    for name, cu in sources.items():
        directory = work / name
        directory.mkdir()
        # Exact base no-implicit-headers build; C++20 on BOTH host and nvcc.
        mod = ce.load_inline(name="sk2x_gate", cpp_sources=literal(base, "_SK2_CPP"), cuda_sources=cu,
                             functions=["sk2", "sk2_set_pdl"], extra_cuda_cflags=flags,
                             extra_cflags=["-std=c++20"], no_implicit_headers=True,
                             build_directory=str(directory), verbose=True)
        ninja = (directory / "build.ninja").read_text()
        assert "-std=c++17" not in ninja and "-std=c++20" in ninja
        for key in ("cflags =", "cuda_cflags ="):
            assert "-std=c++20" in next(line for line in ninja.splitlines() if line.startswith(key)), key
        inc = sum((["-I", p] for p in ce.include_paths(device_type="cuda")), [])
        inc += ["-I", sysconfig.get_paths()["include"]]
        cmd = ["nvcc", "-cubin", "-std=c++20", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
               "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
               "-DTORCH_EXTENSION_NAME=sk2x_gate", "-DTORCH_API_INCLUDE_EXTENSION_H",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", "-Xptxas=-v"]
        cmd += inc + flags + [str(directory / "cuda.cu"), "-o", str(directory / "sk2.cubin")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        record = report["builds"][name] = {"nvcc_command": cmd, "ptxas": proc.stderr, "returncode": proc.returncode,
                                           "ninja": ninja}
        assert proc.returncode == 0, proc.stderr
        spills = re.findall(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", proc.stderr)
        assert len(spills) == 2, "expected exactly the cfg6/cfg11 resource records"
        record["spill_records"] = spills
        assert all(int(stores) == int(loads) == 0 for _, stores, loads in spills), "new spill"
        txt = subprocess.check_output(["cuobjdump", "-sass", mod.__file__], text=True)
        record["sass"] = txt
        record["sass_opcode_counts"] = dict(__import__("collections").Counter(
            re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b", txt)))
        parsed = {}
        for match in re.finditer(r"Function\s*:\s*(\S+)(.*?)(?=Function\s*:|\Z)", txt, re.S):
            if "sk2_kernel" in match[1]:
                parsed[match[1]] = [" ".join(ln.split()) for ln in match[2].splitlines() if re.search(r"/\*[0-9a-f]+\*/", ln)]
        assert len(parsed) == 2, "missing SASS specialization"
        sass[name] = parsed
    report["off_sass_diff"] = {k: sass["base"][k] != sass["off"].get(k) for k in sass["base"]}
    assert not any(report["off_sass_diff"].values()), "dg=0 actual module SASS changed"
    report["default_sass_proof"] = "actual off module is base, not a constant-folded experimental surrogate"
    report["status"] = "PASS"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="CPU gates; do not write generated file")
    parser.add_argument("--linux-gate", action="store_true")
    args = parser.parse_args()
    if args.linux_gate:
        report = {"status": "FAIL", "GPU": "not run"}
        try:
            linux_gate(report)
        except Exception as ex:
            report["error"] = repr(ex)
            raise
        finally:
            (ROOT / "tools/local/sk2x_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return
    base = BASE.read_text(encoding="utf-8")
    result = generate(base)
    if not args.check:
        OUT.write_bytes(result.encode("utf-8"))
    report = cpu_gates(base, OUT.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

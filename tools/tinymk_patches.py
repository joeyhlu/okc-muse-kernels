"""TINY/TINY2 timestamps; measurement only. No GPU work in the generator.

python -B tools/tinymk_patches.py                 # generate + CPU gates/dry run
WSL cudaenv.sh python -B tools/tinymk_patches.py --compile

Only k3_tinymk.py is emitted: its exact, unchanged base prefix is the control
arm. The appended lab block holds SLOTS and the debug arm, like padz's bank.
Compiler inputs/outputs use Linux memfd; no extra source/log artifacts.

Kill reasons for interpreting marks as savings: (1) stamps can perturb issue,
(2) thread-zero waits can overlap other warps/PDL. This proposes no speedup.
Payload/stage remains 32768 code + 4096 scale = 36864 bytes. Math, loads and
sync instructions are unchanged. First A/B iterations add a guarded timer read
and 8-byte store on TIDX=0; other iterations only test i==0. Six existing sites
are enabled. Timestamp operations/elected thread: before 0; after up to eight
mov.u64 globaltimer reads + eight 64-bit stores (source/PTX-level accounting,
not SASS counts). Other threads store no stamps. Predicate/address SASS counts
and registers/spills require the compile gate; do not infer them from C++.
At most 8*8=64 logical stamp bytes/CTA/launch (132*64=8448 bytes).
The first-byte and doneA intervals include preceding setup, not pure latency.
No early exit/barrier/counter/cleanup change. After the combine wait, the same
TIDX 0..255 of every CTA remain: i0=c*256+tid resets ctr[0:2] (CTA0 tid0..1)
and doneA[0:88] (CTA0 tid0..87); combine uses CTA c<M*4, tid0..255, then
strides c+=grid for the remaining M*4 segments. Helpers TIDX256..287 exit at
the original ws_helper return only. Pre-phase other-ss reset covers 3072
entries at i=c*256+tid, stride G*256; CTA0 tid0 resets the next generation.
"""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import pprint
import re
import subprocess
import sys
import sysconfig
import time
import types

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
OUT = ROOT / "kernels/lab/k3_tinymk.py"
MARKER = "# TINYMK measurement-only bank (unchanged control above)."
SLOTS = {
    0: "entry_before_PDL_wait",
    1: "route_end_before_ss_reset_and_WS_JOIN",
    2: "W13_consumer_phase_end_helper_may_still_drain",
    3: "W2_consumer_phase_end_before_WS_JOIN_and_combine_setup",
    4: "combine_grid_barrier_exit_after_CTA_SYNC",
    5: "combine_end_TIDX0_not_all_thread_exit",
    6: "first_W13_ring_full_wait_passed_TIDX0",
    7: "first_W2_doneA_spin_exit_TIDX0_before_syncwarp",
}
EDITS = (
    ("#define OKC_DEBUG 0", "#define OKC_DEBUG 1"),
    ("mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));",
     "mbar_wait(bars + sl, (uint32_t)((q / OKC_NS) & 1));\n"
     "if (!PB && i == 0) { DBG(6); } // TINYMK: first W13 slot observed full"),
    ("if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64); }",
     "if (lane == 0) { while (ld_acquire(p.doneA + j) < RB_A) __nanosleep(64);\n"
     "if (i == 0) { DBG(7); } // TINYMK: first W2 doneA spin exit\n} "),
)

LAB = '''
TINYMK_ARM = "control"
_TINYMK = {"mods": {}, "sc": None, "full": None, "dbg": None, "pdl": {}}
def _tinymk_src():
    src = _HYBLO_CUDA
    for old, new in _TINYMK_EDITS:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    return src
def _tinymk_build():
    from torch.utils.cpp_extension import load_inline
    src = _tinymk_src()
    for k, _, var in _HYBLO_KEYS:
        fl = list(_HYBLO_FLAGS) + list(var)
        tag = hashlib.sha1((_HYB_MOE_CPP + src + " ".join(fl)).encode()).hexdigest()[:12]
        _TINYMK["mods"][k] = load_inline(name=f"tinymk_{k}_{tag}", cpp_sources=_HYB_MOE_CPP, cuda_sources=src,
                                         functions=["okc_run", "okc_set_pdl"], extra_cuda_cflags=fl, verbose=False)
def _tinymk_prepare(dev):
    assert HYBLO_ENABLE and HYBLO_T == (1, 8) and HYBLO_K == "h"
    assert _HYB["nsm"] > 0 and _HYBLO["w"] is not None
    assert all(_HYBLO["ok"].get(k) and k in _HYBLO["mods"] for k, _, _ in _HYBLO_KEYS), _HYBLO["err"]
    _tinymk_build()
    _TINYMK["sc"], _TINYMK["full"] = _hyb_scratch(dev)
    _TINYMK["dbg"] = torch.zeros(_HYB["nsm"] * 8, dtype=torch.int64, device=dev)
    assert len(_TINYMK["sc"]) == 3
    _TINYMK["sc"].append(_TINYMK["dbg"])  # okc_run: scratch[3] -> p.dbg
    _TINYMK["pdl"] = {}
def _tinymk_select(arm):
    global TINYMK_ARM
    assert arm in ("control", "debug"), arm
    TINYMK_ARM = arm
_tinymk_ship_run = _hyblo_run
def _hyblo_run(k, x, ids, wts):
    if TINYMK_ARM == "control":
        return _tinymk_ship_run(k, x, ids, wts)
    assert TINYMK_ARM == "debug" and k in _TINYMK["mods"], (TINYMK_ARM, k)
    ids, wts = ids.to(torch.int32).contiguous(), wts.to(torch.float32).contiguous()
    mod = _TINYMK["mods"][k]
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    pdl = 1 if HYBLO_PDL else 0
    if _TINYMK["pdl"].get(k) != pdl:
        mod.okc_set_pdl(pdl)
        _TINYMK["pdl"][k] = pdl
    mod.okc_run(_HYBLO["w"], _TINYMK["sc"], x, ids, wts, out, 0, 32, _HYB["nsm"])
    return out
_tinymk_ship_prepare = prepare
def prepare(weights):
    _tinymk_select("control")
    state = _tinymk_ship_prepare(weights)
    _tinymk_prepare(weights["w13"].device)
    return state
'''


def assignments(tree, names, ns):
    found = set()
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            exec(compile(ast.Module([n], []), str(BASE), "exec"), ns)
            found.add(n.targets[0].id)
    assert found == set(names), set(names) - found


def base_ns(src):
    ns = {}
    assignments(ast.parse(src), ("_HYBLO_CUDA", "_HYBLO_KEYS", "_HYBLO_FLAGS", "_HYB_MOE_CPP"), ns)
    return ns


def debug_source(cu):
    for old, new in EDITS:
        assert cu.count(old) == 1, old
        cu = cu.replace(old, new)
    return cu


def generate(src):
    debug_source(base_ns(src)["_HYBLO_CUDA"])
    # Multiline literals keep the exact added CUDA stamp lines reviewable in
    # the generated file, without creating an unrequested slots sidecar.
    edits = "_TINYMK_EDITS = (\n" + "".join(
        f'    ({old!r}, r"""{new}"""),\n' for old, new in EDITS) + ")\n"
    return src + "\n" + MARKER + "\nSLOTS = " + pprint.pformat(SLOTS, sort_dicts=True) + "\n" + edits + LAB


def gates(src, result):
    assert result.startswith(src + "\n" + MARKER), "control is not the unchanged base"
    assert result.isascii() and "\r" not in result
    compile(result, str(OUT), "exec")  # no pycache files
    ns = base_ns(src)
    dbg = debug_source(ns["_HYBLO_CUDA"])
    restored = dbg
    for old, new in reversed(EDITS):
        restored = restored.replace(new, old)
    assert restored == ns["_HYBLO_CUDA"]
    assert re.findall(r"DBG\((\d)\);", dbg).count("6") == 1
    assert re.findall(r"DBG\((\d)\);", dbg).count("7") == 1
    assert tuple(k for k, _, _ in ns["_HYBLO_KEYS"]) == ("tiny", "tiny2")
    assert "p.dbg = scratch.size() > 3 ? (unsigned long long*)scratch[3].data_ptr() : nullptr;" in dbg
    assert "#define TIDX ((int)((threadIdx.x + NT) % NTL))" in dbg
    assert "#define DBG(k) do { if (OKC_DEBUG && TIDX == 0 && p.dbg)" in dbg
    # Only these two source insertions and the debug define may differ.
    # Sync/return/counter/cleanup coverage therefore remains byte-for-byte intact.
    print("[tinymk] AST/source gates PASS; control prefix SHA256", hashlib.sha256(src.encode()).hexdigest())


def dry_run(src, result):
    """Execute real base declarations/scratch/run/route and every new Python path.

    GPU tensors and load_inline are strict CPU fakes; no layer math is emulated.
    The k2a33 return dict and its access are taken from the actual base AST.
    """
    class Tensor:
        def __init__(self, shape, dtype=None, device="cpu"):
            self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.dtype, self.device = dtype, device
        def __getitem__(self, key):
            assert isinstance(key, slice)
            a, b, step = key.indices(self.shape[0])
            return Tensor((len(range(a, b, step)),), self.dtype, self.device)
        def view(self, dtype):
            return Tensor(self.shape, dtype, self.device)
        def fill_(self, value):
            return self
        def to(self, dtype):
            return self.view(dtype)
        def contiguous(self):
            return self

    builds, launches = [], []
    class Module:
        def __init__(self, name):
            self.name, self.pdl = name, None
        def okc_set_pdl(self, value):
            assert value in (0, 1)
            self.pdl = value
        def okc_run(self, w, sc, x, ids, tw, out, lo, nloc, grid):
            assert len(w) == 14 and grid == 132 and (lo, nloc) == (0, 32)
            assert len(sc) == (4 if self.name.startswith("tinymk_") else 3)
            if len(sc) == 4:
                assert sc[3].shape == (grid * 8,) and sc[3].dtype == "int64"
            assert out.shape == (x.shape[0], 4096) and self.pdl is not None
            launches.append((self.name, x.shape[0], self.pdl))

    def fake_build(**kw):
        assert kw["cuda_sources"] == debug_source(ns["_HYBLO_CUDA"])
        builds.append(kw["name"])
        return Module(kw["name"])

    fake_torch = types.ModuleType("torch")
    for dtype in ("int16", "int32", "int64", "float32", "bfloat16"):
        setattr(fake_torch, dtype, dtype)
    fake_torch.zeros = fake_torch.empty = Tensor
    utils, ce = types.ModuleType("torch.utils"), types.ModuleType("torch.utils.cpp_extension")
    ce.load_inline = fake_build
    old_modules = {k: sys.modules.get(k) for k in ("torch", "torch.utils", "torch.utils.cpp_extension")}
    sys.modules.update({"torch": fake_torch, "torch.utils": utils, "torch.utils.cpp_extension": ce})
    try:
        tree, ns = ast.parse(src), {"torch": fake_torch, "hashlib": hashlib}
        names = ("_HYB", "_HYBLO", "_HYB_MMAX", "_HYB_EMAX", "_HYB_FIXF", "_HYB_XROWS", "_HYB_GUARD",
                 "HYBLO_ENABLE", "HYBLO_T", "HYBLO_K", "HYBLO_PDL", "_HYBLO_KEYS", "_HYBLO_FLAGS", "_HYBLO_CUDA", "_HYB_MOE_CPP")
        assignments(tree, names, ns)
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name in ("_hyb_scratch", "_hyblo_key", "_hyblo_run", "_hyblo_route"):
                exec(compile(ast.Module([n], []), str(BASE), "exec"), ns)
        kprep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_v11_k2a33_prepare")
        ret = next(n.value for n in ast.walk(kprep) if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
        st = object()
        kstate = eval(compile(ast.Expression(ret), str(BASE), "eval"), {"st": st, "ok": {1: True, 8: True}})
        state = {"k2a33": kstate}
        prep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        call = next(n for n in ast.walk(prep) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_hyblo_prepare")
        assert eval(compile(ast.Expression(call.args[0]), str(BASE), "eval"), {"state": state}) is st
        ns["_HYB"].update(nsm=132, ok=True)
        ns["_HYBLO"].update(w=[object() for _ in range(14)], ok={"tiny": True, "tiny2": True},
                              mods={k: Module(k) for k in ("tiny", "tiny2")})
        ns["_HYBLO"]["sc"], ns["_HYBLO"]["full"] = ns["_hyb_scratch"]("cpu")
        ns["prepare"] = lambda weights: state  # unchanged base is outside new prepare paths
        ns["_hyb_call"] = lambda *args: "wide-control"
        exec(compile(result.split(MARKER, 1)[1], str(OUT), "exec"), ns)
        assert ns["prepare"]({"w13": Tensor((1,), device="cpu")}) is state
        assert len(builds) == 2 and len(ns["_TINYMK"]["full"]) == 3
        assert ns["_TINYMK"]["sc"][3] is ns["_TINYMK"]["dbg"]
        for arm in ("control", "debug"):
            ns["_tinymk_select"](arm)
            for pdl in (True, True, False):  # set, cached, reset branches
                ns["HYBLO_PDL"] = pdl
                for m in (1, 8):
                    k = ns["_hyblo_key"](m)
                    assert k == ("tiny" if m == 1 else "tiny2")
                    out = ns["_hyblo_route"](Tensor((m, 4096)), Tensor((m, 8)), Tensor((m, 8)))
                    assert out.shape == (m, 4096)
        assert len(launches) == 12
        ns["_tinymk_select"]("control")
        assert ns["_hyblo_run"]("w", Tensor((1, 4096)), Tensor((1, 8)), Tensor((1, 8))) == "wide-control"
        for arm in ("bad", None):
            try:
                ns["_tinymk_select"](arm)
            except AssertionError:
                pass
            else:
                raise AssertionError("invalid arm accepted")
        ns["_tinymk_select"]("debug")
        try:
            ns["_hyblo_run"]("missing", Tensor((1, 4096)), Tensor((1, 8)), Tensor((1, 8)))
        except AssertionError:
            pass
        else:
            raise AssertionError("missing debug module accepted")
        ns["_HYBLO"]["ok"]["tiny"] = False
        try:
            ns["_tinymk_prepare"]("cpu")
        except AssertionError:
            pass
        else:
            raise AssertionError("unavailable control accepted")
        print("[tinymk] CPU dry run PASS: base AST layout/scratch, two builds, both arms/routes, PDL set/cache/reset, fail-closed guards")
    finally:
        for k, old in old_modules.items():
            if old is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = old


def compile_gate(src):
    """sm_90a device compile using the same torch prelude/flags as hyblo_sass.

    Compiler-created temporaries are managed by nvcc; our sources/cubins stay
    in anonymous RAM files. Prints ptxas resources, retaining no extra files.
    """
    import torch.utils.cpp_extension as ce
    assert hasattr(os, "memfd_create"), "run --compile inside WSL via tools/local/cudaenv.sh"
    ns = base_ns(src)
    inc = sum((["-I", p] for p in ce.include_paths(device_type="cuda") if os.path.isdir(p)), [])
    inc += ["-I", sysconfig.get_paths()["include"]]
    common = ["nvcc", "-x", "cu", "-cubin", "-std=c++20", "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
              "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr",
              "-DTORCH_EXTENSION_NAME=tinymk_gate", "-DTORCH_API_INCLUDE_EXTENSION_H", "-D_GLIBCXX_USE_CXX11_ABI=1",
              "-Xptxas=-v,--warn-on-spills"] + inc
    resources = {}
    for k, _, var in ns["_HYBLO_KEYS"]:
        for arm in ("control", "debug"):
            cu = ns["_HYBLO_CUDA"] if arm == "control" else debug_source(ns["_HYBLO_CUDA"])
            fds = [os.memfd_create("tinymk_src"), os.memfd_create("tinymk_cubin")]
            paths = [f"/proc/{os.getpid()}/fd/{fd}" for fd in fds]
            try:
                os.write(fds[0], ("#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n" + cu).encode())
                t0 = time.monotonic()
                p = subprocess.run(common + list(ns["_HYBLO_FLAGS"]) + list(var) + [paths[0], "-o", paths[1]],
                                   capture_output=True, text=True, timeout=300)
                print(f"[tinymk] compile {k}/{arm} {time.monotonic()-t0:.1f}s rc={p.returncode}", flush=True)
                print(p.stderr, flush=True)
                assert p.returncode == 0, f"compile failed {k}/{arm}"
                rows = {}
                name = None
                for line in p.stderr.splitlines():
                    m = re.search(r"Function properties for (\S+)", line)
                    if m:
                        name = m[1]
                        rows[name] = {}
                    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
                    if m and name:
                        rows[name].update(zip(("stack", "spill_stores", "spill_loads"), map(int, m.groups())))
                    m = re.search(r"Used (\d+) registers", line)
                    if m and name:
                        rows[name]["regs"] = int(m[1])
                assert rows and any("moe_fused" in n for n in rows), "missing ptxas resources"
                resources[(k, arm)] = rows
            finally:
                for fd in fds:
                    os.close(fd)
        a, b = resources[(k, "control")], resources[(k, "debug")]
        assert a.keys() == b.keys(), (a.keys(), b.keys())
        for name in a:
            for metric in ("spill_stores", "spill_loads"):
                assert b[name][metric] <= a[name][metric], (k, name, metric, a[name], b[name])
            print(f"[tinymk] {k} {name}: control {a[name]} debug {b[name]}")
    print("[tinymk] sm_90a compile/no-added-spills gate PASS", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--check", action="store_true", help="read-only gates, requiring existing generated output")
    args = ap.parse_args()
    src = BASE.read_bytes().decode("ascii")
    result = generate(src)
    gates(src, result)
    dry_run(src, result)
    if args.compile:
        compile_gate(src)
    elif args.check:
        assert OUT.read_bytes() == result.encode("ascii"), "regenerate k3_tinymk.py"
    else:
        OUT.write_bytes(result.encode("ascii"))
        print(f"[tinymk] emitted {OUT.name}: {len(result):,} B; SLOTS and unchanged control embedded")


if __name__ == "__main__":
    main()

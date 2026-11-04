"""HYAEX lab generator and CPU / compile-only gates. Never starts a GPU job.

Only k3_pure_c1's HYA pack2 flat-L8 path changes. Variant 1 is deliberately
incorrect. No expert routing changes. Run with -B; --cpu-gate writes nothing.
The Linux gate retains its temporary compiler outputs (never deletes files).
"""
import argparse
import ast
import contextlib
import difflib
import hashlib
import io
import json
import math
import pathlib
import py_compile
import re
import runpy
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/muse_decode_layer/k3_pure_c1.py"
OUT = ROOT / "kernels/lab/k3_hyaex_ab.py"
COEFFICIENTS = (1.0000001192092896, 0.6931469440460205, 0.24022120237350464,
                0.05550713092088699, 0.009675540961325169, 0.0013276472454890609)
# Degree-5 relative minimax for 2**f on [-.5,.5], rounded to binary32.
# Remez real coefficients, c0..c5:
# 1.0000000716546849, .69314696706476042, .24022119723840085,
# .055507132734986267, .0096755413344491691, .0013276471992296065.
# Real minimax ripple 7.49365e-8; the gate measures rounded FMA evaluation.
FMA = r'''__device__ __forceinline__ float hyaex_exp(float x) {
// Keep masked -inf below the E4M3 zero midpoint; cap only FP32 overflow.
// Positive logits ARE possible before the unchanged PB repack check.
x = fminf(fmaxf(x, -126.f), 127.f);
const float t = __fadd_rn(x, 12582912.f);
const float f = __fsub_rn(x, __fsub_rn(t, 12582912.f));
float p = __fmaf_rn(0.0013276472454890609f, f, 0.009675540961325169f);
p = __fmaf_rn(p, f, 0.05550713092088699f);
p = __fmaf_rn(p, f, 0.24022120237350464f);
p = __fmaf_rn(p, f, 0.6931469440460205f);
p = __fmaf_rn(p, f, 1.0000001192092896f);
return __uint_as_float(__float_as_uint(p) + (__float_as_uint(t) << 23));
}
'''
NULL = r'''__device__ __forceinline__ float hyaex_exp(float x) {
// Timing upper bound ONLY: intentionally wrong, never a shipping candidate.
return __int_as_float(0x3f800000 | (__float_as_int(x) & 7));
}
'''
RESOURCE_CPP = "\nstd::vector<int64_t> hyaex_resources();\n"
RESOURCE_CUDA = r'''
std::vector<int64_t> hyaex_resources() {
cudaFuncAttributes a{};
int occ = 0;
constexpr int sm = sizeof(Smem<8, true>);
TORCH_CHECK(cudaFuncSetAttribute(attn_kernel<8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess);
TORCH_CHECK(cudaFuncGetAttributes(&a, attn_kernel<8, true>) == cudaSuccess);
TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, attn_kernel<8, true>, Cfg<8, true>::NTX, sm) == cudaSuccess);
return {a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes, sm, occ,
        Cfg<8, true>::CREGS, Cfg<8, true>::PREGS};
}
'''
LAB = '''
# HYAEX BEGIN: default-off lab bank; variant 1 is timing-only, never ship.
HYAEX = {}
HYAEX_TWIN = False
HYAEX_BUILD = False
_HYAEX = {k: {"mod": None, "ws": {}, "build_s": 0.0, "err": None, "resources": None}
          for k in ("twin", 1, 2, 3)}

def _hyaex_src(k):
    assert k in (0, "twin", 1, 2, 3), k
    if k in (0, "twin"):
        return HYA_CUDA_SRC
    s = HYA_CUDA_SRC
    begin = "auto pack2 = [&]() {\\n"
    end = "\\n};\\npack2();"
    assert s.count(begin) == 1
    lo = s.index(begin) + len(begin)
    hi = s.index(end, lo)
    body = s[lo:hi]
    changed = []
    for line in body.splitlines(keepends=True):
        selected = k == 1 or "pa[kk][3] =" in line or (k == 3 and "pa[kk][2] =" in line)
        changed.append(line.replace("ex2(", "hyaex_exp(") if selected else line)
    new = "".join(changed)
    assert new.count("hyaex_exp(") == {1: 16, 2: 4, 3: 8}[k]
    # Discarded branch leaves non-flat/L1 instantiations unchanged; SASS gate too.
    s = s[:lo] + "if constexpr (L == 8 && FLAT) {\\n" + new + "\\n} else {\\n" + body + "\\n}" + s[hi:]
    anchor = "__device__ __forceinline__ float ex2(float x) {"
    assert s.count(anchor) == 1
    return s.replace(anchor, (_HYAEX_NULL if k == 1 else _HYAEX_FMA) + anchor) + _HYAEX_RESOURCE_CUDA

def _hyaex_build(k):
    H = _HYAEX[k]
    if H["mod"] is not None:
        return H["mod"]
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    H["err"] = None
    try:
        src = _hyaex_src(k)
        cpp = HYA_CPP_SRC + ("" if k == "twin" else _HYAEX_RESOURCE_CPP)
        tag = hashlib.sha1((cpp + src).encode()).hexdigest()[:12]
        H["mod"] = load_inline(name=f"hyaex_{k}_{tag}", cpp_sources=cpp, cuda_sources=src,
            functions=["run"] + ([] if k == "twin" else ["hyaex_resources"]),
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                               "--expt-relaxed-constexpr", "-Xptxas=-v,-warn-spills"], verbose=False)
    except Exception as ex:
        H["err"] = repr(ex)
        raise
    finally:
        H["build_s"] = time.time() - t0
    return H["mod"]

def _hyaex_prepare(state, t0):
    if not HYAEX_BUILD:
        return
    if not state.get("hya"):
        raise RuntimeError("HYAEX requires the real HYA prepare to succeed")
    dev = state["router"].device  # exact _hya_prepare accessor
    for k, H in _HYAEX.items():
        mod = _hyaex_build(k)
        if k != "twin":
            H["resources"] = list(mod.hyaex_resources())
            reg, local, static, dynamic, occ, cr, pr = H["resources"]
            if reg > 80 or local or occ < 2 or cr != 104 or pr != 32:
                raise RuntimeError(f"HYAEX {k} resource gate: {H['resources']}")
        if str(dev) not in H["ws"]:
            n = HYA_WS_ROWS  # exact _hya_ws layout, one independent bank per arm
            H["ws"][str(dev)] = (torch.zeros(n*64, dtype=torch.float32, device=dev),
                torch.zeros(n, dtype=torch.float32, device=dev), torch.zeros(n, dtype=torch.int32, device=dev))
        print(f"[hyaex] arm {k} build {H['build_s']:.3f}s resources {H['resources']}", flush=True)

def _hyaex_bank(cfg, plan, dev, ws):
    eligible = cfg["batch"] in (16, 32) and cfg["q_len"] == 8 and not cfg["window"] and plan["flat"] == 1
    if not eligible:
        return _HYA["mod"], ws
    v = HYAEX.get((cfg["batch"], "full"), 0)
    if type(v) is not int or v not in (0, 1, 2, 3):
        raise ValueError(f"invalid HYAEX arm {v}")
    if not v and not HYAEX_TWIN:
        return _HYA["mod"], ws
    if v and HYAEX_TWIN:
        raise ValueError("HYAEX_TWIN and a nonzero HYAEX arm are mutually exclusive")
    H = _HYAEX[v if v else "twin"]
    if H["mod"] is None or H["err"] or str(dev) not in H["ws"]:
        raise RuntimeError("HYAEX bank not prepared; set HYAEX_BUILD=True BEFORE prepare")
    bank_ws = H["ws"][str(dev)]
    if bank_ws[1].numel() < cfg["batch"] * (cfg["num_kv_heads"]//2) * plan["nc"] * 8 * 16:
        raise RuntimeError("HYAEX workspace too small")
    return H["mod"], bank_ws
# HYAEX END
'''
PREP_A = ' _rq_prepare(state)\n'
PREP_B = PREP_A + ' _hyaex_prepare(state,t0)\n'
CALL_A = ' out=torch.empty((B,L,hq*64),dtype=torch.bfloat16,device=dev)\n _HYA["mod"].run('
CALL_B = ' hmod,ws=_hyaex_bank(cfg,plan,dev,ws)\n out=torch.empty((B,L,hq*64),dtype=torch.bfloat16,device=dev)\n hmod.run('
END = '"HYA END (derived code above: HYA_CPP_SRC, HYA_CUDA_SRC, _hya_plan, _hya_launch; see _HYA_NOTICE)"\n'


def sub(text, old, new):
    assert text.count(old) == 1, (text.count(old), old[:80])
    return text.replace(old, new)


def literal(source, name):
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def lab_text():
    return ("\n_HYAEX_FMA = " + repr(FMA) + "\n_HYAEX_NULL = " + repr(NULL)
            + "\n_HYAEX_RESOURCE_CPP = " + repr(RESOURCE_CPP)
            + "\n_HYAEX_RESOURCE_CUDA = " + repr(RESOURCE_CUDA) + "\n" + LAB)


def build():
    base = BASE.read_bytes().decode("ascii")
    assert "\r" not in base
    out = sub(sub(sub(base, PREP_A, PREP_B), CALL_A, CALL_B), END, END + lab_text())
    return base, out


def namespace(source):
    names = {"HYA_CPP_SRC", "HYA_CUDA_SRC", "_HYA", "HYA_ENABLE", "HYA_ROUTE", "HYA_DEFAULT",
             "HYA_PDL", "HYA_PDL_T", "HYA_TUNED", "HYA_PREPARE_BY_S", "HYA_WS_ROWS"}
    funcs = {"_hya_build", "_hya_ws", "_hya_inv_freq", "_hya_sms", "_hya_plan", "_hya_launch", "_hya_prepare"}
    nodes = [n for n in ast.parse(source).body if
             (isinstance(n, ast.FunctionDef) and (n.name in funcs or n.name.startswith("_hyaex_"))) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
              (t.id in names or t.id.startswith(("_HYAEX", "HYAEX"))) for t in n.targets))]
    ns = dict(hashlib=hashlib, time=time)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(OUT), "exec"), ns)
    return ns


def dry_run(source):
    # Same padz-style bank / real layer functions; only CUDA boundaries mocked.
    class Tensor:
        def __init__(self, shape, device="cpu-mock"):
            self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
            self.device = device
        def numel(self): return math.prod(self.shape)
        def float(self): return self
        def contiguous(self): return self
        def zero_(self): return self
        def __truediv__(self, _): return self
        def __rtruediv__(self, _): return self
        def __rpow__(self, _): return self

    class Module:
        def __init__(self, name): self.name, self.calls = name, []
        def run(self, *a): self.calls.append(a)
        def hyaex_resources(self): return [80, 0, 0, 114688, 2, 104, 32]

    allocs, builds = [], []
    def tensor(shape, **kw):
        t = Tensor(shape, kw.get("device", "cpu-mock")); allocs.append(t); return t
    def loader(**kw):
        builds.append(kw)
        return Module(kw["name"])
    torch = types.ModuleType("torch")
    torch.zeros = torch.empty = tensor
    torch.arange = lambda *a, **kw: Tensor(32, kw.get("device", "cpu-mock"))
    torch.float32, torch.int32, torch.bfloat16 = "f32", "i32", "bf16"
    torch.cuda = types.SimpleNamespace(get_device_properties=lambda d: types.SimpleNamespace(multi_processor_count=132))
    ext = types.ModuleType("torch.utils.cpp_extension"); ext.load_inline = loader
    ns = namespace(source)
    ns.update(torch=torch, _pt1_wait=lambda _: None, _PT1={"errors": {}},
              ROPE_THETA=500000., THIRD_PARTY_NOTICES="dry", _HYA_NOTICE="dry", _hya_self_test=lambda _: 0.)
    with patch.dict(sys.modules, {"torch": torch, "torch.utils.cpp_extension": ext}), contextlib.redirect_stdout(io.StringIO()):
        state = {"router": Tensor((1,))}
        ns["_hya_prepare"](state, time.time())  # real base state constructor
        assert state["hya"]
        ns["_hyaex_prepare"](state, time.time())
        assert len(builds) == 1, "default must not compile an experiment"
        ns["HYAEX_BUILD"] = True
        ns["_hyaex_prepare"](state, time.time())
        nalloc = len(allocs)
        ns["_hyaex_prepare"](state, time.time())
        assert len(builds) == 5 and len(allocs) == nalloc
        banks = [ns["_HYA"]] + list(ns["_HYAEX"].values())
        assert len({id(b["mod"]) for b in banks}) == 5
        assert len({id(t) for b in banks for t in b["ws"]["cpu-mock"]}) == 15
        cases = 0
        for B in (1, 8, 16, 32, 33):
            for L in (1, 8):
                for win in (0, 2048):
                    for flat in (0, 1):
                        for arm in (0, "twin", 1, 2, 3):
                            ns["HYAEX"] = {(B, "full"): 0 if arm == "twin" else arm}
                            ns["HYAEX_TWIN"] = arm == "twin"
                            cfg = dict(batch=B,q_len=L,window=win,page_size=1,num_kv_heads=2,num_q_heads=16,
                                       rope_theta=0.,sm_scale=.125,qk_scale_factor=5.473)
                            q = Tensor((B*L,2304))
                            d = {k: Tensor(1) for k in ("seq_lens","cache_loc","k_cache","v_cache","k_scale","v_scale")}
                            d["page_table"] = Tensor((B,80008))
                            before = [len(b["mod"].calls) for b in banks]
                            ns["_hya_launch"](q,d,cfg,True,{"flat":flat})
                            chosen = arm if B in (16,32) and L == 8 and not win and flat else 0
                            bank = ns["_HYA"] if chosen == 0 else ns["_HYAEX"][chosen]
                            assert [len(b["mod"].calls)-n for b,n in zip(banks,before)] == [int(b is bank) for b in banks]
                            assert bank["mod"].calls[-1][10:13] == bank["ws"]["cpu-mock"]
                            cases += 1
        def fails(f, error):
            try: f()
            except error: return
            raise AssertionError("failure path silently accepted")
        cfg.update(batch=16,q_len=8,window=0)
        ns["HYAEX_TWIN"] = False
        ws = ns["_HYA"]["ws"]["cpu-mock"]
        plan = dict(flat=1,nc=1)
        choose = lambda: ns["_hyaex_bank"](cfg,plan,"cpu-mock",ws)
        for v in (-1,4,True,"2"):
            ns["HYAEX"] = {(16,"full"):v}; fails(choose,ValueError)
        ns["HYAEX"] = {(16,"full"):2}; ns["HYAEX_TWIN"] = True
        fails(choose,ValueError); ns["HYAEX_TWIN"] = False
        H = ns["_HYAEX"][2]
        saved = H["ws"]; H["ws"] = {}; fails(choose,RuntimeError); H["ws"] = saved
        fails(lambda: ns["_hyaex_bank"](cfg,dict(flat=1,nc=100000),"cpu-mock",ws),RuntimeError)
        fails(lambda: ns["_hyaex_prepare"]({"hya":False},time.time()),RuntimeError)
        H["mod"].hyaex_resources = lambda: [80,0,0,114688,1,104,32]
        fails(lambda: ns["_hyaex_prepare"](state,time.time()),RuntimeError)
        H["mod"] = None
        ext.load_inline = lambda **kw: (_ for _ in ()).throw(RuntimeError("dry compile failure"))
        fails(lambda: ns["_hyaex_build"](2),RuntimeError)
        assert "dry compile failure" in H["err"]
    return dict(actual_launch_paths=cases, distinct_modules=5, independent_workspaces=5,
                real_base_prepare=True, default_no_build=True, failure_paths=True)


def polynomial_check():
    import numpy as np
    c = np.array(COEFFICIENTS, dtype=np.float32)
    def evaluate(x):
        x = np.asarray(x, dtype=np.float32).clip(np.float32(-126), np.float32(127))
        t = np.float32(x + np.float32(12582912))
        f = np.float32(x - np.float32(t - np.float32(12582912)))
        p = np.full_like(x, c[5])
        # Exact product of two binary32s fits binary64; one final binary32
        # rounding models fma.rn for this bounded polynomial (no cancellation).
        for a in c[4::-1]:
            p = np.float32(p.astype(np.float64)*f.astype(np.float64)+np.float64(a))
        return (p.view(np.uint32)+(t.view(np.uint32)<<np.uint32(23))).view(np.float32)
    worst, at, count = 0., 0., 0
    def check(x):
        nonlocal worst, at, count
        x = np.asarray(x,dtype=np.float32)
        e = np.abs(evaluate(x).astype(np.float64)/np.exp2(x.astype(np.float64))-1.)
        i = int(e.argmax())
        if e[i] > worst: worst,at = float(e[i]),float(x[i])
        count += x.size
    # Dense complete used interval; all representable fractions on the
    # negative half of each RN cell at >=1, plus fine fractional coverage.
    for lo in range(-126,0):
        check(np.linspace(lo,lo+1,65537,dtype=np.float32))
    check(np.linspace(-1,0,2000001,dtype=np.float32))
    edge = np.arange(-126.,.01,.5,dtype=np.float32)
    check(np.concatenate([edge,np.nextafter(edge,np.float32(-np.inf))[1:],np.nextafter(edge,np.float32(np.inf))[:-1]]))
    assert worst <= 3e-7, (worst,at)
    masked = float(evaluate([-np.inf])[0])
    assert masked < 2**-10, "must round to exactly E4M3 zero"
    # PB sees the same overflow class even before repacking positive logits.
    assert bool(np.all(evaluate([9.,127.,128.,1000.,np.inf]) > 448.))
    return dict(max_relative_error_MEASURED=worst, at=at, samples=count, interval=[-126,0],
                float32_coefficients=list(COEFFICIENTS), masked_before_E4M3=masked,
                masked_E4M3=0, positive_overflow_preserved=True,
                method="numpy binary32; FMA emulated by exact binary64 product + single binary32 rounding; dense sweep, not exhaustive proof")


def pycompile_without_writes(paths):
    # Exercise actual py_compile including bytecode serialization without extra
    # repository files. Linux additionally writes .pyc to its retained build dir.
    captured = []
    with patch("importlib._bootstrap_external._write_atomic", side_effect=lambda *a,**k: captured.append(a[1])):
        for p in paths:
            py_compile.compile(str(p), cfile=str(ROOT/"hyaex-memory.pyc"), doraise=True)
    assert len(captured) == len(paths)


def gates(base, source):
    compile(source,str(OUT),"exec")
    assert OUT.read_bytes() == source.encode("ascii"), "regenerate the lab file"
    restored = sub(sub(sub(source,lab_text(),""),PREP_B,PREP_A),CALL_B,CALL_A)
    assert restored == base, "unexpected changes outside HYAEX"
    assert literal(source,"HYAEX") == {} and literal(source,"HYAEX_TWIN") is False
    assert literal(source,"HYAEX_BUILD") is False
    for name in ("HYA_CUDA_SRC","HYA_CPP_SRC"):
        assert literal(source,name) == literal(base,name)
    ns = namespace(source); ship = ns["_hyaex_src"](0)
    assert ship == ns["_hyaex_src"]("twin") == literal(base,"HYA_CUDA_SRC")
    changed = {}
    begin, end = "auto pack2 = [&]() {\n", "\n};\npack2();"
    lo = ship.index(begin)+len(begin); hi = ship.index(end,lo)
    original = ship[lo:hi]
    for k in (1,2,3):
        cu = ns["_hyaex_src"](k)
        a = cu.index(begin)+len(begin); b = cu.index(end,a)
        branch = cu[a:b]
        assert branch.startswith("if constexpr (L == 8 && FLAT) {\n")
        assert branch.split("\n} else {\n",1)[1] == original+"\n}"
        back = cu[:a]+original+cu[b:]
        back = sub(sub(back, NULL if k==1 else FMA, ""), RESOURCE_CUDA, "")
        assert back == ship, "L1/non-flat/protocol/PB/Cfg/launch changes"
        changed[k] = [{"kind":tag,"base_lines":[i+1,j] if i<j else [],
                       "insert_after_base_line":i if i==j else None,"variant_lines":[u+1,v]}
                      for tag,i,j,u,v in difflib.SequenceMatcher(None,ship.splitlines(),cu.splitlines(),autojunk=False).get_opcodes() if tag!="equal"]
    shell = (ROOT/"tools/local/hyaex_gate.sh").read_bytes()
    assert b"\r" not in shell
    pycompile_without_writes([pathlib.Path(__file__),OUT,ROOT/"tools/box/hyaex_check.py"])
    checker = runpy.run_path(str(ROOT/"tools/box/hyaex_check.py"))["cpu_schema"]()
    return dict(default_cuda_identical=True, base_restored_byte_identical=True, unchanged_protocol=True, checker=checker,
                L1_nonflat_and_sliding_unchanged=True, cuda_changed_lines=changed,
                generated_changes=[{"kind":tag,"base_lines":[i+1,j] if i<j else [],
                                    "insert_after_base_line":i if i==j else None,"lab_lines":[u+1,v]} for tag,i,j,u,v in
                    difflib.SequenceMatcher(None,base.splitlines(),source.splitlines(),autojunk=False).get_opcodes() if tag!="equal"],
                py_compile=True, bash_LF=True, dry_run=dry_run(source), polynomial=polynomial_check(),
                accounting_EST={"weight_bytes_per_tile":[0,0],"KV_bytes_per_tile":[16384,16384],
                    "page_index_bytes_per_tile":[256,256],"additional_global_or_shared_loop_bytes":0,
                    "extra_prepare_scratch_bytes_per_arm":65536*(64*4+4+4),
                    "pack2_per_thread":{
                        "0":{"ex2":32,"argument_FMA":32,"E4M3_pair_cvt":16,"pack_shift_OR":16,"math_source_ops":96},
                        "1":{"ex2":0,"bit_ops":64,"argument_FMA":32,"E4M3_pair_cvt":16,"pack_shift_OR":16,"math_source_ops":128},
                        "2":{"ex2":24,"offload_ops":8*12,"argument_FMA":32,"E4M3_pair_cvt":16,"pack_shift_OR":16,"math_source_ops":184},
                        "3":{"ex2":16,"offload_ops":16*12,"argument_FMA":32,"E4M3_pair_cvt":16,"pack_shift_OR":16,"math_source_ops":272}},
                    "per_emulated_element":"2 min/max + 3 add/sub RN + 5 FMA + shift + integer add; 12 vs 1 ex2; SASS pending",
                    "instruction_limit":"source arithmetic/inline-PTX operation counts, excludes common moves/addressing; compiler may combine shift/OR",
                    "overflow_repack":"same pack2 counts again only on the existing PB overflow branch"})


def sass_functions(text):
    result = {}
    for b in re.split(r"Function\s*:\s*",text)[1:]:
        name = b.splitlines()[0].strip()
        lines = [re.sub(r"/\*.*?\*/","",s).strip() for s in b.splitlines()
                 if re.match(r"\s*/\*[0-9a-f]+\*/",s)]
        result[name] = [s for s in lines if s]
    assert result
    return result


def resources(text):
    result = {}
    for b in re.split(r"ptxas info\s*: Compiling entry function ",text)[1:]:
        name = b.split("'",2)[1]
        if "attn_kernel" not in name: continue
        r = re.search(r"Used (\d+) registers",b)
        s = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",b)
        assert r and s, b
        key = re.search(r"attn_kernelILi\d+ELb[01]ELi\d+EE",name)[0]
        assert key not in result
        result[key] = dict(symbol=name,registers=int(r[1]),stack=int(s[1]),spill_stores=int(s[2]),spill_loads=int(s[3]))
    assert result
    return result


def linux_gate(base, source, report):
    import os
    import shutil
    import sysconfig
    import tempfile
    import torch
    from torch.utils.cpp_extension import include_paths
    work = pathlib.Path(tempfile.mkdtemp(prefix="hyaex-gate-")) # retained, no cleanup
    report["build_dir"] = str(work)
    for i,p in enumerate((pathlib.Path(__file__),OUT,ROOT/"tools/box/hyaex_check.py")):
        py_compile.compile(str(p),cfile=str(work/f"py{i}.pyc"),doraise=True)
    inc = ["-I"+p for p in include_paths(device_type="cuda")]+["-I"+sysconfig.get_path("include")]
    abi = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    def run(cmd, name):
        start = time.perf_counter()
        p = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (work/(name+".log")).write_bytes(p.stdout.encode())
        report.setdefault("commands",[]).append(dict(command=cmd,seconds_MEASURED=time.perf_counter()-start,rc=p.returncode))
        assert p.returncode == 0, (name,p.stdout[-12000:])
        return p.stdout
    nvcc = shutil.which("nvcc"); cobj = shutil.which("cuobjdump"); cxx = shutil.which(os.environ.get("CXX","c++"))
    assert nvcc and cobj and cxx, "nvcc / cuobjdump / C++20 compiler required"
    report["nvcc"] = run([nvcc,"--version"],"version")
    ns = namespace(source)
    report["builds"] = {}; assemblies = {}
    for k in (0,"twin",1,2,3):
        label = str(k); cuda = ns["_hyaex_src"](k)
        cpp = "#include <torch/extension.h>\n" + literal(base,"HYA_CPP_SRC")
        funcs = ["run"]
        if k not in (0,"twin"):
            cpp += RESOURCE_CPP; funcs.append("hyaex_resources")
        cpp += '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n' + "\n".join(f'm.def("{f}", &{f});' for f in funcs) + "\n}\n"
        cp, cu, co, obj = [work/(label+s) for s in (".cpp",".cu",".cpp.o",".cu.o")]
        cp.write_bytes(cpp.encode()); cu.write_bytes(cuda.encode())
        start = time.perf_counter()
        run([cxx,"-std=c++20","-O3","-fPIC",abi,"-DTORCH_EXTENSION_NAME=hyaex_compile_only",*inc,"-c",str(cp),"-o",str(co)],label+"_host")
        log = run([nvcc,"-std=c++20","-O3","-gencode=arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr","-Xptxas=-v,-warn-spills",
                   "-Xcompiler=-fPIC",abi,*inc,"-c",str(cu),"-o",str(obj)],label+"_nvcc")
        seconds = time.perf_counter()-start
        res = resources(log)
        ass = sass_functions(run([cobj,"--dump-sass",str(obj)],label+"_sass"))
        run([cobj,"--dump-resource-usage",str(obj)],label+"_resources")
        assemblies[label] = {key:ass[row["symbol"]] for key,row in res.items()}
        target = "attn_kernelILi8ELb1ELi0EE"
        assert target in res
        ops = assemblies[label][target]
        report["builds"][label] = dict(resources=res,compile_seconds_MEASURED=seconds,
            target_MUFU_EX2=sum("MUFU.EX2" in s for s in ops),target_instructions=len(ops),
            target_FFMA=sum(bool(re.search(r"\bFFMA\b",s)) for s in ops))
    ship = report["builds"]["0"]; target="attn_kernelILi8ELb1ELi0EE"
    for label,b in report["builds"].items():
        r = b["resources"][target]
        assert r["registers"] <= 80 and r["spill_stores"] == r["spill_loads"] == r["stack"] == 0,(label,r)
        assert not any(re.search(r"\b(?:LDL|STL)\b",s) for s in assemblies[label][target]),label
        if label != "0":
            for key in assemblies["0"]:
                if key == target and label != "twin": continue
                assert assemblies["0"][key] == assemblies[label][key],("non-target SASS differs",label,key)
    # Separate fixed ex2 (rescale/finalize) from pack2. Arm 1 removes pack2 only.
    fixed = report["builds"]["1"]["target_MUFU_EX2"]
    pack = ship["target_MUFU_EX2"]-fixed
    assert pack > 0 and pack % 4 == 0,(ship,fixed)
    for k,num in ((2,3),(3,2)):
        got = report["builds"][str(k)]["target_MUFU_EX2"]-fixed
        assert got*4 == pack*num,("pack2 MUFU reduction",k,pack,got)
    report["pack2_static_MUFU"] = {k:b["target_MUFU_EX2"]-fixed for k,b in report["builds"].items()}
    report["unchanged_other_SASS"] = True
    report["compile_cost_note"] = "host+nvcc cold compile only; actual load_inline/prepare wall time recorded by pod checker"
    report["status"] = "CPU_COMPILE_SASS_PASS"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-gate",action="store_true")
    ap.add_argument("--linux-gate",action="store_true")
    ap.add_argument("--json",type=pathlib.Path)
    args = ap.parse_args()
    base,source = build()
    if not (args.cpu_gate or args.linux_gate):
        OUT.write_bytes(source.encode("ascii"))
        print(f"generated {OUT.relative_to(ROOT)}; HYAEX default 0; {len(source)} bytes")
        return
    artifacts = (BASE,OUT,pathlib.Path(__file__),ROOT/"tools/box/hyaex_check.py",ROOT/"tools/local/hyaex_gate.sh")
    result = dict(status="FAIL",gpu_launched=False,output_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  artifact_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
    try:
        result["cpu"] = gates(base,source)
        result["status"] = "CPU_PASS"
        if args.linux_gate: linux_gate(base,source,result)
    except Exception as ex:
        result["status"] = "FAIL"; result["error"] = repr(ex)
        raise
    finally:
        if args.json: args.json.write_bytes((json.dumps(result,indent=2)+"\n").encode())
        print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()

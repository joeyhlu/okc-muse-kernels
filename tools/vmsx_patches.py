"""R-vmsx-1: exact VMS hi issue-slot diet; CPU-only generator and dry run.

Kill risks first: D1 broadcasts may survive as PRMT/MOV; D2 may retain zero
registers or grow allocation. SASS/resource gates decide, no speed claim.
D3 is below floor alone and is only a rider on d12. No protocol changes.
Bytes/unit unchanged: HBM weight tile 36,864 B; compute lane weight/scale
LDS 8*16+16=144 B, x fragments 16*16=256 B cold / 512 B hot; helper
red/red2 128 B cold / 256 B hot. No added load, store, wait or arrival.
D1 scale ops/lane/unit 16 PRMT+16 IMAD -> 8+8, HMUL2 stays 128.
D2 cold 32 / hot 64 MMA stay; 16 zeroed FP32 values -> first 4 MMA C=+0;
hot post-zero 8 values -> explicit +0 fold, same rounding/sign behavior.
D3 helper hot sum: 32 -> 28 scalar FADD, same eight float4 loads.
Dynamic SASS instruction savings and exactness remain GPU-gated.
"""
import ast
import contextlib
import hashlib
import io
import pathlib
import sys
import time
from types import SimpleNamespace as NS

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmsx_ab.py"
VARIANTS = {"d1": 1, "d2": 2, "d12": 3, "d123": 7}


def literal(s, name):
    n, = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return ast.literal_eval(n.value)


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:90], s.count(old), count)
    return s.replace(old, new)


def conditional(old, new, bit):
    return f"#if VMS_DIET & {bit}\n" + new + "#else\n" + old + "#endif\n"


def diet_zero(cu):
    """PP0-style projection of ONLY our directives, exact original bytes."""
    cu = sub(cu, "#ifndef VMS_DIET\n#define VMS_DIET 0\n#endif\n", "")
    lines, stack = [], []
    for line in cu.splitlines(True):
        t = line.strip()
        if t.startswith(("#if ", "#ifdef ", "#ifndef ")):
            own = "VMS_DIET" in t
            stack.append([own, False if own else True])
            if own: continue
        elif t == "#else" and stack[-1][0]:
            stack[-1][1] = not stack[-1][1]
            continue
        elif t == "#endif":
            own, _ = stack.pop()
            if own: continue
        if all(active for own, active in stack if own): lines.append(line)
    assert not stack
    return "".join(lines)


HELPERS = '''#if VMS_DIET & 1
template <bool H>
__device__ __forceinline__ uint32_t bmulh(uint32_t a, uint32_t b) {
const __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&b);
const __nv_bfloat162 scale = H ? __high2bfloat162(v) : __low2bfloat162(v);
const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), scale);
return *reinterpret_cast<const uint32_t*>(&r);
}
#endif
#if VMS_DIET & 2
__device__ __forceinline__ void mma0(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\\n"
: "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.f));
}
#endif
'''

PYTHON = '''VMSX_ON = 0
_VMSX = {"mods": {}, "flags": None, "errors": {}, "secs": {}, "calls": {}}
VMSX_VARIANTS = {"d1": 1, "d2": 2, "d12": 3, "d123": 7}
def _vmsx_build_one(mask, flags):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((VMS_CUDA_SRC + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"k3_vmshi_d{mask}_{tag}", cpp_sources=VMS_CPP_SRC, cuda_sources=VMS_CUDA_SRC,
                       functions=["run", "attrs"], extra_cuda_cflags=flags)
def _vmsx_prepare(st):
    _VMSX["errors"] = {}
    _VMSX["calls"].clear()
    vs = st["vms"]
    vs["diet"] = {}
    if not st["vms_ok"] or vs.get("mod_hi") is None:
        return
    base = list(vs["mod_hi"].attrs())
    if base[0] > 168 or base[1] or base[2] != 174464:
        raise RuntimeError(f"VMSX base resources {base}")
    def one(name, mask):
        t0 = time.time()
        try:
            if mask not in _VMSX["mods"]:
                flags = list(_VMSX["flags"]) + [f"-DVMS_DIET={mask}", "-Xptxas=-v"]
                _VMSX["mods"][mask] = _vmsx_build_one(mask, flags)
            md = _VMSX["mods"][mask]
            a = list(md.attrs())
            if a[0] > 168 or a[1] or a[2:] != base[2:]:
                raise RuntimeError(f"VMSX {name} resources {a}, base {base}")
            vs["diet"][mask] = md
            print(f"[vmsx] {name} resources {a}", flush=True)
        except Exception as ex:
            _VMSX["errors"][name] = repr(ex)
            print(f"[vmsx] {name} unavailable: {ex!r}", flush=True)
        _VMSX["secs"][name] = time.time() - t0
    import threading
    ths = [threading.Thread(target=one, args=(name, mask)) for name, mask in VMSX_VARIANTS.items()]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
'''


def build(base, report=False):
    assert "VMS_DIET" not in base and "VMSX_ON" not in base
    original = literal(base, "VMS_CUDA_SRC")
    cu = sub(original, "namespace vms {\n", "namespace vms {\n#ifndef VMS_DIET\n#define VMS_DIET 0\n#endif\n")
    cu = sub(cu, "__global__ void __launch_bounds__(NTL, 1) vms_kernel", HELPERS + "__global__ void __launch_bounds__(NTL, 1) vms_kernel")
    old = "const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);\n"
    cu = sub(cu, old, conditional(old, "const uint32_t s2 = scd(sw, (cc & 1) ? 0x4342u : 0x4140u, k16);\n", 1), 3)
    old = "const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);\n"
    new = "const uint32_t A0 = bmulh<false>(r0, s2), A1 = bmulh<true>(r1, s2), A2 = bmulh<false>(r2, s2), A3 = bmulh<true>(r3, s2);\n"
    cu = sub(cu, old, conditional(old, new, 1), 2)
    old = "#pragma unroll\nfor (int s = 0; s < 4; ++s) { acc[s][0] = 0.f; acc[s][1] = 0.f; acc[s][2] = 0.f; acc[s][3] = 0.f; }\n"
    cu = sub(cu, old, conditional(old, "", 2))
    old = "mma(acc[s & 1], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);\nmma(acc[2 + (s & 1)], A0, A1, A2, A3, b2[2 * s], b2[2 * s + 1]);\n"
    new = "if (cc == 0 && s < 2) {\n" + old.replace("mma(", "mma0(") + "} else {\n" + old + "}\n"
    cu = sub(cu, old, conditional(old, new, 2), 2)
    old = "mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);\n"
    args = "acc[s], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]"
    new = conditional("const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);\n", "const uint32_t A0 = bmulh<false>(r0, s2), A1 = bmulh<true>(r1, s2), A2 = bmulh<false>(r2, s2), A3 = bmulh<true>(r3, s2);\n", 1)
    new += conditional(f"mma({args});\n", f"if (cc == 0) mma0({args}); else mma({args});\n", 2)
    # Outer condition keeps the zero-mask source literally identical.
    cu = sub(cu, old, conditional(old, new, 3))
    old = "acc[2][0] = 0.f; acc[2][1] = 0.f; acc[2][2] = 0.f; acc[2][3] = 0.f; acc[3][0] = 0.f; acc[3][1] = 0.f; acc[3][2] = 0.f; acc[3][3] = 0.f;\n"
    cu = sub(cu, old, conditional(old, "", 2), 2)
    old = "".join(f"o.{c} = acc[0][{i}] + acc[1][{i}] + acc[2][{i}] + acc[3][{i}];\n" for i, c in enumerate("xyzw"))
    # Keep both +0 operations: -0 + +0 must remain +0, even if ptxas removes them.
    new = "#if VMS_T16\nif (two) {\n" + "".join(f"o.{c} = ((acc[0][{i}] + acc[1][{i}]) + 0.f) + 0.f;\n" for i, c in enumerate("xyzw")) + "} else\n#endif\n{\n" + old + "}\n"
    cu = sub(cu, old, conditional(old, new, 2))
    old = "#pragma unroll\nfor (int w = 0; w < NW; ++w) { const float4 u = S.red2[bq][w][lane]; sm2.x += u.x; sm2.y += u.y; sm2.z += u.z; sm2.w += u.w; }\n"
    new = "sm2 = S.red2[bq][0][lane];\n" + old.replace("w = 0", "w = 1")
    cu = sub(cu, old, conditional(old, new, 4))
    assert diet_zero(cu) == original, "mask 0 CUDA source identity"
    protocol = ("mbar_", "cta_sync", "__sync", "__threadfence", "S.gen", "p.ctr", "S.red[bq][warp]", "S.red2[bq][warp]")
    assert [l for l in cu.splitlines() if any(k in l for k in protocol)] == [l for l in original.splitlines() if any(k in l for k in protocol)], "protocol/publication changed"
    s = sub(base, original, cu)
    s = sub(s, "VMS_ENABLE = True\n", PYTHON + "VMS_ENABLE = True\n")
    anchor = '    tag_hi = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_hi)).encode()).hexdigest()[:10]\n'
    s = sub(s, anchor, '    _VMSX["flags"] = list(fl_hi)\n' + anchor)
    anchor = '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]\n'
    s = sub(s, anchor, anchor + '''    if m == 16:
        if type(VMSX_ON) is not int or VMSX_ON not in (0, 1, 2, 3, 7):
            raise RuntimeError(f"invalid VMSX_ON {VMSX_ON!r}")
        if VMSX_ON:
            if md is not vs.get("mod_hi") or VMSX_ON not in vs["diet"]:
                raise RuntimeError(f"unavailable VMSX arm {VMSX_ON}")
            md = vs["diet"][VMSX_ON]
        key = f"T16:{VMSX_ON}"
        _VMSX["calls"][key] = _VMSX["calls"].get(key, 0) + 1
''')
    s = sub(s, '        st["vms_ok"] = good\n', '        st["vms_ok"] = good\n        _vmsx_prepare(st)\n')
    ast.parse(s)
    assert len(s.encode()) <= 1_000_000
    if report:
        for i, line in enumerate(s.splitlines(), 1):
            if "VMS_DIET &" in line or line.startswith(("def _vmsx", "def _vms_call", "def _vms_prepare")):
                print(f"[vmsx] {OUT.name}:{i}: {line}")
    return s


def dry_run(base, patched):
    """Execute real _vms_prepare, all new build/bank paths and real _vms_call.

    Only GPU primitives/prepack/reference arithmetic are doubled. The real
    prepare builds its own strict dict; no hand-invented VMS state keys.
    """
    tree = ast.parse(patched)
    names = {"VMSX_ON", "_VMSX", "VMSX_VARIANTS", "_vmsx_build_one", "_vmsx_prepare", "_vms_prepare", "_vms_call", "VMS_SELF_TEST_CASES"}
    def named(n):
        return n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None
    ns = {"time": time, "hashlib": hashlib}
    exec(compile(ast.Module(body=[n for n in tree.body if named(n) in names], type_ignores=[]), "vmsx-dry", "exec"), ns)
    class Tensor:
        device = "cpu"
        def __init__(self, shape=(16, 4096)): self.shape = shape
        def __getitem__(self, k): return 0 if isinstance(k, int) and self.shape == (72,) else Tensor(self.shape[1:])
        def __setitem__(self, k, v): pass
        def __ge__(self, v): return self
        __le__ = __lt__ = __ge__
        def __and__(self, v): return self
        def __mul__(self, v): return self
        def __radd__(self, v): return self
        def all(self): return True
        def reshape(self, *s): return Tensor(s)
        def view(self, *s): return self
        def to(self, *a): return self
        def contiguous(self): return self
        clone = zero_ = float = contiguous
    seen, builds, failed, badattrs = [], [], set(), {}
    class Module:
        def __init__(self, mask): self.mask = mask
        def attrs(self): return badattrs.get(self.mask, [153, 0, 174464, 132, 2])
        def run(self, *a): seen.append((self.mask, a[14]))
    def inputs(bs, *a): return {k: Tensor((bs, 4096)) for k in ("hidden_states", "topk_ids", "topk_weights")}
    # Execute base's actual flags expressions, not copied approximations.
    flags_ns = {"G": 132}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vms_build")
    for n in fn.body:
        if isinstance(n, ast.Assign) and named(n) in ("flags", "fl_hi"):
            exec(compile(ast.Module(body=[n], type_ignores=[]), "flags", "exec"), flags_ns)
    def load_inline(**kw):
        flags = kw["extra_cuda_cflags"]
        mask = int(next(f.split("=")[1] for f in flags if f.startswith("-DVMS_DIET=")))
        assert flags == flags_ns["fl_hi"] + [f"-DVMS_DIET={mask}", "-Xptxas=-v"]
        assert kw["name"].startswith(f"k3_vmshi_d{mask}_") and kw["functions"] == ["run", "attrs"]
        assert kw["cuda_sources"] == literal(patched, "VMS_CUDA_SRC")
        assert kw["cpp_sources"] == literal(base, "VMS_CPP_SRC")
        builds.append(mask)
        if mask in failed: raise RuntimeError("injected build failure")
        return Module(mask)
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **kw):
        return NS(load_inline=load_inline) if name == "torch.utils.cpp_extension" else real_import(name, *a, **kw)
    # Function globals use this per-exec builtins map; no sys.modules mutation.
    local_builtins = dict(vars(builtins), __import__=fake_import)
    ns["__builtins__"] = local_builtins
    build_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vmsx_build_one")
    exec(compile(ast.Module(body=[build_node], type_ignores=[]), "build-one", "exec"), ns)
    ns.update(torch=NS(int32=1, float32=2, bfloat16=3, uint8=4, int64=5,
              empty=lambda s, **kw: Tensor(s), zeros=lambda s, **kw: Tensor(s),
              full_like=lambda t, v: t, where=lambda c, a, b: a, isfinite=lambda t: t,
              cuda=NS(synchronize=lambda: None, get_device_properties=lambda d: NS(multi_processor_count=132))),
              _VMS={"mod": Module(-1), "mod_hi": Module(0), "G": 132}, VMS_ENABLE=True, VMS_HI_ON=True,
              VMS_MAXT=16, VMS_UB=36864, K2A_E_LOC=80, K2A_D_EXPERT=4096, K2A_D_HIDDEN=4096,
              K2A_POST_NORM_EPS=1e-6, _pt1_wait=lambda n: None,
              VMS_CUDA_SRC=literal(patched, "VMS_CUDA_SRC"), VMS_CPP_SRC=literal(base, "VMS_CPP_SRC"),
              _vms_prepack_expert=lambda *a: Tensor(), _k2a_self_test_inputs=inputs, _k2a36_synth_inputs=inputs,
              _k2a_ref_fp32=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0, 0),
              _v11_k2a33_call=lambda *a: Tensor(), _k2a33_reset=lambda s: None)
    ns["_VMSX"]["flags"] = flags_ns["fl_hi"]
    w = {k: Tensor((80, 8192, 2048)) for k in ("w13", "w2", "w13_sf", "w2_sf", "w13_gs", "w2_gs", "post_norm_w")}
    state = {"err": Tensor()}
    with contextlib.redirect_stdout(io.StringIO()): ns["_vms_prepare"](state, w, "cpu")
    assert state["vms_ok"] and set(state["vms"]["diet"]) == {1, 2, 3, 7} and sorted(builds) == [1, 2, 3, 7]
    for mask in (0, 1, 2, 3, 7):
        ns["VMSX_ON"] = mask
        for bs in (1, 8, 12, 16):
            ns["_vms_call"](state, Tensor((bs, 4096)), Tensor(), Tensor())
            assert seen[-1] == ((mask if bs == 16 else 0 if bs > 8 else -1), bs)
    for bad in (-1, 4, True):
        ns["VMSX_ON"] = bad
        try: ns["_vms_call"](state, Tensor(), Tensor(), Tensor())
        except RuntimeError: pass
        else: raise AssertionError("invalid arm accepted")
    ns["VMSX_ON"] = 0
    with contextlib.redirect_stdout(io.StringIO()): ns["_vmsx_prepare"](state)
    assert len(builds) == 4, "cache miss on repeated prepare"
    failed.add(2); ns["_VMSX"]["mods"].pop(2)
    with contextlib.redirect_stdout(io.StringIO()): ns["_vmsx_prepare"](state)
    assert set(state["vms"]["diet"]) == {1, 3, 7} and "d2" in ns["_VMSX"]["errors"]
    ns["VMSX_ON"] = 2
    try: ns["_vms_call"](state, Tensor(), Tensor(), Tensor())
    except RuntimeError: pass
    else: raise AssertionError("missing build silently fell back")
    failed.clear()
    for index, value in ((0, 169), (1, 4), (2, 174465)):
        a = [153, 0, 174464, 132, 2]; a[index] = value; badattrs[1] = a
        with contextlib.redirect_stdout(io.StringIO()): ns["_vmsx_prepare"](state)
        assert 1 not in state["vms"]["diet"] and "d1" in ns["_VMSX"]["errors"]
    badattrs.clear()
    state["vms_ok"] = False; ns["_vmsx_prepare"](state)
    assert not state["vms"]["diet"]
    state["vms_ok"] = True; state["vms"]["mod_hi"] = None; ns["_vmsx_prepare"](state)
    assert not state["vms"]["diet"]
    ns["VMS_ENABLE"] = False; ns["_vms_prepare"](state, w, "cpu"); assert not state["vms_ok"]
    print("[vmsx] CPU dry run PASS: real prepare/state, flags, four builds, cache, arms, independent failure, no silent fallback")


def exact_cpu_gate():
    # Every admissible pair: unsigned packed multiply/add is carry-free.
    for a in range(8, 127):
        for b in range(8, 127):
            s2 = (a | b << 16) * 16 + 0x5C005C00
            assert (s2 & 65535) * 0x10001 == (a * 0x10001) * 16 + 0x5C005C00
            assert (s2 >> 16) * 0x10001 == (b * 0x10001) * 16 + 0x5C005C00
    # First-use coverage: all four cold/hot accumulator chains initialized once.
    for hot in (False, True):
        seen = set()
        for cc in range(8):
            for s in range(4):
                chains = (s & 1, 2 + (s & 1)) if hot else (s,)
                for chain in chains:
                    zero = cc == 0 and (s < 2 if hot else True)
                    assert zero == (chain not in seen)
                    seen.add(chain)
        assert seen == set(range(4))
    print("[vmsx] CPU arithmetic PASS: 14,161 scale pairs; cold/hot first-MMA coverage")


def main():
    assert len(sys.argv) == 1
    base = BASE.read_bytes().decode().replace("\r\n", "\n")
    s = build(base, report=True)
    exact_cpu_gate(); dry_run(base, s)
    OUT.write_bytes(s.encode())
    print(f"[vmsx] mask 0 CUDA source identity PASS; AST PASS; {len(s.encode()):,} B")


if __name__ == "__main__":
    main()

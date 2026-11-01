"""R-vmsx-1 / R-vmsy-1: exact VMS hi diet; CPU generator and dry run.

Kill risks first: D1 broadcasts may survive as PRMT/MOV; D2 may retain zero
registers or grow allocation. SASS/resource gates decide, no speed claim.
D2/D3 code is retained but excluded from the lab bank (D2 regressed).
D4a (bit 3) / D4b (bit 4) hoist compute / issuer-helper integer arithmetic
on D1 (bit 0). Kill risks: ptxas rematerialization and register pressure.
No protocol changes; only d1 / d14a / d14ab are built for the lab.
Bytes/unit unchanged: HBM weight tile 36,864 B; compute lane weight/scale
LDS 8*16+16=144 B, x fragments 16*16=256 B cold / 512 B hot; helper
red/red2 128 B cold / 256 B hot. No added load, store, wait or arrival.
D1 scale ops/lane/unit 16 PRMT+16 IMAD -> 8+8, HMUL2 stays 128.
D2 cold 32 / hot 64 MMA stay; 16 zeroed FP32 values -> first 4 MMA C=+0;
hot post-zero 8 values -> explicit +0 fold, same rounding/sign behavior.
D3 helper hot sum: 32 -> 28 scalar FADD, same eight float4 loads.
D4 SASS/resource acceptance and GPU timing remain unmeasured.
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
VARIANTS = {"d1": 1, "d14a": 9, "d14ab": 25}


def literal(s, name):
    n, = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    return ast.literal_eval(n.value)


def sub(s, old, new, count=1):
    assert s.count(old) == count, (old[:90], s.count(old), count)
    return s.replace(old, new)


def conditional(old, new, bit):
    return f"#if VMS_DIET & {bit}\n" + new + "#else\n" + old + "#endif\n"


def diet_project(cu, mask=0):
    """Project ONLY our mask directives; preserve all native directives."""
    cu = sub(cu, "#ifndef VMS_DIET\n#define VMS_DIET 0\n#endif\n", "")
    lines, stack = [], []
    for line in cu.splitlines(True):
        t = line.strip()
        if t.startswith(("#if ", "#ifdef ", "#ifndef ")):
            own = "VMS_DIET" in t
            stack.append([own, bool(mask & int(t.rsplit("&", 1)[1])) if own else True])
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


def diet_zero(cu):
    return diet_project(cu)


def diet_hoists(cu):
    """D4a/b: arithmetic only. Kill risks: rematerialization; extra live regs.

    Audit C lines refer to k3_hh_5_sk2_pz.py. Conservative EST steady-unit
    targets vs D1 (both hot/cold): D4a 7 slots/compute lane (k16 2, base 3,
    shared q mask 2); D4b 5 on warp 0 (policy 4, UB 1), 7/helper (base 3,
    q mask 3 + reuse 1). Aggregate on SMSP0: 14 / 26 slots. R-vmsx-1's
    MEASURED ~0.042 us/slot gives EST 0.588 / 1.092 us, not a SASS result.
    Bytes are identical to D1: 36,864 HBM B/unit, no added LDS/STS/LDG/STG.
    Phase/lane/sl-par are deliberately left alone: they feed waits, arrival
    predicates and ring counters. No zero-add or ordered reduction changes.
    An opaque register identity discourages frontend sinking; ptxas may
    rematerialize it. Accept ONLY fewer hot AND cold slots, <=168 regs/no spill.
    Native createpolicy is hoisted, never replaced by an opaque literal.
    """
    anchor = "__device__ __forceinline__ void issue("
    helpers = '''#if VMS_DIET & 24
__device__ __forceinline__ uint32_t d4_hold(uint32_t x) {
asm volatile("mov.b32 %0, %1;" : "=r"(x) : "r"(x)); return x;
}
__device__ __forceinline__ int d4_bq(int q) {
static_assert(NRB == 4, "D4 shared mask requires the audited NRB=4");
uint32_t b; asm("bfe.u32 %0, %1, 0, 2;" : "=r"(b) : "r"(q)); return (int)b;
}
#endif
#if VMS_DIET & 16
__device__ __forceinline__ uint64_t d4_policy() {
uint64_t pol;
asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(pol));
return pol;
}
#endif
'''
    cu = sub(cu, anchor, helpers + anchor)
    # Default arguments preserve the dormant VMS_FX pre-route issue site.
    old = "__device__ __forceinline__ void issue(const unsigned char* src, unsigned char* dst, uint64_t* bar, int tid) {\n"
    new = old.replace("int tid)", "int tid, uint64_t pol = d4_policy(), uint32_t UB = vms::UB)")
    cu = sub(cu, old, conditional(old, new, 16))
    policy = 'uint64_t pol;\nasm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\\n" : "=l"(pol));\n'
    cu = sub(cu, "#if __CUDA_ARCH__ >= 900\n" + policy,
             "#if __CUDA_ARCH__ >= 900\n" + conditional(policy, "", 16))
    shadow = 'Smem& S = *reinterpret_cast<Smem*>(__cvta_shared_to_generic(d4_hold(su32(smem))));\n'
    # Preserve every original use of S, including barrier operands. Only its
    # identical shared base is made loop-invariant at the branch entry.
    anchor = "if (warp < NW) {\n"
    cu = sub(cu, anchor, anchor + conditional("", shadow, 8) + conditional("", "const uint64_t d4_pol = d4_policy();\nconst uint32_t d4_ub = d4_hold(UB);\n", 16))
    anchor = "} else {\nconst int g = lane >> 2, t = lane & 3;\nfloat* ssc"
    cu = sub(cu, anchor, "} else {\n" + conditional("", shadow, 16) + "const int g = lane >> 2, t = lane & 3;\nfloat* ssc")
    old = "const uint32_t k16 = 16u + (uint32_t)(p.m >> 30);\n"
    cu = sub(cu, old, conditional(old, "const uint32_t k16 = d4_hold(16u + (uint32_t)(p.m >> 30));\n", 8))
    # All normal-path issue calls are inside the compute branch. The original
    # mbarrier/copy asm, arrival count and byte count remain literally intact.
    for old, count in (("      issue(unit_src(p, S, q, nA, a0, b0), ring + q * UB, &S.full[q], tid);\n", 1),
                       ("        issue(unit_src(p, S, q + NS, nA, a0, b0), ring + sl * UB, &S.full[sl], tid);\n", 3)):
        cu = sub(cu, old, conditional(old, old.replace(", tid);", ", tid, d4_pol, d4_ub);"), 16), count)
    # Keep the wait/phase expression and all predicates unchanged. Opaque bfe
    # makes the same q&3 result common to the barrier and red-address scaling.
    start = cu.index("} else {\n" + conditional("", shadow, 16))
    compute, helper = cu[:start], cu[start:]
    old = "const int bq = q & (NRB - 1);\n"
    compute = sub(compute, old, conditional(old, "const int bq = d4_bq(q);\n", 8), 2)
    helper = sub(helper, old, conditional(old, "const int bq = d4_bq(q);\n", 16))
    return compute + helper


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
VMSX_VARIANTS = {"d1": 1, "d14a": 9, "d14ab": 25}
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
    cu = diet_hoists(cu)
    assert diet_zero(cu) == original, "mask 0 CUDA source identity"
    protocol = ("mbar_", "cta_sync", "__sync", "__threadfence", "S.gen", "p.ctr", "S.red[bq][warp]", "S.red2[bq][warp]")
    assert [l for l in cu.splitlines() if any(k in l for k in protocol)] == [l for l in original.splitlines() if any(k in l for k in protocol)], "protocol/publication changed"
    d1 = diet_project(cu, 1)
    for mask in VARIANTS.values():
        active = diet_project(cu, mask)
        for needles in (protocol, ("mma(", "bmul", "scd(", "acc[", "sm.x", "sm2", "__expf", "st_keep_bf16", "__shfl", "__all_sync")):
            assert [l for l in active.splitlines() if any(k in l for k in needles)] == [l for l in d1.splitlines() if any(k in l for k in needles)], (mask, "D4 FP/protocol identity")
    s = sub(base, original, cu)
    s = sub(s, "VMS_ENABLE = True\n", PYTHON + "VMS_ENABLE = True\n")
    anchor = '    tag_hi = hashlib.sha1((VMS_CUDA_SRC + " ".join(fl_hi)).encode()).hexdigest()[:10]\n'
    s = sub(s, anchor, '    _VMSX["flags"] = list(fl_hi)\n' + anchor)
    anchor = '    md = vs["mod_hi"] if (VMS_HI_ON and m > 8 and vs.get("mod_hi") is not None) else vs["mod"]\n'
    s = sub(s, anchor, anchor + '''    if m == 16:
        if type(VMSX_ON) is not int or VMSX_ON not in (0, *VMSX_VARIANTS.values()):
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
    masks = set(VARIANTS.values())
    assert ns["VMSX_VARIANTS"] == VARIANTS
    assert state["vms_ok"] and set(state["vms"]["diet"]) == masks and sorted(builds) == sorted(masks)
    for mask in (0, *VARIANTS.values()):
        ns["VMSX_ON"] = mask
        for bs in (1, 8, 12, 16):
            ns["_vms_call"](state, Tensor((bs, 4096)), Tensor(), Tensor())
            assert seen[-1] == ((mask if bs == 16 else 0 if bs > 8 else -1), bs)
    for bad in (-1, 2, 3, 4, 7, 8, 16, True):
        ns["VMSX_ON"] = bad
        try: ns["_vms_call"](state, Tensor(), Tensor(), Tensor())
        except RuntimeError: pass
        else: raise AssertionError("invalid arm accepted")
    ns["VMSX_ON"] = 0
    with contextlib.redirect_stdout(io.StringIO()): ns["_vmsx_prepare"](state)
    assert len(builds) == len(masks), "cache miss on repeated prepare"
    failed.add(9); ns["_VMSX"]["mods"].pop(9)
    with contextlib.redirect_stdout(io.StringIO()): ns["_vmsx_prepare"](state)
    assert set(state["vms"]["diet"]) == masks - {9} and "d14a" in ns["_VMSX"]["errors"]
    ns["VMSX_ON"] = 9
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
    print("[vmsx] CPU dry run PASS: real prepare/state, flags, three builds, cache, arms, independent failure, no silent fallback")


def exact_cpu_gate():
    # D4 preserves addresses for every lane/warp/ring slot, including cluster
    # shared-address high bits. Native cvta round-trips the unchanged u32 base;
    # the register identity performs no arithmetic. No protocol is redesigned:
    # all G=132 CTAs keep compute tid 0..255 and helper tid 256..287; compute
    # all lanes store red/red2, lane 0 of each warp arrives pfull; helper all
    # lanes reduce/store and its lane 0 returns pempty/publishes job counts.
    # The post-loop CTA/grid wait, cleanup, reset and combine suffix is untouched.
    for q in range(64 * (512 + 256) + 1):
        assert (q & 3) == ((q >> 0) & ((1 << 2) - 1))
    for smem in (0x400, 0x1000400, 0x7000400):
        for bq in range(4):
            for tid in range(288):
                warp, lane = tid >> 5, tid & 31
                w = warp if warp < 8 else 0
                for red_offset in (0x960, 0x4960):
                    assert smem + red_offset + (bq * 256 + w * 32 + lane) * 16 == smem + red_offset + bq * 4096 + w * 512 + lane * 16
                assert smem + 16 + bq * 8 == smem + 16 + (bq << 3)
    for m in range(1, 17):
        assert 16 + (m >> 30) == 16
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
    print("[vmsx] CPU arithmetic PASS: D4 mask/address/hoist identities; 14,161 scale pairs; cold/hot first-MMA coverage")


def ship_build(base, report=False):
    """D1 only, always enabled in the existing hi build; no host changes.

    VMS_T16 is the EXISTING base/hi compile switch, not a new diet knob.
    Each replacement has an exact inverse and a flag-off source projection.
    No bytes, MMA, stores, counters, waits, cleanup or combine loops change.
    """
    assert "VMS_DIET" not in base and "VMSX_ON" not in base
    original = literal(base, "VMS_CUDA_SRC")
    edits = []
    helper = HELPERS.split("#if VMS_DIET & 1\n", 1)[1].split("#endif\n", 1)[0]
    anchor = "__global__ void __launch_bounds__(NTL, 1) vms_kernel"
    edits.append((anchor, "#if VMS_T16\n" + helper + "#endif\n" + anchor, 1))
    scale = "const uint32_t sg = scd(sw, sel, k16), sg8 = scd(sw, sel + 0x0101u, k16);\n"
    merged = "const uint32_t s2 = scd(sw, (cc & 1) ? 0x4342u : 0x4140u, k16);\n"
    old = "const uint32_t A0 = bmul(r0, sg), A1 = bmul(r1, sg8), A2 = bmul(r2, sg), A3 = bmul(r3, sg8);\n"
    new = "const uint32_t A0 = bmulh<false>(r0, s2), A1 = bmulh<true>(r1, s2), A2 = bmulh<false>(r2, s2), A3 = bmulh<true>(r3, s2);\n"
    cold = "mma(acc[s], bmul(r0, sg), bmul(r1, sg8), bmul(r2, sg), bmul(r3, sg8), bb[2 * s], bb[2 * s + 1]);\n"
    cold_new = new + "mma(acc[s], A0, A1, A2, A3, bb[2 * s], bb[2 * s + 1]);\n"
    for before, after, count in ((scale, merged, 3), (old, new, 2), (cold, cold_new, 1)):
        edits.append((before, "#if VMS_T16\n" + after + "#else\n" + before + "#endif\n", count))
    cu = original
    for before, after, count in edits:
        cu = sub(cu, before, after, count)
    s = sub(base, original, cu)
    ast.parse(s)
    reverse = cu
    for before, after, count in reversed(edits):
        reverse = sub(reverse, after, before, count)
    assert reverse == original and sub(s, cu, reverse) == base, "ship reverse-edit identity"
    # Evaluate only inserted guards with VMS_T16=0. This leaves EVERY original
    # directive/line intact, hence the low build's preprocessor input unchanged.
    low = cu
    for before, after, count in reversed(edits):
        projected = after.split("#else\n", 1)[1].rsplit("#endif\n", 1)[0] if "#else\n" in after else anchor
        low = sub(low, after, projected, count)
    assert low == original, "flag-off source identity"
    # Python state/dispatch/build paths and all other source literals unchanged.
    bt, st = ast.parse(base), ast.parse(s)
    for tree in (bt, st):
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VMS_CUDA_SRC" for t in node.targets):
                node.value = ast.Constant("CUDA checked separately")
    assert ast.dump(bt) == ast.dump(st), "ship host plumbing changed"
    assert len(s.encode()) < 1_000_000
    if report:
        import difflib
        for group in difflib.SequenceMatcher(a=base.splitlines(), b=s.splitlines(), autojunk=False).get_opcodes():
            tag, a0, a1, b0, b1 = group
            if tag != "equal":
                where = f"before base:{a0+1}" if a0 == a1 else f"base:{a0+1}-{a1}"
                print(f"[vmsx-ship] D1 {where} -> ship:{b0+1}-{b1}")
        print("[vmsx-ship] AST, reverse-edit, low flag-off source identity PASS")
    return s


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ship", action="store_true")
    ap.add_argument("--base", type=pathlib.Path, default=BASE)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    base = args.base.read_bytes().decode().replace("\r\n", "\n")
    if args.ship:
        s = ship_build(base, report=True)
        exact_cpu_gate()
        # Execute real base/ship prepare + call paths, without importing a layer.
        import runpy
        runpy.run_path(str(ROOT / "tools/box/vmsx_ship_run.py"))["prepare_dry_run"](base, s)
        target = args.out or ROOT / "kernels/lab/k3_vmsx_ship.py"
        target.write_bytes(s.encode())
        print(f"[vmsx-ship] {target}: {len(s.encode()):,} B; headroom {1_000_000-len(s.encode()):,} B")
        return
    s = build(base, report=True)
    exact_cpu_gate(); dry_run(base, s)
    (args.out or OUT).write_bytes(s.encode())
    print(f"[vmsx] mask 0 CUDA source identity PASS; AST PASS; {len(s.encode()):,} B")


if __name__ == "__main__":
    main()

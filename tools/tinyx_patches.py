"""TINYX: arithmetic-only TINY/TINY2 port of VMSX D1 and VMSD nulls.

CPU: python -B -I tools/tinyx_patches.py
Compile only (no CUDA context): cudaenv.sh python -B -I tools/tinyx_patches.py --local-compile
All artifacts are retained under box_out/tinyx. Never runs a GPU or a pod.
Kill risks, before measurement: TINY2 can hide issue savings under its stream;
extra live registers / probe dispatch can change scheduling. D1 stays HOLD until
both real W13/W2 loops fold and every compiled function is spill-free.
"""
import ast
import builtins
import difflib
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace as NS

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz_rt.py"
OUT = ROOT / "kernels/lab/k3_tinyx_ab.py"
ART = ROOT / "box_out/tinyx"


def node(source, name):
    return next(n for n in ast.parse(source).body if
                getattr(n, "name", None) == name or isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)


def literal(source, name):
    return ast.literal_eval(node(source, name).value)


def _tinyx_sub(s, old, new):
    if s.count(old) != 1:
        raise ValueError(f"TINYX anchor count {s.count(old)}: {old[:100]!r}")
    return s.replace(old, new)


_TINYX_HELP_D1 = r'''template <bool H>
__device__ __forceinline__ uint32_t tinyx_bmulh(uint32_t a, uint32_t b) {
const __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&b);
const __nv_bfloat162 scale = H ? __high2bfloat162(v) : __low2bfloat162(v);
const __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), scale);
return *reinterpret_cast<const uint32_t*>(&r);
}
'''
_TINYX_HELP_PROBE = r'''template<int C> struct TinyxClass { static constexpr int value = C; };
__device__ __forceinline__ void tinyx_sink(uint32_t& z, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
asm volatile("{ .reg .b32 t,u; lop3.b32 t,%1,%2,%3,0x96; lop3.b32 u,%4,%5,%6,0x96; lop3.b32 %0,%0,t,u,0x96; }"
: "+r"(z) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
'''


def _tinyx_source(source, kind):
    """Keep ctl byte-identical; probes share ONE binary and one unit dispatch.

    Existing Params.zero transports mask[7:0], class[9:8], zero salts[31:10].
    The former scale multiplier is preserved with zero >> 10. No ABI growth.
    Logical warp 3/7 = physical warp 4/8, sharing helper physical warp 0.
    """
    if kind == "ctl":
        return source
    if kind not in ("D1", "probe"):
        raise ValueError(kind)
    start = "#pragma unroll\nfor (int c = 0; c < NCH; ++c) {\n"
    end = "#pragma unroll\nfor (int k = 0; k < 4; ++k) acc[k] += acc2[k] + acc3[k] + acc4[k];\n"
    fn = source.index("__device__ void skinny_ring_ws(")
    i = source.index(start, fn)
    j = source.index(end, i)
    old, body = source[i:j], source[i:j]
    assert "__device__" not in body and "load_x_regs" not in body
    assert body.count("dq4(ww[st], r0, r1, r2, r3);") == 1
    helper = _TINYX_HELP_D1 if kind == "D1" else _TINYX_HELP_PROBE
    if kind == "D1":
        body = _tinyx_sub(body,
            "const uint32_t sg = SCDK(swd, (c & 1) * 2, k16), sg8 = SCDK(swd, (c & 1) * 2 + 1, k16);",
            "const uint32_t s2 = __byte_perm(swd, 0u, (c & 1) ? 0x4342u : 0x4140u) * k16 + 0x5C005C00u;")
        for a, s, h in (("r0", "sg", "false"), ("r1", "sg8", "true"), ("r2", "sg", "false"), ("r3", "sg8", "true")):
            body = _tinyx_sub(body, f"bmul2({a}, {s})", f"tinyx_bmulh<{h}>({a}, s2)")
    else:
        body = _tinyx_sub(body, "dq4(ww[st], r0, r1, r2, r3);", """if constexpr (tinyx_mode == 1) {
r0 = ww[st] & 0x81C081C0u;
r1 = r0 ^ (p.zero >> 10);
r2 = r0 ^ (p.zero >> 20);
r3 = r0;
} else { dq4(ww[st], r0, r1, r2, r3); }""")
        mma = "mma_bf16(st == 0 ? acc : st == 1 ? acc2 : st == 2 ? acc3 : acc4, bmul2(r0, sg), bmul2(r1, sg8), bmul2(r2, sg), bmul2(r3, sg8), b[2 * st], b[2 * st + 1]);"
        sink = "tinyx_sink(tinyx_z, bmul2" + mma.split(", bmul2", 1)[1]
        body = _tinyx_sub(body, mma, "if constexpr (tinyx_mode == 2) " + sink + "\nelse " + mma)
        body = """uint32_t tinyx_z = 0;
auto tinyx_unit = [&](auto tag) {
constexpr int tinyx_mode = decltype(tag)::value;
""" + body + """};
if (tinyx_class == 1) tinyx_unit(TinyxClass<1>{});
else if (tinyx_class == 2) tinyx_unit(TinyxClass<2>{});
else tinyx_unit(TinyxClass<0>{});
if (tinyx_class == 2) {
const float z = __uint_as_float(tinyx_z & (p.zero >> 10));
#pragma unroll
for (int k = 0; k < 4; ++k) acc[k] = acc2[k] = acc3[k] = acc4[k] = z;
}
"""
    cu = _tinyx_sub(source, old, body)
    cu = _tinyx_sub(cu, "template <bool PB>\n__device__ void skinny_ring_ws", helper + "template <bool PB>\n__device__ void skinny_ring_ws")
    if kind == "probe":
        cu = _tinyx_sub(cu, "const uint32_t k16 = 16u + p.zero;", "const uint32_t k16 = 16u + (p.zero >> 10);\nconst int tinyx_class = ((p.zero >> warp) & 1) ? ((p.zero >> 8) & 3) : 0;")
        cu = _tinyx_sub(cu, "static int g_okc_pdl = 0;", "static int g_okc_pdl = 0;\nstatic uint32_t g_tinyx_flags = 0;\nvoid okc_set_tinyx(int64_t flags) { g_tinyx_flags = (uint32_t)flags; }")
        cu = _tinyx_sub(cu, "p.zero = 0;", "p.zero = g_tinyx_flags;")
    return cu


TINYX_ON = "ctl"
TINYX_ARMS = {"ctl": ("ctl", 0), "D1": ("D1", 0), "DQ": ("probe", 0x1ff),
              "S0": ("probe", 0x188), "MM": ("probe", 0x2ff), "pctl": ("probe", 0)}
_TINYX = {"mods": {}, "errors": {}, "secs": {}, "calls": {}, "accepted": set()}


def _tinyx_build_one(k, kind, var):
    from torch.utils.cpp_extension import load_inline
    cu = _tinyx_source(_HYBLO_CUDA, kind)
    cpp = _HYB_MOE_CPP
    functions = ["okc_run", "okc_set_pdl"]
    if kind == "probe":
        cpp += "\nvoid okc_set_tinyx(int64_t flags);\n"
        functions.append("okc_set_tinyx")
    flags = list(_HYBLO_FLAGS) + list(var) + ["-Xptxas=-v"]
    tag = hashlib.sha1((cpp + cu + " ".join(flags)).encode()).hexdigest()[:12]
    return load_inline(name=f"tinyx_{k}_{kind}_{tag}", cpp_sources=cpp, cuda_sources=cu,
                       functions=functions, extra_cuda_cflags=flags, verbose=False)


def _tinyx_prepare():
    _TINYX["errors"].clear()
    _TINYX["calls"].clear()
    if not HYBLO_ENABLE:
        return
    # Same real global bank and scratch as _hyblo_run. Never synthesize layer keys.
    if _HYBLO["w"] is None or _HYBLO["sc"] is None:
        raise RuntimeError("TINYX: ship HYBLO prepare unavailable")
    for k, _, var in _HYBLO_KEYS:
        if not _HYBLO["ok"].get(k) or k not in _HYBLO["mods"]:
            raise RuntimeError(f"TINYX: ship {k} unavailable")
        for kind in ("D1", "probe"):
            key, t0 = (k, kind), time.time()
            try:
                if key not in _TINYX["mods"]:
                    _TINYX["mods"][key] = _tinyx_build_one(k, kind, var)
            except Exception as ex:
                _TINYX["errors"][key] = repr(ex)
            _TINYX["secs"][key] = time.time() - t0


def _tinyx_select(k):
    if TINYX_ON not in TINYX_ARMS:
        raise RuntimeError(f"invalid TINYX arm {TINYX_ON!r}")
    kind, flags = TINYX_ARMS[TINYX_ON]
    if kind == "ctl":
        mod = _HYBLO["mods"][k]
    else:
        if (k, kind) not in _TINYX["accepted"]:
            raise RuntimeError(f"TINYX {k}/{kind}: compile/SASS gate not accepted")
        if (k, kind) in _TINYX["errors"] or (k, kind) not in _TINYX["mods"]:
            raise RuntimeError(f"TINYX {k}/{kind}: missing build")
        mod = _TINYX["mods"][(k, kind)]
        if kind == "probe":
            mod.okc_set_tinyx(flags)
    key = f"{k}:{TINYX_ON}"
    _TINYX["calls"][key] = _TINYX["calls"].get(key, 0) + 1
    return mod, (k, kind)


def build(base):
    assert "TINYX_ON" not in base
    runtime = "\n# TINYX LAB: ctl is the unchanged ship binary.\n"
    for name in ("_TINYX_HELP_D1", "_TINYX_HELP_PROBE", "TINYX_ON", "TINYX_ARMS"):
        runtime += f"{name} = {globals()[name]!r}\n"
    runtime += '_TINYX = {"mods": {}, "errors": {}, "secs": {}, "calls": {}, "accepted": set()}\n'
    for fn in (_tinyx_sub, _tinyx_source, _tinyx_build_one, _tinyx_prepare, _tinyx_select):
        runtime += inspect.getsource(fn) + "\n"
    s = _tinyx_sub(base, "HYBLO_ENABLE = True\n", runtime + "HYBLO_ENABLE = True\n")
    run = ast.get_source_segment(s, node(s, "_hyblo_run"))
    new = _tinyx_sub(run, 'mod = _HYBLO["mods"][k]', 'mod, pkey = _tinyx_select(k)')
    new = new.replace('_HYBLO["pdl"].get(k)', '_HYBLO["pdl"].get(pkey)').replace('_HYBLO["pdl"][k]', '_HYBLO["pdl"][pkey]')
    s = _tinyx_sub(s, run, new)
    anchor = next(l for l in s.splitlines(True) if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    s = _tinyx_sub(s, anchor, anchor + "        _tinyx_prepare()\n")
    ast.parse(s)
    assert literal(s, "_HYBLO_CUDA") == literal(base, "_HYBLO_CUDA")
    for kind in ("ctl", "D1", "probe"):
        cu = _tinyx_source(literal(base, "_HYBLO_CUDA"), kind)
        # All protocol, helper, cleanup, combine and stores remain literally equal.
        old = literal(base, "_HYBLO_CUDA")
        a, b = "const int bb = q % NRB;", "#include <torch/extension.h>"
        assert cu[cu.index(a):cu.index(b)] == old[old.index(a):old.index(b)]
        protocol = r"^.*(?:mbar_|CTA_SYNC|WS_JOIN|__syncwarp|__threadfence|ws_issue\(|load_x_regs<).*$"
        assert re.findall(protocol, cu, re.M) == re.findall(protocol, old, re.M)
        assert cu.count("__trap()") == old.count("__trap()")
        # Shared "for(c<NCH)" anchors also exist in load_x_regs. A probe
        # lambda must begin AFTER the skinny unit's four accumulator arrays.
        if kind == "probe":
            assert cu.index("auto tinyx_unit") > cu.index("float acc3[4]") > cu.index("__device__ void skinny_ring_ws(")
            assert cu.index("};\nif (tinyx_class == 1)") < cu.index("const int bb = q % NRB;")
    return s


def exact_cpu_gate():
    # All 65,536 scale-byte pairs, in both even/odd selectors. Carry-free packed
    # multiply/add reproduces both BF16 lanes, hence the same HMUL2 operands.
    for lo in range(256):
        for hi in range(256):
            merged = ((lo | (hi << 16)) * 16 + 0x5C005C00) & 0xffffffff
            assert merged & 65535 == lo * 16 + 0x5C00
            assert merged >> 16 == hi * 16 + 0x5C00
            for shift in (0, 16):
                w = (lo | (hi << 8)) << shift
                assert ((w >> shift) & 255) == lo and ((w >> (shift + 8)) & 255) == hi
    mapped = [((t + 256) % 288) // 32 for t in range(288)]
    assert set(mapped[:32]) == {8}
    assert {mapped[t] for t in range(32, 288) if (t // 32) % 4 == 0} == {3, 7}
    assert TINYX_ARMS["S0"][1] & 255 == (1 << 3 | 1 << 7)


def dry_run(base, patched, fixture=False):
    """Execute real _hyblo_prepare/_hyb_scratch and every new host branch.

    Only tensor arithmetic, prepack, reference values and compiler are doubles.
    State dictionaries, allocations' shapes, flags, build arguments and selectors
    are real AST nodes from the base and emitted layer; no permissive dicts.
    """
    class Tensor:
        device = "cpu-dry"
        def __init__(self, shape=(8, 4096)): self.shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        def __getitem__(self, k):
            if isinstance(k, slice):
                return Tensor((len(range(*k.indices(self.shape[0]))),) + self.shape[1:])
            return Tensor(self.shape)
        def __add__(self, x): return self
        __radd__ = __mul__ = __rmul__ = __eq__ = __add__
        def all(self): return True
        def numel(self):
            import math
            return math.prod(self.shape)
        def view(self, *s): return self if len(s) == 1 and isinstance(s[0], str) else Tensor(s)
        def reshape(self, *s): return Tensor(s)
        def to(self, *a): return self
        def contiguous(self): return self
        clone = float = contiguous
        def fill_(self, v): return self
        def data_ptr(self): return 16
    builds, calls, failures = [], [], set()
    class Module:
        def __init__(self, key): self.key, self.flags, self.pdl = key, 0, None
        def okc_set_pdl(self, pdl): self.pdl = pdl
        def okc_set_tinyx(self, flags): self.flags = flags
        def okc_run(self, *a):
            assert a[0] is ns["_HYBLO"]["w"] and a[1] is ns["_HYBLO"]["sc"]
            assert a[-3:] == (0, 32, 132)
            calls.append((self.key, self.flags, self.pdl, a[2].shape[0]))
    torch = NS(**{s: s for s in ("int16", "int32", "float32", "bfloat16", "uint8")},
        empty=lambda s, **k: Tensor(s), zeros=lambda s, **k: Tensor(s),
        arange=lambda *a, **k: Tensor((a[0],)), stack=lambda *a, **k: Tensor(), cat=lambda *a, **k: Tensor(),
        isfinite=lambda t: t, cuda=NS(synchronize=lambda: None))
    ns = dict(torch=torch, time=time, hashlib=hashlib, _pt1_wait=lambda *a: None,
              _k2a_ref_fp32=lambda *a: Tensor(), _k2a_rel_err=lambda *a: (0., 0.))
    names = ("_HYB", "_HYB_MMAX", "_HYB_EMAX", "_HYB_FIXF", "_HYB_XROWS", "_HYB_GUARD",
             "_HYBLO", "_HYBLO_CUDA", "_HYB_MOE_CPP", "_HYBLO_FLAGS", "_HYBLO_KEYS", "HYBLO_ENABLE",
             "HYBLO_PDL", "HYBLO_T", "HYBLO_K", "HYBLO_SELF_TEST", "TINYX_ON", "TINYX_ARMS",
             "_TINYX", "_TINYX_HELP_D1", "_TINYX_HELP_PROBE", "_tinyx_sub", "_tinyx_source",
             "_tinyx_build_one", "_tinyx_prepare", "_tinyx_select", "_hyb_scratch", "_hyb_guards_ok",
             "_hyblo_key", "_hyblo_build", "_hyblo_run", "_hyblo_route", "_hyblo_prepare")
    for name in names:
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), f"real:{name}", "exec"), ns)
    def load_inline(**kw):
        if kw["name"].startswith("hyblo_"):
            kind, k = "ctl", kw["name"].split("_")[1]
        else:
            _, k, kind, _ = kw["name"].split("_")
        var = next(v for key, _, v in ns["_HYBLO_KEYS"] if key == k)
        assert kw["cuda_sources"] == _tinyx_source(literal(base, "_HYBLO_CUDA"), kind)
        assert kw["extra_cuda_cflags"] == list(ns["_HYBLO_FLAGS"]) + list(var) + ([] if kind == "ctl" else ["-Xptxas=-v"])
        assert kw["functions"] == ["okc_run", "okc_set_pdl"] + (["okc_set_tinyx"] if kind == "probe" else [])
        assert kw["cpp_sources"] == literal(base, "_HYB_MOE_CPP") + ("\nvoid okc_set_tinyx(int64_t flags);\n" if kind == "probe" else "")
        builds.append((k, kind))
        if (k, kind) in failures: raise RuntimeError("injected build failure")
        return Module((k, kind))
    original_import = builtins.__import__
    ns["__builtins__"] = dict(vars(builtins), __import__=lambda n, *a, **k: NS(load_inline=load_inline) if n == "torch.utils.cpp_extension" else original_import(n, *a, **k))
    # Re-create functions with the restricted per-exec imports, never sys.modules.
    for name in ("_hyblo_build", "_tinyx_build_one"):
        exec(compile(ast.Module(body=[node(patched, name)], type_ignores=[]), name, "exec"), ns)
    for name in ("_hyblo_pack_codes", "_hyblo_pack_scales", "_hyblo_fat_order", "_hyblo_unit_order"):
        ns[name] = lambda *a: Tensor()
    ns["_hyblo_inputs"] = lambda bs, *a: {k: Tensor((bs, 4096)) for k in ("hidden_states", "topk_ids", "topk_weights")}
    ns["_v11_k2a33_call"] = lambda *a: Tensor()
    ns["_HYB"].update(ok=True, w=[Tensor() for _ in range(14)], nsm=132, full=[])
    w = {k: Tensor() for k in ("w13", "w13_sf", "w2", "w2_sf")}
    ns["_hyblo_prepare"]({}, w, "cpu-dry")
    assert ns["_HYBLO"]["ok"] == {"tiny": True, "tiny2": True}
    assert len(ns["_HYBLO"]["sc"]) == 3 and len(ns["_HYBLO"]["w"]) == 14
    ns["_tinyx_prepare"](); ns["_tinyx_prepare"]()
    assert len(builds) == 6 and not ns["_TINYX"]["errors"]
    def must_fail(f):
        try: f()
        except (RuntimeError, KeyError): return
        raise AssertionError("failure silently accepted")
    ns["TINYX_ON"] = "D1"
    must_fail(lambda: ns["_tinyx_select"]("tiny"))
    ns["_TINYX"]["accepted"] = set(ns["_TINYX"]["mods"])
    for t in (1, 2, 4, 5, 8):
        k = "tiny" if t <= 4 else "tiny2"
        for arm, (kind, flags) in TINYX_ARMS.items():
            for pdl in (True, False, True):
                ns["TINYX_ON"], ns["HYBLO_PDL"] = arm, pdl
                out = ns["_hyblo_route"](Tensor((t, 4096)), Tensor(), Tensor())
                assert out.shape == (t, 4096) and calls[-1] == ((k, kind), flags, int(pdl), t)
    ns["TINYX_ON"] = "bad"; must_fail(lambda: ns["_tinyx_select"]("tiny"))
    ns["TINYX_ON"] = "D1"
    del ns["_TINYX"]["mods"][("tiny", "D1")]
    failures.add(("tiny", "D1")); ns["_tinyx_prepare"]()
    must_fail(lambda: ns["_tinyx_select"]("tiny"))
    failures.clear(); ns["_tinyx_prepare"](); ns["_tinyx_select"]("tiny")
    saved = ns["_HYBLO"]["sc"]; ns["_HYBLO"]["sc"] = None
    must_fail(ns["_tinyx_prepare"]); ns["_HYBLO"]["sc"] = saved
    ns["_HYBLO"]["ok"]["tiny2"] = False; must_fail(ns["_tinyx_prepare"])
    ns["HYBLO_ENABLE"] = False; ns["_tinyx_prepare"]()
    ns["HYBLO_ENABLE"] = True; ns["TINYX_ON"] = "ctl"
    # Execute the new prepare call site against the REAL outer return layout.
    ns.update(st={}, state={}, weights={k: Tensor() for k in
        ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_expert_norm")}, t0=0)
    for name in ("K2A_M_MAX", "K2A_BUCKETS", "V11_K2A33_ROUTE"):
        exec(compile(ast.Module(body=[node(base, name)], type_ignores=[]), name, "exec"), ns)
    prep = node(base, "_v11_k2a33_prepare")
    ok_stmt = next(n for n in prep.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "ok")
    exec(compile(ast.Module(body=[ok_stmt], type_ignores=[]), "real-ok", "exec"), ns)
    ret = next(n.value for n in prep.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    k33 = eval(compile(ast.Expression(ret), "real-return", "eval"), ns)
    ns["_v11_k2a33_prepare"] = lambda *a: k33
    assignment = next(n for n in ast.walk(ast.parse(base)) if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "_v11_k2a33_prepare")
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "real-state", "exec"), ns)
    site = next(l.strip() for l in patched.splitlines() if '_hyblo_prepare(state["k2a33"]["st"]' in l)
    exec(site + "\n_tinyx_prepare()", ns)
    # The actual layer dispatch must reach the HYBLO branch at both scored T.
    exec(compile(ast.Module(body=[node(base, "_v11_k2a33_call")], type_ignores=[]), "real-route", "exec"), ns)
    for t in (1, 8):
        for arm in TINYX_ARMS:
            ns["TINYX_ON"] = arm
            assert ns["_v11_k2a33_call"](Tensor((t, 4096)), Tensor(), Tensor(), k33["st"]).shape == (t, 4096)
    ns["TINYX_ON"] = "ctl"
    print("[tinyx] CPU dry PASS: real prepare/scratch, six builds, cache, all arms/PDL, missing gate/bank, independent failure/retry")
    if fixture:
        class ModuleProxy:
            def __getattr__(self, k):
                if k not in ns: raise AttributeError(k)
                return ns[k]
            def __setattr__(self, k, v): ns[k] = v
        return ModuleProxy(), ns["state"], Tensor


def emit_artifacts(base, source):
    ART.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(source.encode())
    (ART / "layer.diff").write_text("".join(difflib.unified_diff(base.splitlines(True), source.splitlines(True), fromfile=str(BASE.relative_to(ROOT)), tofile=str(OUT.relative_to(ROOT)))), encoding="utf-8")
    cu = literal(base, "_HYBLO_CUDA")
    changes = {}
    source_start = node(base, "_HYBLO_CUDA").lineno
    for kind in ("ctl", "D1", "probe"):
        text = _tinyx_source(cu, kind)
        (ART / f"{kind}.cu").write_text(text, encoding="utf-8")
        (ART / f"{kind}.diff").write_text("".join(difflib.unified_diff(cu.splitlines(True), text.splitlines(True), fromfile="ship_HYBLO.cu", tofile=f"{kind}.cu")), encoding="utf-8")
        changes[kind] = [{"base_file": str(BASE.relative_to(ROOT)), "base_lines_inclusive": [source_start+a0, source_start+a1-1],
                          "cuda_file": f"box_out/tinyx/{kind}.cu", "cuda_lines_inclusive": [b0+1, b1], "kind": tag}
                         for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(None, cu.splitlines(), text.splitlines(), autojunk=False).get_opcodes() if tag != "equal"]
    manifest = {"base_sha256": hashlib.sha256(base.encode()).hexdigest(), "layer_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "cuda_sha256": {k: hashlib.sha256(_tinyx_source(cu, k).encode()).hexdigest() for k in ("ctl", "D1", "probe")},
                "arms": TINYX_ARMS, "status": "CPU_PASS; compile/SASS/GPU pending"}
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    changes["S0"] = {"cuda_diff": "probe.diff (same binary as DQ/MM/pctl)", "selector": "logical warp mask 0x88, physical warps 4/8; helper physical warp 0"}
    changes["DQ"] = {"cuda_diff": "probe.diff", "runtime_flags": "0x1ff: class1, all 8 compute warps; high salt bits zero"}
    changes["MM"] = {"cuda_diff": "probe.diff", "runtime_flags": "0x2ff: class2, all 8 compute warps; high sink-mask bits zero"}
    (ART / "changes.json").write_text(json.dumps(changes, indent=2))
    accounting = {"label": "EST source accounting, NOT measured SASS; no timing claim",
        "per_unit_bytes_before_and_all_arms_after": {"HBM_weight_and_scale_per_CTA": 36864, "shared_weight_and_scale_per_compute_lane": 144,
            "shared_reduction_store_per_compute_lane": 16, "shared_reduction_read_per_reducer_lane": 128},
        "unchanged": "x-register loading at job change, all global scratch reads/writes, parameter size, launches, barriers/refills; no new weight bytes",
        "per_compute_lane_per_unit_core_instructions": {
            "ctl_and_pctl_class0": {"unpack_INT":256, "scale_PRMT":16, "scale_IMAD":16, "HMUL2":128, "HMMA":32, "weight_scale_LDS128":9, "sum":457},
            "D1": {"unpack_INT":256, "scale_PRMT":8, "scale_IMAD":8, "HMUL2":128, "HMMA":32, "weight_scale_LDS128":9, "sum":441},
            "DQ_class1": {"unpack_INT":96, "scale_PRMT":16, "scale_IMAD":16, "HMUL2":128, "HMMA":32, "weight_scale_LDS128":9, "sum":297},
            "S0": "class1 on logical 3/7 only; other compute warps class0; helper unchanged",
            "MM_class2": {"unpack_INT":256, "scale_PRMT":16, "scale_IMAD":16, "HMUL2":128, "HMMA":0, "sink_LOP3":96, "weight_scale_LDS128":9, "sum":521}},
        "exclusions": "address/control, accumulators/FADD, probe class dispatch and sink publication; real SASS spans must include these before a timing inference",
        "kill_risks": ["TINY/TINY2 issue cuts may be hidden by the stream or another warp", "register pressure or probe dispatch can offset cuts"],
        "measured_baseline_source": "notes/lab.md:1203: TINY 21.79 us; TINY2 55.33/55.81 us. No timing of these new arms.",
        "gate": "HOLD until WSL compile + reviewed W13/W2 SASS + spill-free resources; exactness and critical-path exposure require H200"}
    (ART / "accounting.json").write_text(json.dumps(accounting, indent=2))
    protocol = {"changed_protocol": False, "proof": "generator asserts byte-identical publication/cleanup/combine suffix and identical protocol lines",
        "geometry": "G=132, physical threads 0..287; logical TIDX=(threadIdx.x+256)%288; helper physical 0..31",
        "coverage": [
            {"base_lines":"7057-7087", "work":"unit weight/wsre waits, red stores, CTA_SYNC, refill and wsrf arrive", "owners":"every CTA: logical 0..255 = physical 32..287; logical tid0 (physical32) refill/arrive"},
            {"base_lines":"7089-7119", "work":"final B tail reduction/ybuf/ss", "owners":"every CTA: logical warp0..1 (TINY) or warp0 (TINY2), conditional warp<nU, every lane0..31"},
            {"base_lines":"7128-7200", "work":"helper wsrf wait/reduce, wsre arrive, W13 inter/W2 ybuf+ss, doneA publish, WS_JOIN", "owners":"every CTA: physical warp0 lanes0..31; lane0 arrives and publishes all jj=jf..jl; first/last joins include all288 threads"},
            {"base_lines":"7372-7376", "work":"zero next ss bank and next completion counter", "owners":"CTA0..11 logical0..255 cover EMAX3072 via block*256+tid; CTA0 logical0 resets counter"},
            {"base_lines":"7384-7410", "work":"both phase joins, completion atomic/wait and compute barrier", "owners":"every CTA logical0 waits/increments, all256 compute threads remain; helper returns only after its final WS_JOIN"},
            {"base_lines":"7412-7416", "work":"reset ctr0/1 and doneA[0..JMAX)", "owners":"CTA0 logical0/1 reset ctr; CTA0 logical0..87 reset all88 jobs (JMAX=32+3072/64+8)"},
            {"base_lines":"7417-7441", "work":"prefetched first combine", "owners":"T1 CTA0..3 / T8 CTA0..31, logical0..255, four columns/thread, 4 CTAs/output row"},
            {"base_lines":"7442-7460", "work":"remaining combine loop", "owners":"all CTA c, logical0..255, it=c+132*n while it<M*4; zero iterations at scored T1/T8, retained verbatim"}],
        "new_exits": 0, "new_waits": 0, "new_barriers": 0, "new_resets": 0}
    (ART / "protocol.json").write_text(json.dumps(protocol, indent=2))


def main():
    base = BASE.read_text(encoding="utf-8")
    source = build(base)
    exact_cpu_gate(); dry_run(base, source); emit_artifacts(base, source)
    print(f"[tinyx] generated {OUT.relative_to(ROOT)}, {len(source.encode())} bytes (lab only)")
    if "--local-compile" in sys.argv:
        import runpy
        return runpy.run_path(str(ROOT / "tools/box/tinyx_run.py"))["local_compile"]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

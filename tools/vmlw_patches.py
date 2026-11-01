"""R-vmlw-1: runtime null consumers in the real T=32 VML kernel (lab only).

  python -B -I tools/vmlw_patches.py

Pure anchored source construction; never imports the layer or starts a GPU job.
Independent 4-bit warp-position masks select null mma.sync / RS consumers in one binary.
BLOCKED: a per-wp RS return violates warpgroup collective participation.
The candidate is constructed only in memory; protocol_gate prevents emission.
--dry-run verifies host paths without importing the layer or writing output.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmlw_ab.py"
LIMIT = 1_000_000


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


GLOBALS = '''# VMLW lab only: select warp-position masks (0..15) before capture; fixed within a launch.
VMLW_WPM = 0
VMLW_WPR = 0
_VMLW = {"calls": {}}
'''
CUDA_SHIFTS = '''// VMLW: warp-position masks in bits 18..25; bits 30..31 stay zero (SFX dependency masks).
constexpr int VMLW_SHIFT_M = 18;
constexpr int VMLW_SHIFT_R = 22;
'''
ROUTE = '''    if rep == 32:
        fl |= ((VMLW_WPM & 15) << 18) | ((VMLW_WPR & 15) << 22)
        key = f"{VMLW_WPM},{VMLW_WPR}"
        _VMLW["calls"][key] = _VMLW["calls"].get(key, 0) + 1
'''

# Protocol projection of stage(), including the less common PRE / EA / G2 /
# HOLD cases. Keep loop event order: a deferred flush can release a held slot.
# No hb contents survive for a reader in a null WG: all its calls in this RS
# segment take the same branch and issue no WPMMA group. Other WGs stay live.
NULL_R = '''if ((A.flags >> (22 + wp)) & 1) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
#if VM_TRICKS
if constexpr (VC::EA == 1) {
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
}
#else
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
#endif
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
// seg's RS d is uninitialized: the normal first MMA uses scale-D=0.
if (first) {
#pragma unroll
for (int k = 0; k < N / 2; ++k) d[k] = 0.f;
}
#if VM_TRICKS
if constexpr (VC::PRE) {
#pragma unroll
for (int g = 0; g < 8; ++g) {
const int i = g >> 1, hf = g & 1;
if (g == 5) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
if (g == 0 && prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && hf == VCP_DPH) pend_flush(A, pd, q, wp, lane);
}
}
} else
#endif
{
#pragma unroll
for (int i = 0; i < 4; ++i) {
#pragma unroll
for (int hf = 0; hf < 2; ++hf) {
#if VM_TRICKS
#if VSE_HOLD
if (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0 && !hold) mbar_arrive(bars + 8 * (NS + ws)); }
#else
if (VC::EA == 0 && i == 3 && hf == 0) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
#endif
if (VC::EA == 2 && i == 0 && hf == 1) { __syncwarp(); if (lane == 0) mbar_arrive(bars + 8 * (NS + ws)); }
#endif
if (!VC::G2 || hf == 1) {
if ((VC::G2 ? i : 2 * i + hf) == VC::W - 1 && prevx >= 0) {
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + prevx));
prevx = -1;
}
if constexpr (VM_DEFER) {
if (first && i == VCP_DPI && (VC::G2 || hf == VCP_DPH)) pend_flush(A, pd, q, wp, lane);
}
}
}
}
}
prevx = xs;
++it;
return;
}
'''

# SFX changes the order of weight release vs x-full wait. Its load-dependent
# address addends are zero for all supported flags (including the new bits).
# Keep both warp joins around weight release, and the join before x release.
NULL_M = '''if ((A.flags >> (18 + wp)) & 1) {
const int ws = it % NS, xs = it % NX;
mbar_wait(bars + 8 * ws, (it / NS) & 1, A.err);
#if SFX
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
#if VSE_HOLD
if (lane == 0 && !hold) mbar_arrive(bars + 8 * (NS + ws));
#else
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
#endif
__syncwarp();
#else
__syncwarp();
if (lane == 0) mbar_arrive(bars + 8 * (NS + ws));
mbar_wait(bars + 8 * (2 * NS + xs), (it / NX) & 1, A.err);
__syncwarp();
#endif
__syncwarp();
#if SFX && VSE_HOLD
if (lane == 0 && !hold) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
#else
if (lane == 0) mbar_arrive(bars + 8 * (2 * NS + NX + xs));
#endif
++it;
return;
}
'''


def literal(s, name):
    nodes = [n for n in ast.parse(s).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(nodes) == 1, f"unique assignment: {name}"
    return ast.literal_eval(nodes[0].value)


def build(base):
    cu = literal(base, "_vml_CUDA_T")
    assert "VMLW" not in base, "base already patched"
    # Grep-equivalent mask audit, plus the full host transport anchors.
    reserved = (15 << 18) | (15 << 22)
    masks = [int(x) for x in re.findall(r"\bflags\s*&\s*(\d+)", cu)]
    assert masks and max(masks) == 32768
    assert all(not (x & reserved) for x in masks)
    for bit in range(18, 26):
        assert not re.search(rf"\b(?:{1 << bit}|0[xX]0*{1 << bit:x})[uUlL]*\b|1[uUlL]*\s*<<\s*{bit}\b", cu)
    for anchor in (
        "unsigned long long magG; int flags;",
        "a.flags = (int)V[5];",
        "flags = _vml_FLAGS_DEFAULT if flags is None else flags",
        '[m, ls, sc["G"], sc["magG"], mask, flags]',
        "flags=fl, cost=VML_COST.get(rep)",
    ):
        assert base.count(anchor) == 1, f"flags transport: {anchor}"
    assert all(0 <= v < (1 << 16) for v in literal(base, "VML_FLAGS").values())
    for name in ("_vml_FLAGS_DEFAULT", "VML_DPFLAGS", "VML_SCFLAGS"):
        assert 0 <= literal(base, name) < (1 << 16), name

    stage_anchor = '''using VC = VcpCfg<N>;
#if VM_UD
static_assert(VC::NB == 2 && VC::W == 1 && !VC::G2, "VCP needs VM_UD 0");
#endif
'''
    m_anchor = '''int wp, int lane, bool hold) {
#else
int wp, int lane) {
#endif
'''
    route_anchor = "    fl = (_vml_FLAGS_DEFAULT if fl is None else fl) | (VML_DPFLAGS if dp else 0) | (VML_SCFLAGS if sc else 0)\n"
    edits = (
        ("namespace vm {\n", "namespace vm {\n" + CUDA_SHIFTS, "VML CUDA shift constants"),
        (stage_anchor, stage_anchor + NULL_R, "stage<N> null RS"),
        (m_anchor, m_anchor + NULL_M, "stage_m<N> null mma.sync"),
        ("VML_ENABLE = True\n", GLOBALS + "VML_ENABLE = True\n", "module globals"),
        (route_anchor, route_anchor + ROUTE, "_vml_route T=32 flags/counter"),
    )
    s = base
    for old, new, what in edits:
        s = sub(s, old, new, what)
    ast.parse(s)
    patched_cu = literal(s, "_vml_CUDA_T")
    route = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == "_vml_route")
    route_text = ast.get_source_segment(s, route)
    for suffix, shift in (("M", 18), ("R", 22)):
        assert f"constexpr int VMLW_SHIFT_{suffix} = {shift};" in patched_cu
        assert f"((VMLW_WP{suffix} & 15) << {shift})" in route_text
        assert patched_cu.count(f"if ((A.flags >> ({shift} + wp)) & 1)") == 1
        assert literal(s, f"VMLW_WP{suffix}") == 0
    assert literal(s, "_VMLW") == {"calls": {}}
    assert "VMLW_MODE" not in s and "VMLW_BIT_" not in s
    # Execute only the injected host statements: all mask pairs, counted even
    # when off, and no flag/counter change at other buckets. No layer import.
    route_code = compile("if True:\n" + ROUTE, "<VMLW route injection>", "exec")
    for rep in (16, 32, 128, 256):
        for m in range(16):
            for r in range(16):
                ns = dict(rep=rep, fl=193, VMLW_WPM=m, VMLW_WPR=r, _VMLW={"calls": {}})
                exec(route_code, ns)
                extra = ((m << 18) | (r << 22)) if rep == 32 else 0
                assert ns["fl"] == 193 | extra and ns["fl"] >> 30 == 0
                assert ns["_VMLW"]["calls"] == ({f"{m},{r}": 1} if rep == 32 else {})
    for fl in range(1 << 16):
        assert fl | ((0 & 15) << 18) | ((0 & 15) << 22) == fl
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base, "edits are not exactly reversible"
    return s



def host_dry_run(base, source):
    """Execute real base module layout/build bank and both real route functions."""
    import types
    tree = ast.parse(base)
    wanted = {"_VML", "VML_VAR", "VML_ABVARS", "VML_BVAR", "_VML_DEFS", "VML_FLAGS",
              "VML_COST", "VML_POOL", "VML_DPFLAGS", "VML_SEG", "VML_SCFLAGS",
              "VML_PDEL", "_vml_FLAGS_DEFAULT"}
    selected = [n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets)]
    build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_vml_build_job")
    ns = {"_vml_build": lambda **kw: (object(), 0.0)}
    exec(compile(ast.Module(body=selected + [build], type_ignores=[]), "<real base VML layout>", "exec"), ns)
    # Stub only the CUDA builder; actual bank creation and selection run unchanged.
    ns["_vml_build_job"]()
    assert ns["_VML"]["mod"] is ns["_VML"]["mods"]["bf2"]
    assert set(ns["_VML"]["mods"]) == {"bf2", "w72"}
    for name in ("sc", "scb", "scw"):
        assert name in ns["_VML"]
        ns["_VML"][name] = object()
    class Tensor:
        def __init__(self, m): self.shape = (m,)
        def contiguous(self): return self
        def to(self, dtype): return self
    ns.update(torch=types.SimpleNamespace(int32=object(), float32=object()),
              _nb_vml_rep=lambda m: m, _vml_call=lambda *args, **kw: (args, kw))
    exec(GLOBALS, ns)
    routes = []
    for text in (base, source):
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_vml_route")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<real VML route>", "exec"), ns)
        routes.append(ns["_vml_route"])
    runner = ast.parse((ROOT / "tools/box/vmlw_run.py").read_text())
    arm_node = next(n for n in runner.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "ARMS" for t in n.targets))
    arm_ns = {}
    exec(compile(ast.Module(body=[arm_node], type_ignores=[]), "<runner arms>", "exec"), arm_ns)
    count = 0
    for force in (None, "bf2"):
        ns["_VML"]["force"] = force
        for validated in (False, True):
            ns["_VML"]["okv"] = {(rep, "w72"): validated for rep in (128, 256)}
            for rep in (16, 32, 128, 256):
                data = {"hidden_states": Tensor(rep), "topk_ids": Tensor(rep),
                        "topk_weights": Tensor(rep), "config": {"local_start": 0}}
                st = object()  # unchanged route passes its st argument through
                expected_args, expected_kw = routes[0](st, data)
                for name, arm in arm_ns["ARMS"]:
                    ns.update(arm)
                    args, kw = routes[1](st, data)
                    extra = (arm["VMLW_WPM"] << 18) | (arm["VMLW_WPR"] << 22) if rep == 32 else 0
                    assert args == expected_args and kw == dict(expected_kw, flags=expected_kw["flags"] | extra)
                    assert args[1] is st
                    if rep == 32:
                        assert args[0] is ns["_VML"]["mods"]["bf2"] and args[2] is ns["_VML"]["scb"]
                    count += 1
    assert ns["_VMLW"]["calls"] == {"0,0": 8, **{f"{m},{m}": 4 for m in (1, 2, 4, 8, 3, 12)}}
    print(f"[vmlw] host dry run PASS: {count} real-route arm/shape/bank cases; no new prepare-side state access")


def protocol_gate(base):
    """Reject mixed-WP RS collective participation before emitting GPU code.

    PTX ISA wgmma.*.sync.aligned requires identical participation across 128
    threads. Warp-uniform is insufficient. This is a CPU falsifier, not a
    hardware timing prediction. No GPU result may override this contract.
    """
    cu = literal(base, "_vml_CUDA_T")
    expr = "const int q = warp >> 2, wp = warp & 3;"
    assert cu.count(expr) == 1
    line = next(i for i, s in enumerate(base.splitlines(), 1) if s == expr)
    print(f"[vmlw] base vm_main:{line}: {expr} (wp == (threadIdx.x >> 5) % 4)")
    assert '"-DVM_SYNC7=1"' in base and "if constexpr (N <= 16) sync_path = true;" in cu
    assert "wg_fence();" in cu and "wg_commit();" in cu and "wg_wait<VC::W>();" in cu
    errors = []
    for name, mask in (("mid1", 0), ("w0", 1), ("w1", 2), ("w2", 4),
                       ("w3", 8), ("w01", 3), ("w23", 12), ("mid2", 0)):
        for q in range(4):
            lanes = [((tid >> 5) & 3) for tid in range(128 * q, 128 * (q + 1))]
            take = [(mask >> wp) & 1 for wp in lanes]
            assert all(len(set(take[w:w + 32])) == 1 for w in range(0, 128, 32))
            if len(set(take)) != 1:
                skipped = [w for w in range(4 * q, 4 * q + 4) if mask >> (w & 3) & 1]
                kept = [w for w in range(4 * q, 4 * q + 4) if not (mask >> (w & 3) & 1)]
                errors.append(f"{name} WG{q}: skipped warps {skipped}; WGMMA participants {kept}")
    assert len(errors) == 24
    print("[vmlw] protocol FAIL: six nonzero arms split all four WGs; RS stage<N>=24 requires all 128 threads")
    for e in errors:
        print("[vmlw] " + e)
    raise ValueError("RS wp early return violates WGMMA .aligned; kernel emission blocked; revised probe semantics required")


def main():
    import sys
    if sys.argv[1:] not in ([], ["--dry-run"]):
        raise SystemExit("usage: python -B -I tools/vmlw_patches.py [--dry-run]")
    # Preserve the base's exact newline spelling as well as its source text.
    raw = BASE.read_bytes()
    base = raw.decode("utf-8")
    crlf = b"\r\n" in raw
    assert not crlf or base.count("\n") == base.count("\r\n"), "mixed newlines"
    source = build(base.replace("\r\n", "\n"))
    out = source.replace("\n", "\r\n").encode("utf-8") if crlf else source.encode("utf-8")
    host_dry_run(base.replace("\r\n", "\n"), source)
    print(f"[vmlw] in-memory candidate {len(out):,} B (base {len(raw):,} B)")
    assert len(out) <= LIMIT, "candidate exceeds 1 MB"
    protocol_gate(base.replace("\r\n", "\n"))
    if sys.argv[1:] == ["--dry-run"]:
        return
    OUT.write_bytes(out)
    n = len(out)
    print(f"[vmlw] {OUT.relative_to(ROOT)}: {n:,} bytes; "
          f"{max(0, n - LIMIT):,} bytes over 1 MB cap; headroom {max(0, LIMIT - n):,}")
    print("[vmlw] AST, unused flag bits/transport, and exact reverse-edit gates PASS")


if __name__ == "__main__":
    main()

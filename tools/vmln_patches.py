"""R-vmln-1: runtime null consumers in the real T=32 VML kernel (lab only).

  python -B -I tools/vmln_patches.py

Pure anchored source construction; never imports the layer or starts a GPU job.
Independent 4-bit WG masks select null mma.sync / RS consumers in one binary.
The normal consumer bodies and all producer/segment/epilogue code are unchanged.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "kernels/lab/k3_hh_5_sk2_pz.py"
OUT = ROOT / "kernels/lab/k3_vmln_ab.py"
LIMIT = 1_000_000


def sub(s, old, new, what):
    assert s.count(old) == 1, f"{what}: expected one anchor, got {s.count(old)}"
    return s.replace(old, new, 1)


GLOBALS = '''# VMLN lab only: select WG masks (0..15) before capture; fixed within a launch.
VMLN_WGM = 0
VMLN_WGR = 0
_VMLN = {"calls": {}}
'''
CUDA_SHIFTS = '''// VMLN: WG masks in bits 18..25; bits 30..31 stay zero (SFX dependency masks).
constexpr int VMLN_SHIFT_M = 18;
constexpr int VMLN_SHIFT_R = 22;
'''
ROUTE = '''    if rep == 32:
        fl |= ((VMLN_WGM & 15) << 18) | ((VMLN_WGR & 15) << 22)
        key = f"{VMLN_WGM},{VMLN_WGR}"
        _VMLN["calls"][key] = _VMLN["calls"].get(key, 0) + 1
'''

# Protocol projection of stage(), including the less common PRE / EA / G2 /
# HOLD cases. Keep loop event order: a deferred flush can release a held slot.
# No hb contents survive for a reader in a null WG: all its calls in this RS
# segment take the same branch and issue no WGMMA group. Other WGs stay live.
NULL_R = '''if ((A.flags >> (22 + q)) & 1) {
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
NULL_M = '''if ((A.flags >> (18 + q)) & 1) {
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
    assert "VMLN" not in base, "base already patched"
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
        assert f"constexpr int VMLN_SHIFT_{suffix} = {shift};" in patched_cu
        assert f"((VMLN_WG{suffix} & 15) << {shift})" in route_text
        assert patched_cu.count(f"if ((A.flags >> ({shift} + q)) & 1)") == 1
        assert literal(s, f"VMLN_WG{suffix}") == 0
    assert literal(s, "_VMLN") == {"calls": {}}
    assert "VMLN_MODE" not in s and "VMLN_BIT_" not in s
    # Execute only the injected host statements: all mask pairs, counted even
    # when off, and no flag/counter change at other buckets. No layer import.
    route_code = compile("if True:\n" + ROUTE, "<VMLN route injection>", "exec")
    for rep in (16, 32, 128, 256):
        for m in range(16):
            for r in range(16):
                ns = dict(rep=rep, fl=193, VMLN_WGM=m, VMLN_WGR=r, _VMLN={"calls": {}})
                exec(route_code, ns)
                extra = ((m << 18) | (r << 22)) if rep == 32 else 0
                assert ns["fl"] == 193 | extra and ns["fl"] >> 30 == 0
                assert ns["_VMLN"]["calls"] == ({f"{m},{r}": 1} if rep == 32 else {})
    for fl in range(1 << 16):
        assert fl | ((0 & 15) << 18) | ((0 & 15) << 22) == fl
    restored = s
    for old, new, what in reversed(edits):
        restored = sub(restored, new, old, "undo " + what)
    assert restored == base, "edits are not exactly reversible"
    return s


def main():
    import sys
    if len(sys.argv) != 1:
        raise SystemExit("usage: python -B -I tools/vmln_patches.py (lab only; no --ship)")
    # Preserve the base's exact newline spelling as well as its source text.
    raw = BASE.read_bytes()
    base = raw.decode("utf-8")
    crlf = b"\r\n" in raw
    assert not crlf or base.count("\n") == base.count("\r\n"), "mixed newlines"
    source = build(base.replace("\r\n", "\n"))
    out = source.replace("\n", "\r\n").encode("utf-8") if crlf else source.encode("utf-8")
    OUT.write_bytes(out)
    n = len(out)
    print(f"[vmln] {OUT.relative_to(ROOT)}: {n:,} bytes; "
          f"{max(0, n - LIMIT):,} bytes over 1 MB cap; headroom {max(0, LIMIT - n):,}")
    print("[vmln] AST, unused flag bits/transport, and exact reverse-edit gates PASS")


if __name__ == "__main__":
    main()

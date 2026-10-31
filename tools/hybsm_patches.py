"""hybsm_patches.py (R-hybsm-*): HYB (the public alygg77 WGS MoE engine, A32) x ring 3 -> 2 slots (OKC_WGSNX 2).

  python -I tools/hybsm_patches.py [--base=kernels/lab/k3_hh_3kv.py] [--out=kernels/lab/k3_hybsm_s.py] [--check]      (ship)
  python -I tools/hybsm_patches.py --mode=lab [--base=kernels/lab/k3_wgs32_ab.py] [--out=kernels/lab/k3_hybsm_ab.py]   (A/B)
  python -I tools/lead_gate.py file <out>

Ship mode: the one line `#define OKC_WGSNX 3` in _HYB_MOE_CUDA becomes 2: HYB's x ring keeps 2 x 32 KB slots instead of 3
(dynamic smem 231,168 -> 198,400 B, tools/local/hybsm_probe.py); the weight ring is a separate constant (WGS_NS = 3 x 36,864 B)
and stays 3 deep. Anchor touched: that #define only (HYB unit-end code untouched, so it stacks with R-hybst).
Lab mode: adds a module "nx2" (tools/hybx_common.py helper: own scratch, own PT1 job "hybnx2", ship HYB weight copies) built
from _HYB_MOE_CUDA with that line changed, routed by HYBNX2_T (default (0, -1) = never; arms set it), prepare self-test
HYBNX2_TESTS incl. the concentrated (32, 72, skewed) input. If the base carries the FT_T = 32 module, its self-test list is cut to
two cases (concentrated + conc1) to keep the burst's prepare short.
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import hybx_common  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
LIMIT = 1_000_000
A_NX = "#define OKC_WGSNX 3\n"
N_NX = "#define OKC_WGSNX 2\n"


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybsm_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


LAB = r'''HYBNX2_T = (0, -1)
HYBNX2_TESTS = ((128, 64, "skewed"), (256, 67, "normal"), (32, 72, "skewed"))
def _hybnx2_src(s):
    return s.replace("#define OKC_WGSNX 3\n", "#define OKC_WGSNX 2\n") if s.count("#define OKC_WGSNX 3\n") == 1 else None
_HYBX["nx2"] = {"src": _hybnx2_src(_HYB_MOE_CUDA), "mod": None, "ok": False, "pdl": -1}
def _hybnx2_prepare(st, w, dev):
    if _HYBX["nx2"]["src"] is None:
        print("[hybnx2] OFF: the OKC_WGSNX anchor is missing", flush=True)
        return
    _hybx_prepare("nx2", w, dev, HYBNX2_TESTS)
'''
A_DISPATCH = "    if _hyb_on(x.shape[0]):\n"
N_DISPATCH = ("    if HYBNX2_T[0] <= x.shape[0] <= HYBNX2_T[1] and _HYBX[\"nx2\"][\"ok\"]:\n"
              "        return _hybx_call(\"nx2\", x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              + A_DISPATCH)
A_PT1 = '        jobs.append(("hyb", _hyb_build))\n'
N_PT1 = A_PT1 + '        if _HYBX["nx2"]["src"] is not None:\n            jobs.append(("hybnx2", lambda: _hybx_build("nx2")))\n'
A_PREP = "    _hyb_prepare(st, w, dev)\n"
N_PREP = A_PREP + "    _hybnx2_prepare(st, w, dev)\n"
A_T32 = ('HYB32_TESTS = ((17, 70, "skewed"), (32, 71, "normal"), (32, 72, "skewed"), (48, 73, "normal"), (48, 74, "skewed"),\n'
         '               (32, 75, "conc1"), (32, 76, "conc3"), (24, 77, "conc4"), (48, 78, "conc2"), (40, 79, "conc1"))\n')
N_T32 = 'HYB32_TESTS = ((32, 72, "skewed"), (32, 75, "conc1"))\n'


def apply(s, mode, base_name):
    lines = s.split(NL)
    if mode == "ship":
        check(s.count(A_NX) == 1, f"#define OKC_WGSNX 3 found {s.count(A_NX)}x, expected 1")
        lines[1] = f"# nonce k3-hybsm-s ({base_name} + R-hybsm: HYB x ring 3 -> 2 slots, OKC_WGSNX 2; tools/hybsm_patches.py)"
        return sub(NL.join(lines), A_NX, N_NX, "OKC_WGSNX")
    lines[1] = (f"# nonce k3-hybsm-ab ({base_name} + R-hybsm lab: HYB module nx2 (OKC_WGSNX 2) routed by HYBNX2_T; "
                f"tools/hybsm_patches.py)")
    s = NL.join(lines)
    s = hybx_common.ensure(s, sub)
    s = sub(s, hybx_common.ANCHOR, LAB + hybx_common.ANCHOR, "lab block")
    s = sub(s, A_DISPATCH, N_DISPATCH, "dispatch")
    s = sub(s, A_PT1, N_PT1, "pt1 job")
    s = sub(s, A_PREP, N_PREP, "prepare hook")
    if A_T32 in s:
        s = sub(s, A_T32, N_T32, "HYB32_TESTS (short)")
    return s


def main():
    mode = ARGS.get("mode", "ship")
    check(mode in ("lab", "ship"), f"--mode={mode}")
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3kv.py" if mode == "ship" else "kernels/lab/k3_wgs32_ab.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybsm_s.py" if mode == "ship" else "kernels/lab/k3_hybsm_ab.py")
    s = apply(base.read_bytes().decode("ascii"), mode, base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[hybsm_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[hybsm_patches] {mode}: {base.name} -> {out.name} ({len(s):,} B, {len(s) - base.stat().st_size:+,} B)")


if __name__ == "__main__":
    main()

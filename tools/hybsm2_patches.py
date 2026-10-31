"""hybsm2_patches.py (R-hybsm-2, F1 + F2): two HYB timing-probe modules for the d32s smem pair (lab file only, never a ship patch).

  python -I tools/hybsm2_patches.py [--base=kernels/lab/k3_wgs32_ab.py] [--out=kernels/lab/k3_hybsm_ab2.py] [--check]
  python -I tools/lead_gate.py file <out>

Base = k3_wgs32_ab.py (ship HYB + module "32" = FT_T 32, never routed). Adds two modules built from the HYB CUDA source with a text
transform at import (same helper as R-hybsm-1: own scratch, own PT1 build job, ship HYB weight copies):
  "32p"  module "32" with OKC_SMPAD bytes of UNUSED dynamic smem added to the launch (183,552 + 47,616 = 231,168 B = FT_T 64's
         size). Same code, same x ring bytes (3 x 16 KB), same fences; only the smem request / carveout step (196 -> 228 KB) moves.
         Routed by HYB32P_T. Self-test = the same two cases as "32".
  "nf"   ship HYB (FT_T 64, x ring 3) with every unit-end ordering primitive relaxed: red.release -> red.relaxed, atom.acq_rel ->
         atom.relaxed, ld.acquire -> ld.relaxed, fence.acq_rel -> compiler barrier. TIMING PROBE ONLY: results may be stale, so it has
         no self-test (built, never checked) and must be run with rev_ab3 --check=0. Routed by HYBNF_T.
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


def check(c, msg):
    if not c:
        raise SystemExit(f"[hybsm2_patches] FAIL: {msg}")


def sub(text, old, new, what):
    n = text.count(old)
    check(n == 1, f"anchor {what}: found {n}x, expected 1: {old[:90]!r}")
    return text.replace(old, new)


LAB = r'''HYB32P_T = (0, -1)
HYBNF_T = (0, -1)
HYB32P_PAD = 47616
HYB32P_TESTS = ((32, 72, "skewed"), (32, 75, "conc1"))
def _hyb32p_src(s):
    a = "size_t sm = okc_smem_bytes();\n"
    if s is None or s.count(a) != 1:
        return None
    return "#define OKC_SMPAD %d\n" % HYB32P_PAD + s.replace(a, "size_t sm = okc_smem_bytes() + (size_t)OKC_SMPAD;\n")
_HYBX["32p"] = {"src": _hyb32p_src(_HYBX["32"]["src"]), "mod": None, "ok": False, "pdl": -1}
_HYBNF_SUBS = (("red.release.gpu.global.add.s32", "red.relaxed.gpu.global.add.s32"),
               ("atom.acq_rel.gpu.global.add.s32", "atom.relaxed.gpu.global.add.s32"),
               ("ld.acquire.gpu.global.b32", "ld.relaxed.gpu.global.b32"),
               ('asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");', 'asm volatile("" ::: "memory");'))
def _hybnf_src(s):
    for a, b in _HYBNF_SUBS:
        if s.count(a) < 1:
            return None
        s = s.replace(a, b)
    return s
_HYBX["nf"] = {"src": _hybnf_src(_HYB_MOE_CUDA), "mod": None, "ok": False, "pdl": -1}
def _hyb32p_prepare(st, w, dev):
    if _HYBX["32p"]["src"] is None:
        print("[hyb32p] OFF: a source anchor is missing", flush=True)
        return
    _hybx_prepare("32p", w, dev, HYB32P_TESTS)
def _hybnf_prepare(st, w, dev):
    if _HYBX["nf"]["src"] is None:
        print("[hybnf] OFF: a fence anchor is missing", flush=True)
        return
    _hybx_prepare("nf", w, dev, ())
'''
A_DISPATCH = "    if _hyb_on(x.shape[0]):\n"
N_DISPATCH = ("    if HYB32P_T[0] <= x.shape[0] <= HYB32P_T[1] and _HYBX[\"32p\"][\"ok\"]:\n"
              "        return _hybx_call(\"32p\", x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              "    if HYBNF_T[0] <= x.shape[0] <= HYBNF_T[1] and _HYBX[\"nf\"][\"ok\"]:\n"
              "        return _hybx_call(\"nf\", x, local_ids.to(torch.int32).contiguous(), topk_weights.to(torch.float32).contiguous())\n"
              + A_DISPATCH)
A_PT1 = '        jobs.append(("hyb", _hyb_build))\n'
N_PT1 = (A_PT1 + '        if _HYBX["32p"]["src"] is not None:\n            jobs.append(("hyb32p", lambda: _hybx_build("32p")))\n'
         '        if _HYBX["nf"]["src"] is not None:\n            jobs.append(("hybnf", lambda: _hybx_build("nf")))\n')
A_PREP = "    _hyb_prepare(st, w, dev)\n"
N_PREP = A_PREP + "    _hyb32p_prepare(st, w, dev)\n    _hybnf_prepare(st, w, dev)\n"
A_T32 = ('HYB32_TESTS = ((17, 70, "skewed"), (32, 71, "normal"), (32, 72, "skewed"), (48, 73, "normal"), (48, 74, "skewed"),\n'
         '               (32, 75, "conc1"), (32, 76, "conc3"), (24, 77, "conc4"), (48, 78, "conc2"), (40, 79, "conc1"))\n')
N_T32 = 'HYB32_TESTS = ((32, 72, "skewed"), (32, 75, "conc1"))\n'


def apply(s, base_name):
    lines = s.split(NL)
    lines[1] = (f"# nonce k3-hybsm-ab2 ({base_name} + R-hybsm-2 lab: HYB modules 32p (module 32 + unused smem pad) and nf "
                f"(relaxed unit-end fences, timing probe); tools/hybsm2_patches.py)")
    s = NL.join(lines)
    check("\ndef _hybx_build(name):\n" in s, "base lacks the hybx helper (use k3_wgs32_ab.py)")
    check('_HYBX["32"] = ' in s, 'base lacks module "32" (use k3_wgs32_ab.py)')
    s = sub(s, hybx_common.ANCHOR, LAB + hybx_common.ANCHOR, "lab block")
    s = sub(s, A_DISPATCH, N_DISPATCH, "dispatch")
    s = sub(s, A_PT1, N_PT1, "pt1 jobs")
    s = sub(s, A_PREP, N_PREP, "prepare hooks")
    if A_T32 in s:
        s = sub(s, A_T32, N_T32, "HYB32_TESTS (short)")
    return s


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_wgs32_ab.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hybsm_ab2.py")
    s = apply(base.read_bytes().decode("ascii"), base.stem)
    ast.parse(s)
    check(s.isascii(), "non-ASCII output")
    check(len(s.encode()) < LIMIT, f"size {len(s.encode()):,} >= {LIMIT:,}")
    if "check" in FLAGS:
        check(out.read_bytes().decode("ascii") == s, f"{out.name} != apply({base.name})")
        print(f"[hybsm2_patches] --check PASS: {out.name} == apply({base.name}) ({len(s):,} B)")
        return
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"[hybsm2_patches] {base.name} -> {out.name} ({len(s):,} B, {len(s) - base.stat().st_size:+,} B)")


if __name__ == "__main__":
    main()

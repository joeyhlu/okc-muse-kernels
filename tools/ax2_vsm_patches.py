"""ax2_vsm_patches.py: FA16-VSM retest on a slimmed bundle (notes/lab.md R-fa16 / R-ax2-7).

  python -I tools/ax2_vsm_patches.py --base=kernels/lab/k3_b29_1.py --out=kernels/lab/k3_ax2_vsm.py [--default=16]
  then: python -I tools/slim_patches.py --base=kernels/lab/k3_ax2_vsm.py --out=<slim> (run it last; the base is already slim)

The fixed VSM (V^T per-thread constants in a 16-B smem slot per consumer thread, off_adr = off_misc + 64, smem_bytes = off_adr
+ 4 KB + 1 KB slack; db = da ^ 64 with selector b from the slot) already sits in vfa_main as DIAG 16, but the non-lab build only
instantiates DIAG 0. This adds DIAG 16 to vfa_diag's always-built dispatch and makes it the file default (VFA_DIAG), so the VFA
prepare self-test runs VSM and rev_ab3 can pair it with the ship (variant set VFA_DIAG 0).
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)


def _one(s, old, new, what):
    if s.count(old) != 1:
        raise SystemExit(f"[ax2_vsm] {what}: expected 1 match, got {s.count(old)}")
    return s.replace(old, new)


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_b29_1.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_ax2_vsm.py")
    dflt = int(ARGS.get("default", "16"))
    s = base.read_text(encoding="utf-8")
    if s.count("constexpr bool VSM = (DIAG == 16 || (DIAG == 0 && false));") != 1:
        raise SystemExit("[ax2_vsm] the fixed VSM block (DIAG 16) is not in the base")
    s = _one(s, "#endif\n(void)diag;\nvfa_go<ST, 0, O32>(p, grid, stream, pdl);\n",
             "#endif\nif (diag == 16) { vfa_go<ST, 16, O32>(p, grid, stream, pdl); return; }\n"
             "(void)diag;\nvfa_go<ST, 0, O32>(p, grid, stream, pdl);\n", "vfa_diag dispatch")
    s = _one(s, "\nVFA_DIAG = 0\n", f"\nVFA_DIAG = {dflt}\n", "VFA_DIAG default")
    lines = s.split("\n")
    for i in range(min(3, len(lines))):
        if lines[i].startswith("# nonce"):
            lines[i] = f"# nonce ax2-vsm ({base.name} + tools/ax2_vsm_patches.py: VFA DIAG 16 built, default {dflt})"
            break
    s = "\n".join(lines)
    ast.parse(s)
    out.write_bytes(s.encode("utf-8"))
    print(f"[ax2_vsm] {base.name} -> {out.relative_to(ROOT)} ({len(s.encode())} B), VFA_DIAG default {dflt}")


if __name__ == "__main__":
    main()

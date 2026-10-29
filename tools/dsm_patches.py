"""dsm_patches.py (moe lane, R-dsm-1): VML weight-ring depth at fence time (notes/study/dsm_merge.md).

R-dsm-0b: a lone fence.acq_rel.gpu waits for the SM's own in-flight bulk prefetch (4.6 us at 3 x 36 KB, 2.2 us at
2 x 36 KB, stream BW -0.2 %). VML's weight ring is `constexpr int NS = 3` (US = 36 KB slots; x ring NX = VM_NX = 3 x 32 KB,
untouched). VM_NSW makes the weight-ring depth a compile-time define (default 3 = ship, bitwise the same source);
bf2n / w72n = bf2 / w72 with -DVM_NSW=2. Only timing changes (producer waits one slot earlier); consumers, plan, epilogue
untouched. SMEM shrinks by 36 KB (BAR_OFF / TAB_OFF follow NS).

  python tools/dsm_patches.py arms [--base=kernels/muse_decode_layer/k3_gc_21.py]
      -> kernels/lab/k3_dsm_1.py (base + bf2n / w72n variants; default route bf2 / w72n; ship bf2 / w72 built by name)
  python tools/dsm_patches.py rider --out=kernels/...py [--base=...] [--var=bf2|bf2n] [--bvar=w72n]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ae_patches as AE  # noqa: E402  (_sub, defs, nonce: anchored helpers)

BASE = ROOT / "kernels/muse_decode_layer/k3_gc_21.py"
_sub, defs, nonce = AE._sub, AE.defs, AE.nonce
SCX = 'scx = _VML["scb"] if var in ("b", "bd", "p", "bf", "bg", "bf2", "w72") else _VML["sc"]'


def nsw(s):
    s = _sub(s, "constexpr int NS = 3, NX = VM_NX;\n",
             "#ifndef VM_NSW\n#define VM_NSW 3   // R-dsm-1: weight-ring slots (tools/dsm_patches.py)\n#endif\n"
             "static_assert(VM_NSW >= 2 && VM_NSW <= 3, \"VM_NSW\");\nconstexpr int NS = VM_NSW, NX = VM_NX;\n")
    s = _sub(s, "VM_BRED, VM_DEFER2, VM_PS}; }", "VM_BRED, VM_DEFER2, VM_PS, VM_NSW}; }")   # appended: _vml_cfg()[10] = VM_PS
    s = defs(s, {"bf2n": ("bf2", ("-DVM_NSW=2",)), "w72n": ("w72", ("-DVM_NSW=2",))})
    s = _sub(s, SCX, SCX.replace('"w72")', '"w72", "bf2n", "w72n")'))
    return s


def main():
    mode = sys.argv[1]
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith(f"--{k}=")), d)  # noqa: E731
    base_p = ROOT / arg("base", str(BASE.relative_to(ROOT)))
    s0 = base_p.read_text(encoding="utf-8")
    if mode == "arms":
        AE.build("k3_dsm_1.py", nonce(AE.route(nsw(s0), "bf2", "w72n"),
                                      f"k3-dsm-1 ({base_p.name} + VM_NSW weight-ring depth: bf2n / w72n = NS 2; tools/dsm_patches.py)"))
    elif mode == "rider":
        out_p = arg("out")
        assert out_p, "--out=kernels/...py required"
        Path(ROOT / out_p).write_text(AE.route(nsw(s0), arg("var", "bf2"), arg("bvar", "w72n")), encoding="utf-8", newline="\n")
        print(f"wrote {out_p} (VM_NSW rider on {base_p.name}: var {arg('var', 'bf2')} bvar {arg('bvar', 'w72n')})")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

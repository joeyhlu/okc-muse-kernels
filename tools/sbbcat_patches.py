"""sbbcat_patches.py: wonder_optimization.md Rank 8, "sbb1 -> cat register cap at T = 16" (lab only, default OFF).

Claim: at T = 16 the sbb1 -> cat (GD pooler+router, 256 CTAs x 256 thr, launch_bounds(256, 2) -> 114-128 regs) hop costs
1.2-1.8 us because two cat CTAs fill the 64K RF, so with sbb1 resident at most one cat CTA per SM can enter early.
Candidate (--nr112): the cat instance at T = 16 (gd_kernel CAT == 2, MT == 2) gets launch_bounds(NT + 32, MB) = a 112-reg cap
(9 warps x 2 CTAs x 32 x 112 <= 64K), so 2 cat CTAs (57344 regs) + the sbb1 CTAs (64 thr each) are co-resident, plus
LAT_SBB_TRIG[16] = True (sbb1 launch_dependents at entry; without it cat cannot launch before sbb1 exits at all).
The kernel still launches 256 threads (<= NT + 32); every other gd_kernel instance keeps launch_bounds(NT, MB) text-identically.

Composes onto k3_gc_20 or kernels/lab/k3_l1f_<n>.py (anchored replacements, each must match exactly once).

    python -I tools/sbbcat_patches.py in.py out.py [--nr112]      (no flag: a verbatim copy, for the composing builder)
Default output: kernels/lab/k3_sbbcat_1.py.
"""
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

GD_SIG = "__global__ void __launch_bounds__(WR * WK * 32, MB) gd_kernel(const __grid_constant__ Args A) {"
GD_SIG_NR = ("__global__ void __launch_bounds__(WR * WK * 32 + ((CAT == 2 && MT == 2) ? 32 : 0), MB) "
             "gd_kernel(const __grid_constant__ Args A) {   // SBBCAT: T = 16 cat instance capped at 112 regs")
TRIG = "LAT_SBB_TRIG = {1: True, 8: True}"
TRIG_NR = "LAT_SBB_TRIG = {1: True, 8: True, 16: True}   # SBBCAT: cat CTAs resident before sbb1 exits"


def patch(src, nr112):
    if not nr112:
        return src
    for old, new in ((GD_SIG, GD_SIG_NR), (TRIG, TRIG_NR)):
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"[sbbcat] anchor matched {n}x: {old[:70]}")
        src = src.replace(old, new)
    return src


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    pos = [a for a in argv if not a.startswith("--")]
    src_p = ROOT / (pos[0] if pos else "kernels/muse_decode_layer/k3_gc_20.py")
    out_p = ROOT / (pos[1] if len(pos) > 1 else "kernels/lab/k3_sbbcat_1.py")
    out = patch(src_p.read_text(encoding="utf-8"), "--nr112" in flags)
    out_p.write_text(out, encoding="utf-8", newline="\n")
    py_compile.compile(str(out_p), doraise=True)
    size = out_p.stat().st_size
    if size >= 1_000_000:
        raise SystemExit(f"[sbbcat] {out_p.name} is {size} B >= 1 MB")
    print(f"[sbbcat] {out_p.name}: {size} B, nr112={'--nr112' in flags}")


if __name__ == "__main__":
    main(sys.argv[1:])

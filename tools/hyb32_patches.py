"""hyb32_patches.py (R-hyb32-1): A/B file for HYB (the public alygg77 WGS MoE engine, A32) at T = 17..48, i.e. d32 / v-none.

  python -I tools/hyb32_patches.py [--base=kernels/lab/k3_hh_3rr.py] [--out=kernels/lab/k3_hyb32_1r.py] [--t=49,320]
  python -I tools/slim_patches.py --base=kernels/lab/k3_hyb32_1r.py --out=kernels/lab/k3_hyb32_1.py      (LAST)

Only data changes: HYB_SELF_TEST gains T = 17 / 32 / 32 / 48 cases (alygg77's own file runs this same WGS kernel at
16 < M <= 32, there with -DOKC_B4FIN=1; our HYB build has no B4FIN), so prepare validates the engine at the sizes the A/B
routes to it; --t sets the default HYB_T (49,320 = k3_hh_3 behaviour). A/B arms set HYB_T = [17, 320] at capture time
(_hyb_on reads it per call). If any self-test case fails, HYB is OFF everywhere and the arms are identical (void run).
"""
import ast
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)


def sub(text, old, new):
    n = text.count(old)
    if n != 1:
        sys.exit(f"hyb32_patches: anchor found {n}x: {old[:90]!r}")
    return text.replace(old, new)


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3rr.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_hyb32_1r.py")
    lo, hi = (int(v) for v in ARGS.get("t", "49,320").split(","))
    s = base.read_bytes().decode("ascii")
    lines = s.split(NL)
    lines[1] = f"# nonce k3-hyb32-1r (k3_hh_3rr + R-hyb32-1: HYB self-test at T 17..48, HYB_T default {lo},{hi}; tools/hyb32_patches.py)"
    s = NL.join(lines)
    s = sub(s, "HYB_T = (49, 320)\n", f"HYB_T = ({lo}, {hi})\n")
    s = sub(s, 'HYB_SELF_TEST = ((49, 61, "normal"), ',
            'HYB_SELF_TEST = ((17, 70, "skewed"), (32, 71, "normal"), (32, 72, "skewed"), (48, 73, "normal"), (49, 61, "normal"), ')
    # R-hyb32-1 attempt 1: our VML returned NON-FINITE output on (32, 72, skewed) (3 local experts, 69 routes) while HYB was
    # 2.6e-3 vs FP32. The VML comparison is a sanity bound only: gate HYB on FP32 and log a non-finite VML instead.
    s = sub(s, "                ok = tot < 1e-2 and row < 5e-2 and ev < 1e-2 and rep_d < 1e-3 and bool(torch.isfinite(o.float()).all())\n",
            "                vfin = bool(torch.isfinite(o_v.float()).all())\n"
            "                ok = (tot < 1e-2 and row < 5e-2 and (ev < 1e-2 or not vfin) and rep_d < 1e-3\n"
            "                      and bool(torch.isfinite(o.float()).all()))\n"
            "                if not vfin and k == 0:\n"
            "                    bad = ~torch.isfinite(o_v.float()).all(1)\n"
            "                    print(f\"[hyb32] VML output NON-FINITE at bs={bs} {dist} seed {seed}: {int(bad.sum())} rows \"\n"
            "                          f\"{bad.nonzero().flatten().tolist()[:16]} (HYB vs FP32 {tot:.2e})\", flush=True)\n")
    ast.parse(s)
    out.write_bytes(s.encode("ascii"))
    py_compile.compile(str(out), doraise=True)
    print(f"hyb32_patches: {base.name} -> {out.name} ({len(s):,} B), HYB_T default ({lo}, {hi})")


if __name__ == "__main__":
    main()

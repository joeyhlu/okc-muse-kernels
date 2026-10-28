"""gen_k3_vml9.py: k3_vml_<n> = k3_vf_10 with the VML block swapped for the current lab source (kernels/lab/vml_moe.py v13:
VM_TRICKS consumer by default = the public WGS consumer's per-stage tricks, R-vml-15), VML at 128 / 256 / 32 with
k3_vf_10's flags (193) and tables (R-vml-10 refit + T=128 nt8 262 / 267). --var picks the compile-time variant the file runs
("" = all tricks, "s" = tricks without the opaque-multiply dequant, "o" = the v12 consumer = k3_vf_10's); --abvars builds
more variants in prepare for an in-layer A/B (box copy only). Base must be a finish() fixed point; output goes through
gen_k3_m32.finish. Refuses to overwrite.

    python -I tools/gen_k3_vml9.py --lab-only --abvars=o,s        # kernels/lab/k3_vml_9_ab.py (A/B copy)
    python -I tools/gen_k3_vml9.py --var=s                        # kernels/muse_decode_layer/k3_vml_9.py (+ its box copy)
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAB_ONLY = "--lab-only" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import vml_patches  # noqa: E402

N = ARGS.get("n", "9")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vf_10')}.py"
VAR = ARGS.get("var", "")
ABV = tuple(a for a in ARGS.get("abvars", "").split(",") if a)
OUT = ROOT / f"kernels/muse_decode_layer/k3_vml_{N}.py"
LAB = ROOT / f"kernels/lab/k3_vml_{N}_ab.py"
TAG = f"k3-vml-{N}"
NL = chr(10)
FLAGS = {128: 193, 256: 193, 32: 193}
COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
        128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
        32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}


def main():
    GM.check(not LAB.exists() and (LAB_ONLY or not OUT.exists()), f"{OUT.name} or its lab copy exists; pick another --n")
    GM.check(VAR in ("", "o", "s") and all(a in ("", "o", "s") for a in ABV), "unknown variant")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    for ln, want in (("VML_COST = ", COST), ("VML_FLAGS = ", FLAGS)):
        line = next(x for x in b.split(NL) if x.startswith(ln))
        GM.check(line == ln + repr(want), f"base {ln.strip()} differs")
    nonce = (f"# nonce {TAG} ({BASE.stem} with VML lab v13: the WGS per-stage consumer tricks (VM_TRICKS), variant "
             f"{VAR or 'all'}{', A/B variants ' + '/'.join(ABV) if ABV else ''})")
    t = vml_patches.replace(b.replace(bn, nonce), t32=True, flags=FLAGS, cost=COST, var=VAR, abvars=ABV)
    rep.append(f"VML block -> lab v13, VML_VAR {VAR!r}, VML_ABVARS {ABV}")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_vml9] OK  {r}")
    if not LAB_ONLY:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vml9] wrote {'' if LAB_ONLY else str(OUT.relative_to(ROOT)) + ' and '}{LAB.relative_to(ROOT)} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

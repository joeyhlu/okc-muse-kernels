"""gen_k3_mb.py: k3_mb_<n> = a layer file (default k3_vfa_7) + MoE-board-loop riders measured on the judge GPU
(notes/study/moe_board_loop.md). Rider 1 (k3_mb_1): VML route()'s balanced-job-size countdown loop
`if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }` (warp 0, up to 63 dependent iterations per lane while every
CTA waits; ~1-1.6 us per call at T >= 32, mb_4 / mb_5 bit 10 / 13 on layer-like routings r 0.984 / 0.994 / 0.993) ->
its closed form `if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));`, which this tool asserts equal to the
loop for every (base0, nj, tot) in range before writing (identical jobs, bitwise-identical outputs).
Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_mb.py [--base=k3_vfa_7] [--n=1]
"""
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_vfa_7')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_mb_{N}.py"
TAG = f"k3-mb-{N}"
NL = chr(10)
LOOP = "if (nj) { while (base > 1 && (base - 1) * nj >= tot) --base; }"
CLOSED = ("if (nj && base > 1) base = max(1, min(base, (tot + nj - 1) / nj));   "
          "// == the countdown loop for every (base, nj, tot) (tools/gen_k3_mb.py asserts it); mb bit 13")


def assert_closed_form(bmax=72, njmax=96, totmax=2600):
    """brute force: the loop's result vs the closed form over base0 in [-2, bmax], nj in [0, njmax], tot in [0, totmax]
    (C int semantics; the code has base0 = 64, nj = ceil(tot / 64) <= 40, tot <= 320 x 8)"""
    nj = np.arange(0, njmax + 1, dtype=np.int64)[:, None]
    tot = np.arange(0, totmax + 1, dtype=np.int64)[None, :]
    bad = 0
    for b0 in range(-2, bmax + 1):
        b = np.full((nj.shape[0], tot.shape[1]), b0, dtype=np.int64)
        on = np.broadcast_to(nj != 0, b.shape)
        for _ in range(max(0, b0) + 1):   # the loop runs at most b0 - 1 times
            dec = on & (b > 1) & ((b - 1) * nj >= tot)
            b = np.where(dec, b - 1, b)
        njs = np.where(nj == 0, 1, nj)
        cf = np.where((nj != 0) & (b0 > 1), np.maximum(1, np.minimum(b0, (tot + njs - 1) // njs)), b0)
        bad += int((cf != b).sum())
    GM.check(bad == 0, f"closed form differs from the loop at {bad} points")
    return (bmax + 3) * (njmax + 1) * (totmax + 1)


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    npts = assert_closed_form()
    print(f"[gen_k3_mb] closed form == countdown loop at all {npts:,} (base0, nj, tot) points")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    GM.check(b.count(LOOP) == 1, f"VML countdown loop found {b.count(LOOP)} times in {BASE.name}")
    nonce = (f"# nonce {TAG} ({BASE.stem} + VML route(): balanced job size in closed form instead of the countdown loop, "
             "identical jobs; MoE board loop mb_4 / mb_5)")
    t = b.replace(bn, nonce).replace(LOOP, CLOSED)
    rep.append(f"VML route(): countdown loop -> closed form ({LOOP!r})")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_mb] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_mb] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    # A/B copy (box only): the loop and the closed form as compile-time variants of one VML block (VM_CF), both built in
    # prepare; arms switch VML_VAR 's' (= the base) / 'sc' (= k3_mb_1's VML) at call time
    ab = b.replace(bn, nonce.replace(TAG, TAG + "-ab"))
    ab = ab.replace(LOOP, "#ifndef VM_CF" + NL + "#define VM_CF 0" + NL + "#endif" + NL + "#if VM_CF" + NL + CLOSED + NL
                    + "#else" + NL + LOOP + NL + "#endif")
    old_defs = '_VML_DEFS = {"": (), "o": ("-DVM_TRICKS=0",), "s": ("-DVM_TRICKS_DQ=0",)}'
    GM.check(ab.count(old_defs) == 1 and ab.count(NL + "VML_ABVARS = ()" + NL) == 1 and ab.count(NL + "VML_VAR = 's'" + NL) == 1,
             "VML variant glue of the base differs")
    ab = ab.replace(old_defs, old_defs[:-1] + ', "sc": ("-DVM_TRICKS_DQ=0", "-DVM_CF=1")}')
    ab = ab.replace(NL + "VML_ABVARS = ()" + NL, NL + "VML_ABVARS = ('sc',)" + NL)
    labp = ROOT / f"kernels/lab/k3_mb_{N}_ab.py"
    GM.check(not labp.exists(), f"{labp.name} exists")
    labp.write_text(ab, encoding="utf-8", newline=NL)
    import py_compile
    py_compile.compile(str(labp), doraise=True)
    print(f"[gen_k3_mb] wrote {labp.relative_to(ROOT)} ({len(ab.encode())} B; VML_VAR 's' = base, 'sc' = closed form)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

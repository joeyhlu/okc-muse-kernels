"""gen_k3_v36.py: kernels/muse_decode_layer/k3_v36.py = k3_v35_fa_pt1 + TIE1 + TL1 + K3P (K3B every bucket, K3 grid 4T at
T >= 128).

Composition (each step asserts its anchors hit exactly once in their scope; the steps touch different code paths):
  1. k3_v35_fa_pt1's chain, unstripped (tools/gen_k3_v35.py): k3_v34 -> su1 + su1b + x_4 config -> KW1 -> PR206 (fa206_patches
     minus its nonce) -> PT1.
  2. tools/tie1_patches.py: tie-tolerant top-8 checks in the v8 tail / router32 self-tests (cd12f343 root cause).
  3. tools/tl1_patches.py: study-b's tail_kernel2 (routing warp concurrent with the norm, redux picks; R-study-b-4), default
     on, checked bitwise against tail_kernel in prepare after the (TIE1) tail self-tests, which now run through it.
  4. tools/k3p_patches.py: study-b's K3B in every bucket + K3 grid 4T at buckets 128 / 256 (R-study-b-3).
  (WF, R-explorer-5: killed, not included.)
Then the nonce, tools/strip_layer.py (comments + docstrings only; AST-equivalent; license lines and lines 1-2 kept; kernel
strings untouched) and gen_k3_v35's checks: compiles, ASCII, line 1, nonce, no duplicate names, judge-checker scan <= base,
< 1 MB. Also writes the byte-identical lab copy kernels/lab/k3_v36_ab.py.

    python -I tools/gen_k3_v36.py [--check]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
_argv, sys.argv = sys.argv, [sys.argv[0]]   # gen_k3_v35 reads its flags at import; use it only as a library here
import gen_k3_v35 as G  # noqa: E402
sys.argv = _argv
import fa206_patches  # noqa: E402
import k3p_patches  # noqa: E402
import kw1_patches  # noqa: E402
import pt1_patches  # noqa: E402
import strip_layer  # noqa: E402
import tie1_patches  # noqa: E402
import tl1_patches  # noqa: E402

OUT = ROOT / "kernels/muse_decode_layer/k3_v36.py"
LAB = ROOT / "kernels/lab/k3_v36_ab.py"
NONCE = ("# nonce k3-v36-1 (k3_v35_fa_pt1 [k3_v34 + x_4 levers + KW1 + PR206 + PT1] + TIE1 tie-tolerant top-8 self-tests + "
         "TL1 tail routing warp + K3P: K3B every bucket, K3 grid 4T at T>=128)")


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_k3_v36] FAIL: {msg}")


def generate():
    base = G.BASE.read_text(encoding="utf-8")
    check(hashlib.sha1(base.encode()).hexdigest() == G.BASE_SHA1, "k3_v34 changed (base sha1)")
    rep = []
    text = kw1_patches.apply(G.x4_levers(base))
    check(text.count("KW1_K2_W4 = 5   #") == 1, "KW1 default 5")
    text = fa206_patches.apply(text, [p for p in fa206_patches.PATCHES if p[0] != "nonce"])
    text = pt1_patches.apply(text)
    rep.append("v35_fa_pt1 chain: su1 + su1b + x_4 config + KW1 + PR206 (minus its nonce) + PT1")
    text = tie1_patches.apply(text)
    rep.append(f"TIE1 applied ({len(tie1_patches.PATCHES)} patches hit once)")
    text = tl1_patches.apply(text)
    check(text.count("tail_kernel2(") == 1 and text.count("(2 if TL1_TAIL2 else 0)") == 1 and text.count("[tl1] tail_kernel2 ==") == 1,
          "TL1 anchors")
    rep.append("TL1 applied (tail_kernel2 + 6 host edits + _v8_tail pdl bit + prepare bitwise check + TL1_TAIL2)")
    text = k3p_patches.apply(text)
    check(text.count("k3_kernel_b(const Params P)") == 1 and text.count("| _k3p_bits(m))") == 3, "K3P anchors")
    rep.append("K3P applied (K3B kernel + 3 launches + 3 call sites + module globals)")
    check(text.count(G.NONCE_OLD) == 1, "base nonce line")
    text = text.replace(G.NONCE_OLD, NONCE)
    before = len(text.encode())
    stripped = strip_layer.strip(text)
    ok, msg = strip_layer.equivalent(text, stripped)
    check(ok, f"strip: {msg}")
    lic0 = [l for l in text.splitlines() if strip_layer.LICENSE.search(l)]
    lic1 = [l for l in stripped.splitlines() if strip_layer.LICENSE.search(l)]
    check(lic0 == lic1, "strip: license lines changed")
    rep.append(f"stripped {before} -> {len(stripped.encode())} B; {msg}; {len(lic1)} license lines kept")
    return stripped, base, rep


def verify(text, base):
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
    finally:
        tmp.unlink()
    check(text.isascii(), "non-ASCII character")
    check(text.split("\n", 1)[0] == "# okc: problem=muse_decode_layer", "line 1")
    check(text.count("k3-v36-1") == 1 and "k3-v34-1" not in text and "k3-v35-1" not in text, "nonce")
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
             for n in ast.parse(text).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    check(not dup, f"duplicate top-level names {dup}")
    s0, s1 = G.scan(base), G.scan(text)
    check(all(s1[k] <= s0[k] for k in s0), f"judge-checker scan grew: { {k: (s0[k], s1[k]) for k in s0 if s1[k] > s0[k]} }")
    check(len(text.encode()) < 1_000_000, "file >= 1 MB")
    return [f"compiles, ASCII, line 1, nonce, no duplicate names, checker scan <= base, {len(text.encode())} B"]


def main():
    text, base, rep = generate()
    rep += verify(text, base)
    for r in rep:
        print(f"[gen_k3_v36] OK  {r}")
    if "--check" in sys.argv:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == text, f"{OUT.name} on disk differs (regenerate)")
        print("[gen_k3_v36] --check: up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    LAB.write_text(text, encoding="utf-8", newline="\n")
    print(f"[gen_k3_v36] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} (sha1 {hashlib.sha1(text.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

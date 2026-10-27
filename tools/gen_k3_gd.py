"""gen_k3_gd.py: kernels/muse_decode_layer/k3_gd_<n>.py = k3_v39 + GD (tools/gd_patches.py: gv1's P12 weight-streaming
GEMVs for qkv / o / the bucket-8 pooler / the unpooler at T <= 32), plus the byte-identical A/B copy kernels/lab/k3_gd_<n>_ab.py.

Composition: tools/gen_k3_v37.py --v39 regenerates k3_v39 (asserted byte-identical to the file on disk), gd_patches.apply
(every anchor asserted unique), the nonce, tools/strip_layer.py (comments / docstrings; strings untouched; AST-equivalence
asserted), then gen_k3_v37's checks with this nonce: compiles, ASCII, line 1, nonce, no duplicate top-level names,
judge-checker scan <= base, < 1 MB.

    python -I tools/gen_k3_gd.py [--check]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
CHECK = "--check" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0], "--v39"]   # gen_k3_v37 reads its flags at import
import gen_k3_v37 as G37  # noqa: E402
sys.argv = _argv
import gd_patches  # noqa: E402
import strip_layer  # noqa: E402

N = 1
V39 = ROOT / "kernels/muse_decode_layer/k3_v39.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_gd_{N}.py"
LAB = ROOT / f"kernels/lab/k3_gd_{N}_ab.py"
TAG = f"k3-gd-{N}"
NONCE = (f"# nonce {TAG} (k3_v39 + GD: gv1 P12 weight-streaming GEMVs for qkv / o / unpooler at T <= 32 and the pooler at"
         " T = 2-8, PDL producer loads before griddepcontrol.wait)")


def check(cond, msg):
    if not cond:
        raise SystemExit(f"[gen_k3_gd] FAIL: {msg}")


def main():
    v39, base, rep = G37.generate()
    check(G37.V39 and v39 == V39.read_text(encoding="utf-8"), "gen_k3_v37 --v39 does not reproduce k3_v39.py on disk")
    rep.append(f"k3_v39 reproduced ({len(v39.encode())} B, sha1 {hashlib.sha1(v39.encode()).hexdigest()[:12]})")
    text = gd_patches.apply(v39)
    check(text.count(G37.NONCE) == 1, "k3_v39 nonce line")
    text = text.replace(G37.NONCE, NONCE)
    stripped = strip_layer.strip(text)
    ok, msg = strip_layer.equivalent(text, stripped)
    check(ok, f"strip: {msg}")
    rep.append(f"GD applied, stripped {len(text.encode())} -> {len(stripped.encode())} B; {msg}")
    tmp = OUT.with_suffix(".py.gen_tmp")
    tmp.write_text(stripped, encoding="utf-8", newline="\n")
    try:
        py_compile.compile(str(tmp), doraise=True)
    finally:
        tmp.unlink()
    check(stripped.isascii(), "non-ASCII character")
    check(stripped.split("\n", 1)[0] == "# okc: problem=muse_decode_layer", "line 1")
    check(stripped.split("\n")[1] == NONCE and stripped.count(TAG) == 1 and "k3-v39-1" not in stripped, "nonce")
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(stripped).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    check(not dup, f"duplicate top-level names {dup}")
    s0, s1 = G37.G.scan(base), G37.G.scan(stripped)
    check(all(s1[k] <= s0[k] for k in s0), f"judge-checker scan grew: { {k: (s0[k], s1[k]) for k in s0 if s1[k] > s0[k]} }")
    check(len(stripped.encode()) < 1_000_000, "file >= 1 MB")
    rep.append(f"compiles, ASCII, line 1, nonce, no duplicate names, checker scan <= base, {len(stripped.encode())} B")
    for r in rep:
        print(f"[gen_k3_gd] OK  {r}")
    if CHECK:
        check(OUT.exists() and OUT.read_text(encoding="utf-8") == stripped, f"{OUT.name} on disk differs (regenerate)")
        check(LAB.exists() and LAB.read_text(encoding="utf-8") == stripped, f"{LAB.name} on disk differs (regenerate)")
        print("[gen_k3_gd] --check: up to date")
        return 0
    OUT.write_text(stripped, encoding="utf-8", newline="\n")
    LAB.write_text(stripped, encoding="utf-8", newline="\n")
    print(f"[gen_k3_gd] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} "
          f"(sha1 {hashlib.sha1(stripped.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

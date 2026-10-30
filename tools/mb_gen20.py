"""mb_gen20.py (R-vse): kernels/muse_moe_w4a16/mb_<n>.py = mb_14 + tools/vse_patches.py on the VML source (VSE_HOLD 0 = SASS
identical) + VSE_HOLD arms at buckets 32 / 128 / 256: bf2h1 / bf2h2 / bf2h3 and w72h1 / w72h3 next to the routed bf2 / w72
(ref = bf2 at 32, w72 at 128 / 256; cost tables 'L' as the shipped arms). VMS arms (buckets 1 / 8 / 16) unchanged. Refuses to
overwrite.
    python -I tools/mb_gen20.py [--n=20] [--src=mb_14] [--holds=1,2,3] [--ps=1]   (mb_21: --n=21 --holds= --ps=1)
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vse_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 20))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_14')}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
HOLDS = [int(h) for h in ARGS.get("holds", "1,2,3").split(",") if h]
PS = int(ARGS.get("ps", 0))   # R-vse PS arms bf2ps / w72ps (VSE_PS 1)
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def rep(s, old, new, what):
    assert s.count(old) == 1, f"{what}: {s.count(old)}"
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = P.apply(SRC.read_text(encoding="utf-8"))
    ob, builds = _assign(s, "MB_BUILDS")
    for b in ("bf2", "w72"):
        for h in HOLDS:
            if b == "w72" and h == 2:
                continue
            builds[f"{b}h{h}"] = ("vml", tuple(builds[b][1]) + (f"-DVSE_HOLD={h}",))
    if PS:
        for b in ("bf2", "w72"):
            builds[f"{b}ps"] = ("vml", tuple(builds[b][1]) + ("-DVSE_PS=1",))
    oa, arms = _assign(s, "MB_ARMS")
    hb = [f"bf2h{h}:193:L" for h in HOLDS]
    hw = [f"w72h{h}:193:L" for h in HOLDS if h != 2]
    arms[32] = ["bf2:193:L"] + hb + ["w72:193:L"] + hw
    arms[128] = ["w72:193:L"] + hw + ["bf2:193:L"] + hb
    arms[256] = ["w72:193:L"] + hw + ["bf2:193:L"] + ([f"bf2h{HOLDS[-1]}:193:L"] if HOLDS else [])
    if PS:
        arms[32] += ["bf2ps:193:L", "w72ps:193:L"]
        arms[128] += ["w72ps:193:L", "bf2ps:193:L"]
        arms[256] += ["w72ps:193:L", "bf2ps:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = rep(s, "MB_N = 14\n", f"MB_N = {N}\n", "MB_N")
    s = s.replace('"""mb_14 (', f'"""mb_{N} (tools/mb_gen20.py: mb_14 + tools/vse_patches.py; VSE_HOLD arms {HOLDS}, VSE_PS arms {bool(PS)} at 32 / 128 / 256). mb_14 (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen20] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms 32 {arms[32]}")


if __name__ == "__main__":
    main()

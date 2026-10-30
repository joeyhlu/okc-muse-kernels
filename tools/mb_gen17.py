"""mb_gen17.py: kernels/muse_moe_w4a16/mb_<n>.py = <src> (mb_16: w112 arms, or mb_15) + tools/vse_patches.py on the VML source
(VSE_PS / VSE_HOLD 0 preprocess to the base) + build bf2ps (= bf2 + -DVSE_PS=1: phase-A private-partial split shares; parent's
R-vse: mb_21 bs32 r 0.9934 +- 0.0012, neutral at 128 / 256) as an arm at bucket 32 only. The other buckets keep <src>'s arms.
Refuses to overwrite.
    python -I tools/mb_gen17.py [--n=17] [--src=mb_16]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vse_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 17))
SRCN = ARGS.get("src", "mb_16")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = P.apply(SRC.read_text(encoding="utf-8"))
    ob, builds = _assign(s, "MB_BUILDS")
    builds["bf2ps"] = ("vml", tuple(builds["bf2"][1]) + ("-DVSE_PS=1",))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2:193:L", "bf2ps:193:L", "w72:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen17.py: {SRCN} + tools/vse_patches.py, VSE_PS arm bf2ps at bucket 32). {SRCN} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen17] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

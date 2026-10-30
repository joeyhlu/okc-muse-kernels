"""mb_gen41.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_40 + tools/vud_patches.py on the VML source (VM_UD2 0 = SASS identical) with
the killed w72ff build dropped and REDUX-uniform x-descriptor builds w72ud / bf2psud (+ -DVM_UD2=3; local sm_90a: R2UR 928 -> 76
(w72) / 820 -> 70 (bf2ps), static instr -240 / -296, spills unchanged) as arms: 32 bf2ps / bf2psud, 128 / 256 w72 / w72ud (all
:193:L). Board routings carry hot nt 5-8 jobs at 128 / 256 (compute-bound stages: R-gap-0 candidate 1). Refuses to overwrite.
    python -I tools/mb_gen41.py [--n=41] [--src=mb_40]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vud_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 41))
SRCN = ARGS.get("src", "mb_40")
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
    builds.pop("w72ff", None)
    builds["w72ud"] = ("vml", tuple(builds["w72"][1]) + ("-DVM_UD2=3",))
    builds["bf2psud"] = ("vml", tuple(builds["bf2ps"][1]) + ("-DVM_UD2=3",))
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2ps:193:L", "bf2psud:193:L"]
    arms[128] = ["w72:193:L", "w72ud:193:L"]
    arms[256] = ["w72:193:L", "w72ud:193:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen41.py: {SRCN} + vud REDUX-uniform x descriptor arms at 32 / 128 / 256).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen41] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

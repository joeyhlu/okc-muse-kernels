"""mb_gen25.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_24 + tools/vpre_patches.py on the layer-VMS source, with the killed builds
(lvn4 / lvn5 / lvf1n5: mb_24 +1.5..+4.6 %) dropped and shallower-start arms: lvn2 (VMS_NS 2) and lvp2 / lvf1p2 (NS 3, VMS_PRE 2:
two prologue copies, the third after unit 0 landed). VML arms trimmed to the live ones (fitted 'f' tables killed in mb_23):
32 bf2ps:193:L / bf2ps:192:L / bf2:193:L, 128 / 256 w72:193:L / w72:192:L. Refuses to overwrite.
    python -I tools/mb_gen25.py [--n=25] [--src=mb_24]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vpre_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 25))
SRCN = ARGS.get("src", "mb_24")
SRC = ROOT / f"kernels/muse_moe_w4a16/{SRCN}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)
LVQ = "MB_LV_CUDA = r'''"


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    a = s.index(LVQ) + len(LVQ)
    b = s.index("'''", a)
    s = s[:a] + P.apply(s[a:b]) + s[b:]
    ob, builds = _assign(s, "MB_BUILDS")
    for k in ("lvn4", "lvn5", "lvf1n5"):
        builds.pop(k)
    builds["lvn2"] = ("lvms", ("-DVMS_NS=2",))
    builds["lvp2"] = ("lvms", ("-DVMS_PRE=2",))
    builds["lvf1p2"] = ("lvms", ("-DVMS_FX=1", "-DVMS_PRE=2"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lv", "lvf1", "lvn2", "lvp2", "lvf1p2"]
    arms[8] = ["lv", "lvf1", "lvn2", "lvp2"]
    arms[16] = ["lv", "lvn2", "lvp2"]
    arms[32] = ["bf2ps:193:L", "bf2ps:192:L", "bf2:193:L"]
    arms[128] = ["w72:193:L", "w72:192:L"]
    arms[256] = ["w72:193:L", "w72:192:L"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen25.py: {SRCN} + vpre; VMS NS 2 / PRE 2 arms at 1 / 8 / 16, live VML arms).'
                  f'{NL}{SRCN} (', 1)   # new line: the lineage line is near the judge's 1000-char limit
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen25] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

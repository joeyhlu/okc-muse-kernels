"""mb_gen26.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_25 + tools/vlc_patches.py on the layer-VMS source (VMS_LC 1: at M == 1 the
CTA completing the arrival generation resets the counters and runs the whole 4096-column combine, the others exit after their add;
local gate tools/vlc_local_test.py: sm_90a defaults SASS identical, LC 142 regs 0 spill; RTX 3070 63 calls incl. graph replays
within tolerance, generation / arrival counters coherent). The killed NS 2 build (lvn2, mb_25 +1.2..+1.7 %) is dropped.
Arms 1: lvf1 (ref) / lvf1lc / lvf1p2 / lvf1p2lc / lv, 8: lv / lvf1 / lvp2, 16: lv / lvp2. VML buckets keep mb_25's. No overwrite.
    python -I tools/mb_gen26.py [--n=26] [--src=mb_25]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vlc_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 26))
SRCN = ARGS.get("src", "mb_25")
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
    builds.pop("lvn2")
    builds["lvf1lc"] = ("lvms", ("-DVMS_FX=1", "-DVMS_LC=1"))
    builds["lvf1p2lc"] = ("lvms", ("-DVMS_FX=1", "-DVMS_PRE=2", "-DVMS_LC=1"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lvf1", "lvf1lc", "lvf1p2", "lvf1p2lc", "lv"]
    arms[8] = ["lv", "lvf1", "lvp2"]
    arms[16] = ["lv", "lvp2"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen26.py: {SRCN} + vlc last-arriver finish arms at bucket 1).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen26] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

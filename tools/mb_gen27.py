"""mb_gen27.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_26 + tools/vhp_patches.py on the layer-VMS source (VMS_HP 1: the helper loads
unit q + 1's epilogue constants gsA / gsB / nw before unit q's pfull wait; bitwise equal; local gate tools/vlc_local_test.py
patch=vhp: sm_90a defaults SASS identical, HP 143 regs 0 spill, RTX 3070 within tolerance + counters coherent), the killed VMS_LC
builds dropped (mb_26 b1 lvf1lc r 1.0120 +- 0.0010), and MB_SEEDS 6 -> 16 (small-bucket sweeps: mb_25 / mb_26 b1 calls of
-1 % / +0.8 % for the same arm were inside the 6-routing noise; the tuner's per-bucket pick uses the same set).
Arms 1: lv / lvf1 / lvhp / lvf1hp, 8: lv / lvhp / lvp2, 16: lv / lvhp. VML buckets keep mb_26's. Refuses to overwrite.
    python -I tools/mb_gen27.py [--n=27] [--src=mb_26]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import vhp_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 27))
SRCN = ARGS.get("src", "mb_26")
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


def _rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    a = s.index(LVQ) + len(LVQ)
    b = s.index("'''", a)
    s = s[:a] + P.apply(s[a:b]) + s[b:]
    ob, builds = _assign(s, "MB_BUILDS")
    for k in ("lvf1lc", "lvf1p2lc", "lvf1p2"):
        builds.pop(k)
    builds["lvhp"] = ("lvms", ("-DVMS_HP=1",))
    builds["lvf1hp"] = ("lvms", ("-DVMS_FX=1", "-DVMS_HP=1"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lv", "lvf1", "lvhp", "lvf1hp"]
    arms[8] = ["lv", "lvhp", "lvp2"]
    arms[16] = ["lv", "lvhp"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = _rep(s, "MB_SEEDS = 6         # board-like timing routings per bucket (T <= 16)",
             "MB_SEEDS = 16        # board-like timing routings per bucket (T <= 16; mb_gen27: 6 -> 16, b1 arms were inside the noise)")
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen27.py: {SRCN} + vhp helper-constant prefetch arms, 16 small-bucket '
                  f'routings).{NL}mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen27] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

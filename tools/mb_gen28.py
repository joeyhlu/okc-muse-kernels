"""mb_gen28.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_27 with the killed VMS arms dropped (lvhp / lvf1hp: mb_27 b8 / b16 r 1.0070 /
1.0099 at 16 routings; lvp2: noise) and in-prepare tuned cost tables 't' (_mb_tune: greedy per-entry steps from the bucket's 'L'
table on MB_TUNE_N = 24 board routings separate from the timing set, kept only when the mean drops > 0.3 %, 12 s budget) as arms
at the VML buckets; MB round 1 tuned on 4 routings and overfit. Arms 1: lvf1 / lv, 8: lv / lvf1, 16: lv; 32: bf2ps:193:L /
bf2ps:192:L / bf2ps:193:t; 128 / 256: w72:193:L / w72:192:L / w72:193:t. Refuses to overwrite.
    python -I tools/mb_gen28.py [--n=28] [--src=mb_27]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 28))
SRCN = ARGS.get("src", "mb_27")
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
    s = SRC.read_text(encoding="utf-8")
    ob, builds = _assign(s, "MB_BUILDS")
    for k in ("lvhp", "lvf1hp", "lvp2"):
        builds.pop(k)
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lvf1", "lv"]
    arms[8] = ["lv", "lvf1"]
    arms[16] = ["lv"]
    arms[32] = ["bf2ps:193:L", "bf2ps:192:L", "bf2ps:193:t"]
    arms[128] = ["w72:193:L", "w72:192:L", "w72:193:t"]
    arms[256] = ["w72:193:L", "w72:192:L", "w72:193:t"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""mb_{SRCN[3:]} (', f'"""mb_{N} (tools/mb_gen28.py: {SRCN} - killed VMS arms + tuned tables t at 32 / 128 / 256).{NL}'
                  f'mb_{SRCN[3:]} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen28] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

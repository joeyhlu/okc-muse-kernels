"""mb_gen24.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_23 with the killed vxp builds (lvl2 / lvxp / lvf1l2 / lvf1xp: mb_22 b8 / b16
+0.9 % and +2.7 %) dropped and compile-time ring-depth arms on the layer VMS: VMS_NS 4 / 5 (36 KB unit slots; lvms smem 129 KB
at NS 3 -> ~166 / ~203 KB). R-vms-3's ring 4 / 5 result was an invalid A/B (runtime-flag codegen); mb_22's bs1 marks show a 1.9 us
A -> B hop (bw0 15.8 -> bw1 17.7) and a 2.5 us ramp, where deeper B prefetch could keep HBM busy. Arms 1: lv / lvaq / lvf1 / lvn4 /
lvn5 / lvf1n5, 8: lv / lvf1 / lvn4 / lvn5 / lvf1n5, 16: lv / lvn4 / lvn5. VML buckets keep mb_23's arms. Refuses to overwrite.
    python -I tools/mb_gen24.py [--n=24] [--src=mb_23]
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 24))
SRCN = ARGS.get("src", "mb_23")
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
    for b in ("lvl2", "lvxp", "lvf1l2", "lvf1xp"):
        builds.pop(b)
    builds["lvn4"] = ("lvms", ("-DVMS_NS=4",))
    builds["lvn5"] = ("lvms", ("-DVMS_NS=5",))
    builds["lvf1n5"] = ("lvms", ("-DVMS_FX=1", "-DVMS_NS=5"))
    oa, arms = _assign(s, "MB_ARMS")
    arms[1] = ["lv", "lvaq", "lvf1", "lvn4", "lvn5", "lvf1n5"]
    arms[8] = ["lv", "lvf1", "lvn4", "lvn5", "lvf1n5"]
    arms[16] = ["lv", "lvn4", "lvn5"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    m = re.search(r"\nMB_N = (\d+)\n", s)
    s = s.replace(m.group(0), f"{NL}MB_N = {N}{NL}")
    s = s.replace(f'"""{SRCN} (', f'"""mb_{N} (tools/mb_gen24.py: {SRCN} - vxp builds + VMS ring-depth arms NS 4 / 5 at 1 / 8 / 16). '
                  f'{SRCN} (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen24] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"builds {list(builds)}; arms {arms}")


if __name__ == "__main__":
    main()

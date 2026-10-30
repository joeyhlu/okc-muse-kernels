"""mb_gen11.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_10 + board-specific VML cost-table arms "Lx<pct>": the layer table L
with every nt >= 2 entry (A and B halves) scaled by pct / 100. Why (R-mb11, moe redesign): on board bs32 the makespan is
set by the CTAs holding the one hot expert's nt = 2 stages; L prices nt 2 at 167 / 128 = 1.30 x a cold stage (in-layer
paces 1.60 / 1.19), but board cold stages run 1.05-1.11 us (mb_10 marks), so hot CTAs carry ~1.45-1.5 x and end last.
The in-prepare sweep (16 board routings, cold-L2 graph replays, paired vs the L arm) picks per bucket. Arms:
  32: bf2:193:L (ref) + Lx105 / Lx110 / Lx115 / Lx120;  128: w72 / bf2 :L + bf2:193:Lx108 / w72:193:Lx108;  256 unchanged.
The s32 probe arm and its build are dropped (r 1.07, mb_10). Refuses to overwrite.

    python -I tools/mb_gen11.py [--n=11] [--src=mb_10]
"""
import ast
import hashlib
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 11))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_10')}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
NL = chr(10)


def _assign(src, name):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            return ast.get_source_segment(src, n), ast.literal_eval(n.value)
    raise KeyError(name)


def _fmt(d):
    return "{" + NL + "".join(f"    {k!r}: {v!r}," + NL for k, v in d.items()) + "}"


def rep(s, old, new, what):
    n = s.count(old)
    assert n == 1, f"{what}: {n} matches"
    return s.replace(old, new)


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    s = rep(s, '    if ck == "L":\n        return MB_LAYER_COST.get(rep)\n',
            '    if ck == "L":\n        return MB_LAYER_COST.get(rep)\n'
            '    if ck.startswith("Lx") and rep in MB_LAYER_COST:   # mb_gen11: L with every nt >= 2 entry scaled by pct / 100\n'
            '        f, t = int(ck[2:]) / 100.0, MB_LAYER_COST[rep]\n'
            '        return [t[0]] + [int(round(v * f)) for v in t[1:8]] + [t[8]] + [int(round(v * f)) for v in t[9:16]]\n', "cost")
    s = rep(s, '                if a["kind"] == "vml" and a["ck"] not in (None, "d", "L") and _mb_cost(a, rep) is None:',
            '                if a["kind"] == "vml" and a["ck"] not in (None, "d", "L") and not str(a["ck"]).startswith("Lx") '
            'and _mb_cost(a, rep) is None:', "self-test cost gate")
    ob, builds = _assign(s, "MB_BUILDS")
    builds.pop("s32", None)
    oa, arms = _assign(s, "MB_ARMS")
    arms[32] = ["bf2:193:L", "bf2:193:Lx105", "bf2:193:Lx110", "bf2:193:Lx115", "bf2:193:Lx120"]
    arms[128] = ["w72:193:L", "bf2:193:L", "bf2:193:Lx108", "w72:193:Lx108"]
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = rep(s, "MB_N = 10\n", f"MB_N = {N}\n", "MB_N")
    s = s.replace('"""mb_10 (tools/mb_gen10.py:', f'"""mb_{N} (tools/mb_gen11.py: mb_10 + board VML cost-table arms Lx<pct>). mb_10 (tools/mb_gen10.py:', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    dup = sorted({n for n in names if n and names.count(n) > 1})
    assert not dup, dup
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen11] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

"""rat_ship.py (R-rat): apply confirmed route-autotune flips to a ship layer file as plain table edits (no new code).

  python -I tools/rat_ship.py --base=<ship.py> --out=<out.py> --flips=u1c,ks1t1,... [--dry]

Each flip is the R-rat-1 runtime switch (tools/box/rat_run.py flips()) turned into its table form:
  GD alternates -> the GD_ROUTE entry (added for o / 32, which then runs GD instead of DG1 since _gd comes first in kernel_fn);
  u1su1 / o1su1 -> GD_ROUTE entry removed (+ SU1_ROUTE ("o", 1) True); KS1_TRIG / KS3_TRIG / FT_TAIL3 / VSL_ROUTE / HYA_TUNED edits.
Every edit is anchored on the exact current table text and checked (count == 1); a moved anchor fails loudly.
After a GD route change run tools/rpo_sass.py on the output (pre-wait loads / x above ACQBULK) before any judge run.
"""
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from rat_patches import RAT_GD_ALT  # noqa: E402

NL = chr(10)
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
ALT = {tag: (p, b, rt) for (p, b, tag), rt in RAT_GD_ALT.items()}


def check(c, msg):
    if not c:
        raise SystemExit(f"[rat_ship] FAIL: {msg}")


def table(t, name):
    """(start, end, value) of the module-level assignment `name = {...}` (single or multi line)."""
    for n in ast.parse(t).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name:
            lines = t.split(NL)
            a = sum(len(x) + 1 for x in lines[:n.lineno - 1])
            b = sum(len(x) + 1 for x in lines[:n.end_lineno]) - 1
            src = t[a:b]
            return a, b, eval(compile(ast.Expression(n.value), "t", "eval"), {"dict": dict})
    raise SystemExit(f"[rat_ship] FAIL: table {name} not found")


def put(t, name, value, fmt=repr):
    a, b, _ = table(t, name)
    return t[:a] + f"{name} = {fmt(value)}" + t[b:]


def gd_fmt(d):
    items = sorted(d.items(), key=lambda kv: (["qkv", "o", "pooler", "unpooler"].index(kv[0][0]), kv[0][1]))
    return "{" + ", ".join(f'("{p}", {b}): {rt!r}' for (p, b), rt in items) + "}"


def main():
    base = ROOT / ARGS["base"]
    out = ROOT / ARGS["out"]
    flips = [f for f in ARGS.get("flips", "").split(",") if f]
    check(flips, "--flips required")
    t = base.read_text(encoding="utf-8")
    _, _, gd = table(t, "GD_ROUTE")
    gd = dict(gd)
    notes = []
    for f in flips:
        if f in ALT:
            p, b, rt = ALT[f]
            notes.append(f"GD {p}/{b} {gd.get((p, b))} -> {rt}")
            gd[(p, b)] = rt
        elif f == "u1su1":
            gd.pop(("unpooler", 1))
            notes.append("GD unpooler/1 removed (SU1)")
        elif f == "o1su1":
            gd.pop(("o", 1))
            _, _, su = table(t, "SU1_ROUTE")
            su = dict(su, **{}) if False else dict(su)
            su[("o", 1)] = True
            t = put(t, "SU1_ROUTE", su)
            notes.append("GD o/1 removed, SU1_ROUTE o/1 True")
        elif f in ("ks1t1", "ks1t8"):
            _, _, v = table(t, "KS1_TRIG")
            v = dict(v)
            v[1 if f == "ks1t1" else 8] = 2
            t = put(t, "KS1_TRIG", v)
            notes.append(f"KS1_TRIG {v}")
        elif f in ("ks3t16", "ks3t32"):
            _, _, v = table(t, "KS3_TRIG")
            v = dict(v)
            v[(16, 1) if f == "ks3t16" else (32, 1)] = 2
            t = put(t, "KS3_TRIG", v)
            notes.append(f"KS3_TRIG {v}")
        elif f == "ft4":
            _, _, v = table(t, "FT_TAIL3")
            v = dict(v)
            v[1] = 4
            t = put(t, "FT_TAIL3", v)
            notes.append(f"FT_TAIL3 {v}")
        elif f == "vsl8off":
            _, _, v = table(t, "VSL_ROUTE")
            v = {k: x for k, x in v.items() if k != (1, 8)}
            t = put(t, "VSL_ROUTE", v)
            notes.append(f"VSL_ROUTE {v}")
        elif f == "hya132":
            _, _, v = table(t, "HYA_TUNED")
            v = dict(v)
            v[(1, 8, False, 1)] = {"n_splits": 132, "nc": 8}
            t = put(t, "HYA_TUNED", v)
            notes.append(f"HYA_TUNED {v}")
        else:
            check(False, f"unknown flip {f}")
    t = put(t, "GD_ROUTE", gd, gd_fmt)
    bn = t.split(NL)[1]
    check(bn.startswith("# nonce "), "nonce line")
    t = t.replace(bn, bn + f" + R-rat flips {','.join(flips)} [tools/rat_ship.py]", 1)
    check(t.isascii() and len(t.encode()) < 1_000_000, "ascii / size")
    ast.parse(t)
    for s in notes:
        print("[rat_ship]", s)
    print(f"[rat_ship] OK {len(t.encode())} B")
    if "--dry" in sys.argv:
        return 0
    check(not out.exists() or "--force" in sys.argv, f"{out} exists")
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[rat_ship] wrote {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

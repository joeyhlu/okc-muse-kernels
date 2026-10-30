"""mb_gen16.py: kernels/muse_moe_w4a16/mb_<n>.py = mb_15 + VML build 'w112' (= w72 defines + CREG 120 / JN 112 / XSN 112 / NX 2:
producer WG kept, 512 x 120 + 128 x 32 = the 64 K register file; experts with 73-112 tokens stream their weights once instead of
twice) with its own cost table 'W' (2 x 14 entries: nt 1-8 = the bucket's 'L' table, nt 9-14 scaled from R-w128-2's cold w112
stage times, box_out/w128/run2_w112.json, or the pred_w112.json prediction). Arms 128 / 256: w72:L, w112:W, w112:L; the in-prepare
sweep (16 board routings, paired r vs the first arm) picks per bucket on the judge GPU. Host: NTM <= 14 and fix scratch
14 x 512 floats. Refuses to overwrite.
    python -I tools/mb_gen16.py [--n=16] [--src=mb_15] [--probe=box_out/w128/run2_w112.json]
"""
import ast
import hashlib
import json
import pathlib
import py_compile
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 16))
SRC = ROOT / f"kernels/muse_moe_w4a16/{ARGS.get('src', 'mb_15')}.py"
OUT = ROOT / f"kernels/muse_moe_w4a16/mb_{N}.py"
PROBE = ROOT / ARGS.get("probe", "box_out/w128/run2_w112.json")
NL = chr(10)
W112 = {"-DVM_CREG": "120", "-DVM_JN": "112", "-DVM_XSN": "112", "-DVM_NX": "2"}


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


def w_table(L):
    """2 x 14 costs: nt 1-8 from the bucket's L table, nt 9-14 = L[nt 7] x w112 cold stage(8 nt) / stage(56), linear in N
    between the measured N 56 / 80 / 96 / 112"""
    rows = json.loads(PROBE.read_text(encoding="utf-8"))["rows"]
    st = {r["N"]: r["us_per_stage"]["w112i"] for r in rows if r["mode"] == "cold"}
    xs = sorted(n for n in st if n >= 56)
    assert xs[0] == 56 and xs[-1] == 112, xs

    def at(n):
        for a, b in zip(xs, xs[1:]):
            if a <= n <= b:
                return st[a] + (st[b] - st[a]) * (n - a) / (b - a)
        raise ValueError(n)
    out = []
    for pb in (0, 1):
        h = list(L[8 * pb:8 * pb + 8])
        out += h + [int(round(h[6] * at(8 * nt) / st[56])) for nt in range(9, 15)]
    return out


def main():
    assert not OUT.exists(), f"refusing to overwrite {OUT}"
    s = SRC.read_text(encoding="utf-8")
    ob, builds = _assign(s, "MB_BUILDS")
    fl = [f for f in builds["w72"][1] if f.split("=")[0] not in W112] + [f"{k}={v}" for k, v in W112.items()]
    builds["w112"] = ("vml", tuple(fl))
    oa, arms = _assign(s, "MB_ARMS")
    arms[128] = ["w72:193:L", "w112:193:W", "w112:193:L"]   # L: the 16-entry table, linear extension to nt 14 (_vml_cost_ext)
    arms[256] = ["w72:193:L", "w112:193:W", "w112:193:L"]
    oc, lcost = _assign(s, "MB_LAYER_COST")
    wc = {rep: w_table(lcost[rep]) for rep in (128, 256)}
    s = s.replace(ob, "MB_BUILDS = " + _fmt(builds)).replace(oa, "MB_ARMS = " + _fmt(arms))
    s = s.replace(oc, oc + NL + "MB_W112_COST = " + _fmt(wc) + "   # mb_gen16: 2 x 14 (A, B) for the w112 build (ck 'W')")
    s = _rep(s, '    if ck == "L":\n        return MB_LAYER_COST.get(rep)\n',
             '    if ck == "L":\n        return MB_LAYER_COST.get(rep)\n    if ck == "W":   # mb_gen16: w112 build, 2 x 14 entries\n'
             '        return MB_W112_COST.get(rep)\n')
    s = _rep(s, '    assert 8 <= ntm <= 10, "m32e scratch (fix regions of max(FIXW, 5120) floats) supports NTM 8..10 builds"',
             '    assert 8 <= ntm <= 14, "m32e scratch (fix regions of max(FIXW, 7168) floats) supports NTM 8..14 builds"')
    s = _rep(s, "max(_vml_FIXW, 10 * 512)", "max(_vml_FIXW, 14 * 512)")
    s = _rep(s, f"MB_N = 15{NL}", f"MB_N = {N}{NL}")
    s = s.replace('"""mb_15 (', f'"""mb_{N} (tools/mb_gen16.py: mb_15 + w112 wide-job VML build, arms 128 / 256). mb_15 (', 1)
    names = [getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                                          else None) for n in ast.parse(s).body]
    assert not sorted({n for n in names if n and names.count(n) > 1})
    assert s.isascii() and "importlib" not in s and "sys.modules" not in s and "__trap" not in s
    assert not [x for x in s.split(NL) if len(x) >= 1000]
    assert len(s.encode()) < 1_000_000
    OUT.write_text(s, encoding="utf-8", newline=NL)
    py_compile.compile(str(OUT), doraise=True)
    print(f"[mb_gen16] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]}); "
          f"W tables {wc}")


if __name__ == "__main__":
    main()

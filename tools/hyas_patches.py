"""hyas_patches.py (R-hyas-1): route vjeux's public HYA attention kernel (vendored with its Apache-2.0 notice, see
_HYA_NOTICE in the file) at the q_len 1 decode shapes it is NOT routed to today, ahead of our KS1 / KS3 / AX1.

  python -I tools/hyas_patches.py --add=s1[,s16,s32,f1] [--base=kernels/lab/k3_hh_3tekv_hyblo18.py] [--out=<new.py>]
  python -I tools/hyas_patches.py --arms=<arms.json> [--base=...]     # rev_ab3 arms (no file written)
  python -I tools/hyas_patches.py --check [--base=...]                # $0 support audit only

Keys: s1 = ('sliding', 1, 1) (d1s, KS1 today), s16 = ('sliding', 16, 1) (d16s, KS3), s32 = ('sliding', 32, 1) (d32s, KS3),
f1 = ('full', 1, 1) (d1f, AX1 fb1_l1). The patch is ONE line: the `HYA_ROUTE = {...}` assignment gains the listed keys
(True = HYA first, before AX1 / KS1). No other byte changes (HYA_CUDA_SRC / HYA_CPP_SRC untouched -> same extension hash,
same build). Route keys are (kind, batch, q_len) and every arm key belongs to exactly one bench shape, so a single "union" arm
(all four keys) measured at shape X is the separate arm for X; the --arms JSON uses that (ship / hyas / mid1 / hyas_b / mid2).

Audit (--check, also run before every write): the HYA_ROUTE line exists once, each key is new, `_hya_wanted` accepts q_len 1,
the CUDA dispatch has the L == 1 template (launch<1>), each key has a prepare self-test case (HYA_SELF_TEST, same B / q_len /
sliding-or-full, page 1), and _hya_plan (exec'd from the file) gives a valid plan at the bench shape (H200, 132 SMs, kv pairs 1,
window 2048 / ctx 80000 page 1) with HYA_TUNED applied as _hya_launch does.
"""
import ast
import json
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "kernels/lab/k3_hh_3tekv_hyblo18.py"
LIMIT = 1_000_000
KEYS = {"s1": ("sliding", 1, 1), "s16": ("sliding", 16, 1), "s32": ("sliding", 32, 1), "f1": ("full", 1, 1)}
ROUTE_RE = re.compile(r"^HYA_ROUTE = (\{.*\})$", re.M)


def arg(name, default=None):
    for a in sys.argv[1:]:
        if a == f"--{name}":
            return True
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def fail(msg):
    print(f"[hyas] FAIL: {msg}")
    sys.exit(1)


def route_of(text):
    m = list(ROUTE_RE.finditer(text))
    if len(m) != 1:
        fail(f"{len(m)} HYA_ROUTE assignment lines (want 1)")
    return m[0], ast.literal_eval(m[0].group(1))


def audit(text, keys):
    _, route = route_of(text)
    for k in keys:
        if k in route:
            fail(f"{k} already in HYA_ROUTE ({route[k]})")
    if "if (L == 1) launch<1>(p, grid, stream, (int)cs, pdl);" not in text:
        fail("no L == 1 template dispatch in HYA_CPP_SRC")
    if "L in (1, 8)" not in text:
        fail("_hya_wanted does not accept q_len 1")
    st = ast.literal_eval(re.search(r"^HYA_SELF_TEST = (\(.*\))$", text, re.M).group(1))
    tuned = eval(re.search(r"^HYA_TUNED = (\{.*\})$", text, re.M).group(1), {"__builtins__": {}, "dict": dict})
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_hya_plan")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "_hya_plan", "exec"), ns)
    plan = ns["_hya_plan"]
    for k in keys:
        kind, B, L = k
        win = 2048 if kind == "sliding" else 0
        cases = [c for c in st if c[0] == B and c[2] == L and bool(c[3]) == bool(win) and c[6] == 1]
        if not cases:
            fail(f"{k}: no HYA_SELF_TEST case at B {B} q_len {L} {'sliding' if win else 'full'} page 1")
        p = plan(B, 1, L, win, 80000 + L, 1, 132)
        ov = tuned.get((B, L, bool(win), 1))
        if ov:
            p = dict(p, **ov)
        if not (p["n_splits"] >= 1 and p["nc"] >= 1):
            fail(f"{k}: bad plan {p}")
        print(f"[hyas] {k}: self-test cases {[c[:4] for c in cases]}; plan {p}{' (HYA_TUNED)' if ov else ''}")
    return route


def patched(text, keys):
    m, route = route_of(text)
    new = dict(route)
    for k in keys:
        new[k] = True
    out = text[:m.start()] + f"HYA_ROUTE = {new!r}" + text[m.end():]
    a, b = text.splitlines(), out.splitlines()
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(a) != len(b) or len(diff) != 1:
        fail(f"diff not one line: {len(diff)} lines, {len(a)} vs {len(b)}")
    return out, new


def main():
    base = pathlib.Path(arg("base", str(BASE)))
    text = base.read_bytes().decode("utf-8")
    if "\r\n" in text:
        fail("base has CRLF")
    names = [x for x in (arg("add", "") or "").split(",") if x]
    for n in names:
        if n not in KEYS:
            fail(f"unknown key {n} (want {sorted(KEYS)})")
    if arg("check"):
        audit(text, list(KEYS.values()))
        print("[hyas] audit PASS")
        return
    if arg("arms"):
        route = audit(text, list(KEYS.values()))
        _, union = patched(text, list(KEYS.values()))
        enc = lambda r: {",".join(map(str, k)): v for k, v in r.items()}  # noqa: E731  (rev_ab3 keyfix decodes)
        arms = [{"name": "ship", "set": {"HYA_ROUTE": enc(route)}}, {"name": "hyas", "set": {"HYA_ROUTE": enc(union)}},
                {"name": "mid1", "set": {"HYA_ROUTE": enc(route)}}, {"name": "hyas_b", "set": {"HYA_ROUTE": enc(union)}},
                {"name": "mid2", "set": {"HYA_ROUTE": enc(route)}}]
        pathlib.Path(arg("arms")).write_bytes((json.dumps(arms, indent=1) + "\n").encode())
        print(f"[hyas] arms -> {arg('arms')}: {[a['name'] for a in arms]}")
        return
    if not names:
        fail("--add=<keys> or --arms=<json> or --check")
    keys = [KEYS[n] for n in names]
    audit(text, keys)
    out_text, new = patched(text, keys)
    n = len(out_text.encode())
    ast.parse(out_text)
    if not out_text.isascii():
        fail("non-ascii")
    if n >= LIMIT:
        fail(f"size {n} B >= cap {LIMIT}")
    out = pathlib.Path(arg("out", str(base.with_name(base.stem + "_hyas_" + "_".join(names) + ".py"))))
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "k.py"
        tmp.write_bytes(out_text.encode())
        py_compile.compile(str(tmp), doraise=True)
    out.write_bytes(out_text.encode())
    print(f"[hyas] {out}: {n:,} B ({LIMIT - n:,} B under the cap, {n - len(text.encode()):+d} B vs base); "
          f"HYA_ROUTE += {keys}; py_compile ok")


if __name__ == "__main__":
    main()

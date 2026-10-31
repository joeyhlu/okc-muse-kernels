"""rat_patches.py (R-rat, route-autotune builder, 2026-10-31): one lab A/B file where every route flip is a run-time switch.

  python -I tools/rat_patches.py [--base=kernels/lab/k3_hh_3tekv.py] [--out=kernels/lab/k3_rat_ab.py] [--dry]
  python -I tools/rat_patches.py --check [--out=...]       re-verify an existing output against its base (anchors, inertness)

Lab form = base + GD geometry alternates RAT_GD_ALT {(proj, b, tag): route} (built, packed, self-tested at prepare like the
shipped GD_ROUTE keys) and RAT_GD_ON {(proj, b): tag} (picked at capture time; empty = the shipped route). An alternate may name
a (proj, b) the shipped GD_ROUTE does not route (o / 32 -> GD instead of DG1). With RAT_GD_ON empty the dispatch is the base's:
_gd_live returns the same record for every shipped key (check() proves it on the source text). The other flips of the sweep are
existing host globals (KS1_TRIG, KS3_TRIG, FT_TAIL3, SU1_ROUTE, GD_ROUTE, VSL_ROUTE, HYA_TUNED) set by the runner at capture
time (tools/box/rat_run.py), so they need no patch. Mechanism = tools/gdh_patches.alt_routes (R-gdh-0) on today's anchors.
"""
import ast
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
NL = chr(10)
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)

# (MT, WR, WK, KU, ST, S, XP). Ship at T 1: o (1,1,8,2,2,1,6), unpooler slice (1,1,16,4,2,1,6); T 8: o (1,1,8,2,2,1,0),
# unpooler (1,1,16,4,2,1,0); T 16: o (2,2,4,4,2,1,6); T 32: o = DG1 (32,1,1,1). Already measured in-layer (R-rpo-4, gated):
# o/1 (1,2,4,4,2,1,2|6) and (1,1,8,2,2,1,2|6), XP 6 on the T 8 base geometries (R-dpf-1 'd8') -> not repeated here.
RAT_GD_ALT = {
    ("o", 1, "o1b"): (1, 2, 4, 4, 4, 1, 6),         # 256 CTAs x 8 warps, every unit pre-wait (ST = KU)
    ("o", 1, "o1c"): (1, 1, 4, 4, 4, 1, 6),         # 512 CTAs x 4 warps, every unit pre-wait
    ("o", 1, "o1d"): (1, 1, 4, 2, 2, 2, 6),         # 1024 CTAs (512 2-CTA clusters) x 4 warps, finer tail
    ("unpooler", 1, "u1a"): (1, 1, 8, 2, 2, 4, 6),  # 256 CTAs (64 4-CTA clusters) x 8 warps, every unit pre-wait
    ("unpooler", 1, "u1b"): (1, 1, 16, 2, 2, 2, 6), # 128 CTAs (2-CTA clusters) x 16 warps, every unit pre-wait
    ("unpooler", 1, "u1c"): (1, 1, 16, 4, 4, 1, 6), # ship geometry, ST 4 = every unit pre-wait
    ("unpooler", 1, "u1d"): (1, 1, 8, 4, 4, 2, 6),  # 128 CTAs x 8 warps, every unit pre-wait
    ("unpooler", 1, "u1e"): (1, 1, 4, 4, 4, 4, 6),  # 256 CTAs (4-CTA clusters) x 4 warps, every unit pre-wait
    ("o", 8, "o8a"): (1, 1, 4, 4, 4, 1, 6),         # o/1 'o1c' at T 8
    ("o", 8, "o8b"): (1, 2, 4, 4, 2, 1, 0),         # the pre-RPO o/1 geometry at T 8, ship XP
    ("unpooler", 8, "u8a"): (1, 1, 8, 2, 2, 4, 6),  # 'u1a' at T 8
    ("unpooler", 8, "u8b"): (1, 1, 16, 4, 4, 1, 6), # 'u1c' at T 8
    ("o", 16, "o16b"): (2, 1, 4, 4, 4, 1, 6),       # R-rpo-4's 'pfw1' shape at MT 2 (untested in-layer)
    ("o", 32, "o32a"): (4, 2, 4, 4, 1, 1, 6),       # GD o at T 32 instead of DG1 (R-gdst-1 standalone o ST1 4.58 us)
    ("o", 32, "o32b"): (4, 2, 4, 4, 1, 1, 0),
}


def check(c, msg):
    if not c:
        raise SystemExit(f"[rat_patches] FAIL: {msg}")


def sub(t, old, new, what):
    n = t.count(old)
    check(n == 1, f"anchor {what!r}: found {n}, expected 1")
    return t.replace(old, new)


def patch(b):
    t = sub(b, "GD_ONE_WAVE = True\n", "GD_ONE_WAVE = True\n" + f"RAT_GD_ALT = {RAT_GD_ALT!r}" + "\nRAT_GD_ON = {}\n"
            "def _rat_items():\n"
            "    return ([((p_, b_, None), r_) for (p_, b_), r_ in sorted(GD_ROUTE.items(), key=str)]\n"
            "            + [(k_, r_) for k_, r_ in sorted(RAT_GD_ALT.items(), key=str)])\n", what="alt globals")
    t = sub(t, """    keys = []
    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", """    keys = []
    for (proj, b, *_), rt in sorted(GD_ROUTE.items(), key=str) + sorted(RAT_GD_ALT.items(), key=str):
        key = _gd_key(*_gd_nk(proj, b), rt) if rt else None""", what="alt keys")
    t = sub(t, """    for (proj, b), rt in sorted(GD_ROUTE.items(), key=str):
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", """    for (proj, b, alt), rt in _rat_items():
        if not rt:
            continue
        n, k = _gd_nk(proj, b)""", what="alt prepare loop")
    t = sub(t, """                routes[(proj, b)] = r
                live.append(f"{proj}/{b} {desc}")""", """                routes[(proj, b, alt) if alt else (proj, b)] = r
                live.append(f"{proj}/{b}{' ALT ' + alt if alt else ''} {desc}")""", what="alt store")
    t = sub(t, """            print(f"[gd] FALLBACK {proj}/{b}: {why} -> today's route", flush=True)""",
            """            print(f"[gd] FALLBACK {proj}/{b}{' ALT ' + alt if alt else ''}: {why} -> today's route", flush=True)""",
            what="alt fallback print")
    t = sub(t, """    if b is None or not GD_ROUTE.get((proj, b)):
        return None
    r = st["route"].get((proj, b))""", """    if b is None:
        return None
    r = st["route"].get((proj, b, RAT_GD_ON[(proj, b)])) if RAT_GD_ON.get((proj, b)) else None
    if r is None:
        if not GD_ROUTE.get((proj, b)):
            return None
        r = st["route"].get((proj, b))""", what="alt live")
    return t


def verify(b, t):
    """Inertness: with RAT_GD_ON empty the patched _gd_live / _gd_keys / _gd_prepare reduce to the base's."""
    tree = ast.parse(t)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    check({"_rat_items", "_gd_live", "_gd_keys", "_gd_prepare"} <= names, "functions present")
    check(t.count("RAT_GD_ON") == 4 and t.count("RAT_GD_ALT") == 4, "RAT_GD_* reference count (3 + nonce)")
    check("_rat_items()" in t, "prepare loop uses _rat_items")
    check(t.isascii(), "non-ASCII")
    check(t.split(NL)[0] == "# okc: problem=muse_decode_layer", "okc header")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")

    # every module-level def other than the 4 GD host functions, and every module-level constant (incl. all CUDA / C++ source
    # strings and GD_ROUTE), is unchanged
    def top(src):
        d = {}
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef):
                d["def " + n.name] = ast.dump(n)
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                d.setdefault("= " + n.targets[0].id, ast.dump(n.value))
        return d
    tb, tt = top(b), top(t)
    changed = sorted(k for k in tt if tb.get(k) != tt[k])
    want = ["= RAT_GD_ALT", "= RAT_GD_ON", "def _gd_keys", "def _gd_live", "def _gd_prepare", "def _rat_items"]
    check(changed == want, f"changed top-level names {changed} != {want}")
    check(sorted(k for k in tb if k not in tt) == [], "nothing removed")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)


def main():
    base = ROOT / ARGS.get("base", "kernels/lab/k3_hh_3tekv.py")
    out = ROOT / ARGS.get("out", "kernels/lab/k3_rat_ab.py")
    b = base.read_text(encoding="utf-8")
    t = patch(b)
    bn = b.split(NL)[1]
    check(bn.startswith("# nonce "), "nonce line 2")
    nonce = f"# nonce k3-rat-ab ({base.stem} + GD route alternates RAT_GD_ALT / RAT_GD_ON, inert unless selected; tools/rat_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    verify(b, t)
    if "--check" in sys.argv:
        check(out.exists() and out.read_text(encoding="utf-8") == t, f"{out.name} differs from a fresh patch of {base.name}")
        print(f"[rat_patches] check OK: {out.relative_to(ROOT).as_posix()} == patch({base.name}), {len(t.encode())} B")
        return 0
    print(f"[rat_patches] OK {len(t.encode())} B (base {len(b.encode())} B); {len(RAT_GD_ALT)} GD alternates")
    if "--dry" in sys.argv:
        return 0
    check(not out.exists() or "--force" in sys.argv, f"{out.name} exists (never overwrite; --force)")
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[rat_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

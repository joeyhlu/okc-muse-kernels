"""dpf_patches.py: R-dpf (dense pre-wait prefetch audit, 2026-10-29). rpo_sass.py (now incl. the GD-cat keys and --audit units dg1 /
p1) on k3_gc_21 shows the R-rpo-0 lost prefetch on three LIVE routes: GD-cat T = 16 (0 of 6 weight LDGs above ACQBULK), GD-cat T = 32
(0 of 3) and GD unpooler T = 32 (0 of 3); all three still load x with ld.global.nc (xp 0). The fix is R-rpo-4's: XP 6 = pre-wait
stages via coherent ld.global.L1::no_allocate (ldwp) + x via plain ld.global (ldxg). The GD kernel template already supports XP 6
(cat paths share R[] / ldxk<XP>); only _gd_cat_key rejected xp outside (0, 1).
DG1 (o T 32 / 64; UBLKCP weights stay above the wait), VD (cat T 128; UBLKCP above) and P1M / P1S (lost too, but shadowed by GD /
GD-cat on every judged shape) need no change.

  python tools/dpf_patches.py --n=1 --mode=lab     -> kernels/lab/k3_dpf_1.py  (ship routes + inert alternates for rev_ab3:
        RPO_GD_ON {"unpooler,32": "pf"}, DPF_CAT_ON {"16": "pf", "32": "pf"})
  python tools/dpf_patches.py --n=2 --mode=clean [--cat=16,32] [--up32=1] [--d8=0|1]  -> kernels/lab/k3_dpf_2.py (routes switched, no toggles)
"""
import hashlib
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from gdd_patches import check, sub  # noqa: E402
from rpo_patches import alt_routes  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
CAT_PF = {16: "(2, 1, 8, 18, 2, 1, 6)", 32: "(4, 1, 8, 18, 1, 1, 6)"}
CAT_SHIP = {16: "(2, 1, 8, 18, 2, 1, 0)", 32: "(4, 1, 8, 18, 1, 1, 0)"}
UP32_SHIP, UP32_PF = '("unpooler", 32): (4, 2, 4, 16, 1, 1, 0)', '("unpooler", 32): (4, 2, 4, 16, 1, 1, 6)'
# D (R-rpo-5's T 8 loss): g20_6's pooler/8 xp 7 tripped _vfr_prepare's desc check ("s2x1") -> VFR FALLBACK to the side-stream router32
# in the clone (rev_ab3 kept the base xp 1 route, so VFR stayed live there). D8 = PF + plain x on o/8 and unpooler/8 only; pooler/8 stays xp 1.
D8 = (('("o", 8): (1, 1, 8, 2, 2, 1, 0)', '("o", 8): (1, 1, 8, 2, 2, 1, 6)'),
      ('("unpooler", 8): (1, 1, 16, 4, 2, 1, 0)', '("unpooler", 8): (1, 1, 16, 4, 2, 1, 6)'))
ALT = ('RPO_GD_ALT = {("unpooler", 32, "pf"): (4, 2, 4, 16, 1, 1, 6), ("o", 8, "pf"): (1, 1, 8, 2, 2, 1, 6),\n'
       '              ("unpooler", 8, "pf"): (1, 1, 16, 4, 2, 1, 6)}')


def cat_key(t):
    return sub(t, "or st_ > ku or xp not in (0, 1):", "or st_ > ku or xp not in (0, 1, 6):", what="cat key xp 6")


def cat_alt(t):
    t = sub(t, "GD_CAT_UNITS = 2\n", "GD_CAT_UNITS = 2\n# R-dpf lab: GD-cat alternates (PF + plain x), armed per bucket by DPF_CAT_ON {b: 'pf'} (rev_ab3 arms)\n"
            f"DPF_CAT_ALT = {{(16, 'pf'): {CAT_PF[16]}, (32, 'pf'): {CAT_PF[32]}}}\nDPF_CAT_ON = {{}}\n", what="cat alt globals")
    t = sub(t, """    for b, rt in sorted(GD_CAT_ROUTE.items()):
        key = _gd_cat_key(rt) if rt else None
        if key is not None and key not in keys:""", """    for b, rt in sorted(GD_CAT_ROUTE.items()) + sorted(DPF_CAT_ALT.items()):
        key = _gd_cat_key(rt) if rt else None
        if key is not None and key not in keys:""", what="cat alt keys")
    t = sub(t, """    for b, rt in sorted(GD_CAT_ROUTE.items()):
        why = None""", """    for bk, rt in sorted(GD_CAT_ROUTE.items()) + sorted(DPF_CAT_ALT.items()):
        b = bk[0] if isinstance(bk, tuple) else bk
        why = None""", what="cat alt prepare loop")
    t = sub(t, """                routes[b] = r
                live.append(f"cat/{b} {desc}")""", """                routes[bk] = r
                live.append(f"cat/{bk} {desc}")""", what="cat alt store")
    t = sub(t, """    r = st.get(b)
    return r if r is not None and tokens <= r["T"] else None""", """    r = (st.get((b, DPF_CAT_ON[b])) if DPF_CAT_ON.get(b) else None) or st.get(b)
    return r if r is not None and tokens <= r["T"] else None""", what="cat alt live")
    return t


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    mode = ARGS.get("mode", "lab")
    check(mode in ("lab", "clean"), "--mode=lab|clean")
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_21')}.py"
    out = ROOT / f"kernels/lab/k3_dpf_{n}.py"
    check(not out.exists(), f"{out.name} exists (never overwrite)")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    t = cat_key(b)
    if mode == "lab":
        t = alt_routes(t, ALT)
        t = cat_alt(t)
        what = "GD-cat xp 6 allowed + inert alternates: RPO_GD_ON unpooler/32 / o/8 / unpooler/8 pf, DPF_CAT_ON 16 / 32 pf"
    else:
        cats = [int(c) for c in ARGS.get("cat", "16,32").split(",") if c]
        for c in cats:
            t = sub(t, f"{c}: {CAT_SHIP[c]}", f"{c}: {CAT_PF[c]}", what=f"cat {c} route")
        parts = [f"cat{c}" for c in cats]
        if ARGS.get("up32", "1") == "1":
            t = sub(t, UP32_SHIP, UP32_PF, what="unpooler 32 route")
            parts.append("unpooler32")
        if ARGS.get("d8", "0") == "1":
            for a, b_ in D8:
                t = sub(t, a, b_, what="d8 route")
            parts += ["o8", "unpooler8"]
        what = "GD pre-wait PF + plain x (XP 6) on " + "/".join(parts)
    nonce = f"# nonce {out.stem.replace('_', '-')} ({base.stem} + {what}; tools/dpf_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    check(re.search(r"importlib|sys\.modules", t) is None or re.search(r"importlib|sys\.modules", b) is not None, "checker")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[dpf_patches] OK {mode} {len(t.encode())} B (base {len(b.encode())} B, +{len(t.encode()) - len(b.encode())})")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[dpf_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

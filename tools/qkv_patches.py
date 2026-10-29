"""qkv_patches.py: R-qkv lab files (qkv GD route geometry at T = 16 / 32) on the GDH lab form.

  python -I tools/qkv_patches.py --n=<n> [--base=kernels/lab/k3_gdh_0.py] [--dry]   -> kernels/lab/k3_qkv_<n>.py

k3_gdh_0 = k3_gc_14 + inert GD / sbb1 marks + LAT_GD_ALT / LAT_GD_ON alternates (tools/gdh_patches.py). This swaps the
alternates for the qkv geometry candidates below (picked at capture time with LAT_GD_ON {("qkv", b): tag}) and drops
the GC alternates (build time).
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

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
NL = chr(10)
# (MT, WR, WK, KU, ST, S, XP); ship qkv 16 / 32 = (2|4, 1, 4, 8, 1, 4, 0): 144 clusters x 4 CTAs x 4 warps
CANDS = {"s8": (1, 2, 8, 1, 8, 0),     # 144 clusters x 8 CTAs x 2 warps (same 16 warp K-chunks per row tile)
         "s2": (1, 8, 8, 1, 2, 0)}     # 144 clusters x 2 CTAs x 8 warps
# dropped after the local compile: s8w4 (1, 4, 4, 1, 8, 0) = 56 regs at MB 9 with 624 / 80 B spills (T 32 / 16)


def main():
    n = ARGS.get("n")
    check(n, "--n=<n> required")
    base = ROOT / ARGS.get("base", "kernels/lab/k3_gdh_0.py")
    out = ROOT / f"kernels/lab/k3_qkv_{n}.py"
    check(not out.exists(), f"{out.name} exists (never overwrite)")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    alt = {("qkv", bk, tag): (mt,) + g for tag, g in CANDS.items() for bk, mt in ((16, 2), (32, 4))}
    t = re.sub(r"^LAT_GD_ALT = \{.*\}$", lambda m: f"LAT_GD_ALT = {alt!r}", b, count=1, flags=re.M)
    check(t != b, "LAT_GD_ALT anchor")
    t = sub(t, re.search(r"^LAT_CAT_ALT = \{.*\}$", t, flags=re.M).group(0), "LAT_CAT_ALT = {}", what="cat alt")
    nonce = f"# nonce k3-qkv-{n} ({base.stem} with qkv 16/32 geometry alternates {sorted(CANDS)}; tools/qkv_patches.py)"
    t = t.replace(bn, nonce, 1)
    check(t.split(NL)[1] == nonce, "nonce")
    check(t.isascii(), "non-ASCII")
    check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[qkv_patches] OK {len(t.encode())} B; alternates {alt}")
    if "--dry" in sys.argv:
        return 0
    out.write_text(t, encoding="utf-8", newline=NL)
    print(f"[qkv_patches] wrote {out.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

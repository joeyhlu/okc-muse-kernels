"""gen_k3_dgp.py: kernels/muse_decode_layer/k3_dgp_<n>.py = <base layer file> + DGP (DG1 fed with lossless P12 stage blocks as
the T = 17..32 cat / o route; tools/dg1p12_patches.py, notes/study/dgp_dense.md). Base = a generated layer file on disk
(default k3_slim_4.py); its own strip is asserted to be a fixed point (finish(base) == base up to the nonce), so the output
differs from the base only by the DGP patch. Then gen_k3_m32.finish (strips, compiles, ASCII, nonce, no duplicate names,
judge-checker scan <= base, < 1 MB). Writes the layer file and a byte-identical box copy kernels/lab/k3_dgp_<n>_ab.py.
Never overwrites an existing k3_dgp_<n>.py.

    python -I tools/gen_k3_dgp.py [--n=1] [--base=k3_slim_4] [--routes=cat,o] [--lab-only]
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import dg1p12_patches  # noqa: E402

N = ARGS.get("n", "1")
_b = ARGS.get("base", "k3_slim_4")
BASE = pathlib.Path(_b) if (_b.endswith(".py") or "/" in _b or chr(92) in _b) else ROOT / f"kernels/muse_decode_layer/{_b}.py"
ROUTES = ARGS.get("routes", "cat,o").split(",")
LABONLY = "--lab-only" in sys.argv
DRY = "--dry" in sys.argv   # run every check, write nothing
OUT = ROOT / f"kernels/muse_decode_layer/k3_dgp_{N}.py"
LAB = ROOT / f"kernels/lab/k3_dgp_{N}_ab.py"
TAG = f"k3-dgp-{N}"
NL = chr(10)


def main():
    rep = []
    for p in (OUT, LAB):
        GM.check(not p.exists(), f"{p.name} exists; pick another --n (never overwrite)")
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    nonce = (f"# nonce {TAG} ({BASE.stem} + DGP: DG1's SS-wgmma dense GEMM on lossless 12-bit P12 stage blocks for"
             f" {' / '.join(ROUTES)} at T = 17..32, self-tested bit-exact with DG1 fallback)")
    tag0 = bn.split()[2]
    fb = GM.finish(b, bn, tag0, chk_base, [])
    GM.check(fb == b, f"{BASE.name} is not a fixed point of finish()")
    rep.append(f"base {BASE.name} {len(b.encode())} B is a finish() fixed point")
    t = b.replace(bn, nonce)
    route = {(p, 32): True for p in ROUTES}
    geometry = {k: v for k, v in dg1p12_patches.GEOMETRY.items() if k in route}
    GM.check(set(geometry) == set(route), f"no DGP geometry for {sorted(set(route) - set(geometry))}")
    t = dg1p12_patches.apply(t, route, geometry)
    rep.append(f"DGP routes {sorted(route)} geometry {geometry}")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_dgp] OK  {r}")
    if DRY:
        print(f"[gen_k3_dgp] --dry: nothing written ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B)")
        return 0
    if not LABONLY:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_dgp] wrote {'' if LABONLY else OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

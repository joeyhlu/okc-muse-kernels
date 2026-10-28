"""gen_k3_vml.py: kernels/muse_decode_layer/k3_vml_<n>.py = <base layer file> + VML1 as the T >= 17 MoE route
(tools/vml_patches.py, notes/study/vm_large.md). Base = a generated layer file on disk (default k3_vms_1.py = k3_m32_3 +
VMS); its own strip is asserted to be a fixed point (finish(base) == base up to the nonce), so the output differs from the
base only by the VML1 patch (+ optional --slim: tools/vml_slim.py removes K2A v3.7, the 128-bucket code VML1 replaces).
Then gen_k3_m32.finish (strips, compiles, ASCII, nonce, no duplicate names, judge-checker scan <= base, < 1 MB). Writes the
layer file and a byte-identical box copy kernels/lab/k3_vml_<n>_ab.py.

    python -I tools/gen_k3_vml.py [--n=1] [--base=k3_vms_1] [--t32] [--slim] [--lab-only] [--replace]
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
import vml_patches  # noqa: E402

N = ARGS.get("n", "1")
_b = ARGS.get("base", "k3_vms_1")
BASE = pathlib.Path(_b) if (_b.endswith(".py") or "/" in _b or chr(92) in _b) else ROOT / f"kernels/muse_decode_layer/{_b}.py"
T32 = "--t32" in sys.argv
SLIM = "--slim" in sys.argv
LABONLY = "--lab-only" in sys.argv
OUT = ROOT / f"kernels/muse_decode_layer/k3_vml_{N}.py"
LAB = ROOT / f"kernels/lab/k3_vml_{N}_ab.py"
TAG = f"k3-vml-{N}"
NL = chr(10)


def main():
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    nonce = (f"# nonce {TAG} ({BASE.stem} + VML1: one-launch RS-wgmma stream-K MoE at buckets 128 / 256"
             f"{' / 32' if T32 else ''}{', K2A v3.7 removed' if SLIM else ''}, self-tested vs K2A with fallback)")
    tag0 = bn.split()[2]
    fb = GM.finish(b, bn, tag0, chk_base, [])
    GM.check(fb == b, f"{BASE.name} is not a fixed point of finish()")
    rep.append(f"base {BASE.name} {len(b.encode())} B is a finish() fixed point")
    t = b.replace(bn, nonce)
    if SLIM:
        import vml_slim
        t = vml_slim.apply(t)
        rep.append(f"slim: K2A v3.7 removed ({len(t.encode())} B)")
    import json
    fl = {int(k): int(v) for k, v in json.loads(ARGS.get("flags", "{}")).items()}
    co = {int(k): [int(x) for x in v] for k, v in json.loads(ARGS.get("cost", "{}")).items()}
    if "--replace" in sys.argv:   # base already carries a VML block (k3_va_3, ...): swap it for the current lab source
        t = vml_patches.replace(t, t32=T32, flags=fl, cost=co)
    else:
        t = vml_patches.apply(t, t32=T32, flags=fl, cost=co)
    if LABONLY:   # box A/B copy only: the 1 MB judge limit is reported, not enforced
        chk0 = GM.check
        GM.check = lambda ok, msg: chk0(ok or "file >= 1 MB" in msg, msg)
        try:
            s = GM.finish(t, nonce, TAG, chk_base, rep)
        finally:
            GM.check = chk0
    else:
        s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_vml] OK  {r}")
    if not LABONLY:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vml] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

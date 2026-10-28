"""gen_k3_slim.py: kernels/muse_decode_layer/k3_slim_<n>.py = <base layer file> with K2A v3.7 removed (tools/vml_slim.py;
VML1 serves bucket 128, whose K2A fallback becomes the v3.6 / v3.3 plan). Frees room under the 1 MB limit for riders.
The base must be a finish() fixed point; the output goes through gen_k3_m32.finish (strips, compiles, ASCII, nonce, no
duplicate names, judge-checker scan <= base, < 1 MB). Refuses to overwrite an existing output.

    python -I tools/gen_k3_slim.py --base=k3_vf_5 --n=1
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
import vml_slim  # noqa: E402

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS['base']}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_slim_{N}.py"
LAB = ROOT / f"kernels/lab/k3_slim_{N}_ab.py"
TAG = f"k3-slim-{N}"
NL = chr(10)


def main():
    GM.check(not OUT.exists(), f"{OUT.name} exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = f"# nonce {TAG} ({BASE.stem} with K2A v3.7 removed: VML1 serves bucket 128, K2A fallback = v3.6 / v3.3)"
    t = vml_slim.apply(b.replace(bn, nonce))
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_slim] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_slim] {BASE.name} {len(b.encode())} B -> {OUT.name} {len(s.encode())} B "
          f"(freed {len(b.encode()) - len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

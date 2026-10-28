"""gen_k3_slim3.py: kernels/muse_decode_layer/k3_slim_<n>.py = k3_slim_2 with the M32 PX route removed (tools/vml_px_strip.py)
and the VML block swapped for the current lab source (kernels/lab/vml_moe.py v8: v7 + flags bit 8 = vm_main as a PDL
secondary, off by default), keeping k3_vml_8's VML settings (VML at 128 / 256 / 32, flags 193, R-vml-10 tables).
Bucket 32's fallback chain: VML1 -> r36 (K2A v3.6 vid 3) -> the v3.3 plan. Base must be a finish() fixed point; output goes
through gen_k3_m32.finish (strips, compiles, ASCII, nonce, no duplicate names, judge-checker scan <= base, < 1 MB).
Refuses to overwrite. Writes the layer file and the byte-identical box copy kernels/lab/k3_slim_<n>_ab.py.

    python -I tools/gen_k3_slim3.py [--n=3] [--base=k3_slim_2]
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
import vml_px_strip  # noqa: E402

N = ARGS.get("n", "3")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_slim_2')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_slim_{N}.py"
LAB = ROOT / f"kernels/lab/k3_slim_{N}_ab.py"
TAG = f"k3-slim-{N}"
NL = chr(10)
FLAGS = {128: 193, 256: 193, 32: 193}
COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
        128: [128, 157, 174, 194, 208, 230, 246, 204, 128, 154, 179, 189, 219, 229, 250, 204],
        32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}


def main():
    GM.check(not OUT.exists() and not LAB.exists(), f"{OUT.name} or its lab copy exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    nonce = (f"# nonce {TAG} ({BASE.stem} with M32 PX removed: bucket 32 = VML1 -> K2A v3.6 vid 3 -> v3.3; VML lab v8, "
             f"flags bit 8 = PDL vm_main off by default)")
    t = vml_px_strip.apply(b.replace(bn, nonce))
    rep.append(f"PX removed: {len(b.encode())} -> {len(t.encode())} B pre-finish")
    t = vml_patches.replace(t, t32=True, flags=FLAGS, cost=COST)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_slim3] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_slim3] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

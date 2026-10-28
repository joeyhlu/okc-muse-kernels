"""gen_k3_slim5.py: kernels/muse_decode_layer/k3_slim_<n>.py = k3_slim_4 with T=128's VML nt8 cost entries (never fitted:
the module default 204, below nt7's 246 / 250) set to 262 / 267 (T=256's nt7 -> nt8 steps; R-vml-12 synthetic 60-token
routings). Only the VML_COST line changes, so plans differ only on routings with a 57-64-token job at bucket 128. Base must be a
finish() fixed point; output goes through gen_k3_m32.finish. Refuses to overwrite. Writes the layer file and the identical
box copy kernels/lab/k3_slim_<n>_ab.py.

    python -I tools/gen_k3_slim5.py [--n=5] [--base=k3_slim_4]
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

N = ARGS.get("n", "5")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_slim_4')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_slim_{N}.py"
LAB = ROOT / f"kernels/lab/k3_slim_{N}_ab.py"
TAG = f"k3-slim-{N}"
NL = chr(10)
OLD = "128: [128, 157, 174, 194, 208, 230, 246, 204, 128, 154, 179, 189, 219, 229, 250, 204]"
NEW = "128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267]"


def main():
    GM.check(not OUT.exists() and not LAB.exists(), f"{OUT.name} or its lab copy exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    line = next(ln for ln in b.split(NL) if ln.startswith("VML_COST = "))
    GM.check(b.count(line) == 1 and line.count(OLD) == 1, "VML_COST line / T=128 table not as expected")
    nonce = f"# nonce {TAG} ({BASE.stem} with VML T=128 nt8 cost entries 262 / 267 instead of the unfitted default 204)"
    t = b.replace(bn, nonce).replace(line, line.replace(OLD, NEW))
    rep.append("VML_COST[128] nt8 entries 204 / 204 -> 262 / 267")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_slim5] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_slim5] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

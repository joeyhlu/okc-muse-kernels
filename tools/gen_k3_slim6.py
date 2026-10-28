"""gen_k3_slim6.py: k3_slim_<n> = k3_slim_5 with the VML block swapped for the current lab source (kernels/lab/vml_moe.py v10:
v8 + v9 marks columns + flags bit 9 = phase-B stage order by predicted A readiness), VML at 128 / 256 / 32 with k3_slim_5's
tables (R-vml-10 refit + T=128 nt8 262 / 267) and per-bucket flags (default 193 everywhere = k3_slim_5 behaviour; e.g.
--flags='{"128": 705, "256": 705, "32": 705}' turns bit 9 on). Base must be a finish() fixed point; output goes through
gen_k3_m32.finish. Refuses to overwrite. --lab-only writes only the box copy kernels/lab/k3_slim_<n>_ab.py.

    python -I tools/gen_k3_slim6.py [--n=6] [--base=k3_slim_5] [--flags=JSON] [--lab-only]
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAB_ONLY = "--lab-only" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32 as GM  # noqa: E402
sys.argv = _argv
import vml_patches  # noqa: E402

N = ARGS.get("n", "6")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_slim_5')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_slim_{N}.py"
LAB = ROOT / f"kernels/lab/k3_slim_{N}_ab.py"
TAG = f"k3-slim-{N}"
NL = chr(10)
FLAGS = {int(k): int(v) for k, v in json.loads(ARGS.get("flags", '{"128": 193, "256": 193, "32": 193}')).items()}
COST = {256: [128, 153, 172, 186, 198, 216, 232, 248, 128, 149, 168, 180, 191, 208, 219, 236],
        128: [128, 157, 174, 194, 208, 230, 246, 262, 128, 154, 179, 189, 219, 229, 250, 267],
        32: [128, 167, 183, 229, 171, 182, 193, 204, 128, 162, 184, 215, 171, 182, 193, 204]}


def main():
    GM.check(not LAB.exists() and (LAB_ONLY or not OUT.exists()), f"{OUT.name} or its lab copy exists; pick another --n")
    rep = []
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    line = next(ln for ln in b.split(NL) if ln.startswith("VML_COST = "))
    GM.check(line == "VML_COST = " + repr(COST), "base VML_COST differs from the k3_slim_5 tables")
    nonce = (f"# nonce {TAG} ({BASE.stem} with VML lab v10: flags bit 9 = B stage order by predicted A readiness; "
             f"flags {json.dumps({str(k): v for k, v in FLAGS.items()})})")
    t = vml_patches.replace(b.replace(bn, nonce), t32=True, flags=FLAGS, cost=COST)
    rep.append(f"VML block -> lab v10, flags {FLAGS}")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_slim6] OK  {r}")
    if not LAB_ONLY:
        OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_slim6] wrote {'' if LAB_ONLY else str(OUT.relative_to(ROOT)) + ' and '}{LAB.relative_to(ROOT)} "
          f"({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

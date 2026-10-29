"""gen_k3_l1b.py: kernels/lab/k3_l1b_<n>.py = a layer (default k3_gc_20) + L1 ROUTE-EARLY (tools/l1b_patches.py).

Step 0 (n = 0): RE_MODE per bucket {1, 8, 16} read at call time (0 = ship path, 1 = inert: router first + routing CTAs in
the pooler launch write scratch routing; tail2 / VMS unchanged); RE_MK_ON = lab marks (RE = 2 instances, _RE["mk"]).
Also writes tools/box/ab_l1b<n>.json (ship / re / mid / re2 arms). Refuses to overwrite (--dry: checks only).

    python -I tools/gen_k3_l1b.py --n=0 [--base=k3_gc_19] [--dry]
"""
import hashlib
import json
import pathlib
import py_compile
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import l1b_patches as P  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = ARGS.get("n", "0")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_20')}.py"
OUT = ROOT / f"kernels/lab/k3_l1b_{N}.py"
AB = ROOT / f"tools/box/ab_l1b{N}.json"
NL = chr(10)


def main():
    dry = "--dry" in sys.argv
    P.check(dry or not OUT.exists(), f"{OUT.name} exists")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    P.check(bn.startswith("# nonce "), "line 2 is not a nonce line")
    t = P.apply(b)
    nonce = (f"# nonce k3-l1b-{N} ({BASE.stem} + L1b ROUTE-EARLY step 0: router + routing kernels in front of a flag-gated pooler, "
             "L1B_MODE per bucket at call time; tools/gen_k3_l1b.py)")
    t = t.replace(bn, nonce, 1)
    P.check(t.split(NL)[1] == nonce, "nonce")
    P.check(t.isascii(), "non-ASCII")
    P.check(len(t.encode()) < 1_000_000, f"{len(t.encode())} B >= 1 MB")
    P.check(max(len(x) for x in t.split(NL)) <= max(len(x) for x in b.split(NL)), "line longer than the base's longest")
    P.check(len(re.findall(r"importlib|sys\.modules", t)) == len(re.findall(r"importlib|sys\.modules", b)), "checker scan")
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "k.py"
        f.write_text(t, encoding="utf-8", newline=NL)
        py_compile.compile(str(f), doraise=True)
    print(f"[gen_k3_l1b] OK {OUT.name}: {len(t.encode())} B (base {len(b.encode())} B, +{len(t.encode()) - len(b.encode())})")
    if dry:
        return 0
    OUT.write_text(t, encoding="utf-8", newline=NL)
    off, on = {"1": 0, "8": 0}, {"1": 1, "8": 1}
    js = [{"name": "ship", "set": {"L1B_MODE": off}}, {"name": "re", "set": {"L1B_MODE": on}},
          {"name": "mid", "set": {"L1B_MODE": off}}, {"name": "re2", "set": {"L1B_MODE": on}}]
    if not AB.exists():
        AB.write_text("[" + ("," + NL + " ").join(json.dumps(x) for x in js) + "]" + NL)
    print(f"[gen_k3_l1b] wrote {OUT.relative_to(ROOT).as_posix()} (sha1 {hashlib.sha1(t.encode()).hexdigest()[:12]}), {AB.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

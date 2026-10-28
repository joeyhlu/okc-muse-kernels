"""gen_k3_b1a.py: kernels/lab/b1a_<base>_ab.py = a layer file (default k3_gc_1) + B1A, the one-launch B = 1 full attention
(tools/b1a_patches.py, notes/study/b1a.md).

The base must be a gen_k3_m32.finish() fixed point (asserted); B1A's block, PT1 build job, prepare hook and route are applied
(every anchor asserted unique), the nonce line is replaced, and the text goes through gen_k3_m32.finish (strip_layer + strip_cxx
with AST-equivalence checks, compiles, ASCII, line 1, nonce, no duplicate top-level names, judge-checker scan <= base, < 1 MB).
The output is asserted to be a finish() fixed point too. Refuses to overwrite an existing file.

    python -I tools/gen_k3_b1a.py [--base=k3_gc_1] [--n=1] [--set=NAME=VALUE,...] [--check]
        --check: regenerate and compare with the file on disk (no write)
"""
import hashlib
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
CHECK = "--check" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0]]
try:
    import gen_k3_m32 as GM  # noqa: E402
    _STUB = False
except ModuleNotFoundError as ex:   # a clone without a generator-chain module that finish() itself never uses (sbb1_patches)
    if ex.name != "sbb1_patches":
        raise
    sys.modules["sbb1_patches"] = types.ModuleType("sbb1_patches")
    import gen_k3_m32 as GM  # noqa: E402
    _STUB = True
sys.argv = _argv
import b1a_patches  # noqa: E402

N = ARGS.get("n", "1")
BASE_NAME = ARGS.get("base", "k3_gc_1")
BASE = ROOT / f"kernels/muse_decode_layer/{BASE_NAME}.py"
OUT = ROOT / f"kernels/lab/b1a_{BASE_NAME}_ab.py" if N == "1" else ROOT / f"kernels/lab/b1a{N}_{BASE_NAME}_ab.py"
TAG = f"k3-b1a-{N}"
NL = chr(10)


def _sets():
    raw = ARGS.get("set", "")
    return [tuple(kv.split("=", 1)) for kv in raw.split(",") if kv] if raw else []


def _checker_base(b):
    """The judge-checker scan baseline: the k3_v37 chain text when this clone can regenerate it (as gen_k3_gc does), else
    the base layer file itself (stricter: B1A may not add a single checked pattern to the shipped file)."""
    if not _STUB:
        try:
            return GM.G37.generate()[1], "k3_v37 chain"
        except (SystemExit, Exception) as ex:   # noqa: BLE001
            print(f"[gen_k3_b1a] note: k3_v37 chain not regenerable here ({ex!r:.120}); checker baseline = {BASE.name}")
    return b, BASE.name


def main():
    GM.check(BASE.exists(), f"{BASE.name} not found")
    if not CHECK:
        GM.check(not OUT.exists(), f"{OUT.relative_to(ROOT)} exists; pick another --n")
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    chk_base, chk_name = _checker_base(b)
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    rep = [f"{BASE.name} is a finish() fixed point ({len(b.encode())} B); checker baseline {chk_name}"]
    sets = _sets()
    nonce = (f"# nonce {TAG} ({BASE.stem} + B1A: one-launch B = 1 full attention at q_len 1 / 8, 264 split-KV CTAs with all K / V "
             "rows in one cp.async burst, FP8 mma.sync, red.add merge + bounded cooperative finalize; B1A_ENABLE read at call time"
             + (f"; sets {sets}" if sets else "") + ")")
    t = b.replace(bn, nonce)
    t = b1a_patches.apply(t, sets)
    rep.append("b1a_patches.apply: block before kernel_fn, PT1 job, prepare hook after _vfa_prepare, route before the FA4 B = 1 branch")
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    GM.check(GM.finish(s, nonce, TAG, chk_base, []) == s, "output is not a finish() fixed point")
    rep.append(f"{TAG}: output is a finish() fixed point")
    for r in rep:
        print(f"[gen_k3_b1a] OK  {r}")
    sha = hashlib.sha1(s.encode()).hexdigest()[:12]
    if CHECK:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_b1a] --check: {OUT.relative_to(ROOT)} up to date ({len(s.encode())} B, sha1 {sha})")
        return 0
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_b1a] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, headroom {1_000_000 - len(s.encode())} B, sha1 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

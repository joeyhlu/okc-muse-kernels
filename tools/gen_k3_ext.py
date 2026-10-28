"""gen_k3_ext.py: k3_ext_<n> = a layer file (default k3_gc_1) + VM_TMAR (tools/tmar_patches.py): VML split shares of jobs with
<= 16 tokens are committed through shared memory and a dedicated writer warp's 1-D TMA bulk reduce-add
(cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32, the FlashAttention-3 store_dq pattern) instead of per-thread
red.add + WG barrier + counter atomic. Variant "t" is the default (VML_VAR), the shipped variant "s" is built beside it (A/B
arms and the self-test fallback); the async phases are A.flags bits 20 (A) / 21 (B), set in VML_FLAGS.

Output goes through gen_k3_m32.finish (strip, compile, ASCII, nonce, checker scan, < 1 MB). Refuses to overwrite.

    python -I tools/gen_k3_ext.py [--base=k3_gc_1] [--n=1]
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
import tmar_patches as P  # noqa: E402

N = ARGS.get("n", "1")
BASE = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_1')}.py"
OUT = ROOT / f"kernels/muse_decode_layer/k3_ext_{N}.py"
LAB = ROOT / f"kernels/lab/k3_ext_{N}_ab.py"
TAG = f"k3-ext-{N}"
NL = chr(10)


def main():
    GM.check(not OUT.exists() and not LAB.exists(), f"{OUT.name} or its lab copy exists; pick another --n")
    _, chk_base, _ = GM.G37.generate()
    b = BASE.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    GM.check(GM.finish(b, bn, bn.split()[2], chk_base, []) == b, f"{BASE.name} is not a finish() fixed point")
    t, rep = P.apply(b)
    nonce = (f"# nonce {TAG} ({BASE.stem} + VM_TMAR: VML N <= 16 split shares via smem + a writer warp's TMA bulk reduce-add "
             "(FA3 store_dq pattern), shipped variant kept for A/B and self-test fallback; tools/gen_k3_ext.py)")
    t = t.replace(bn, nonce)
    s = GM.finish(t, nonce, TAG, chk_base, rep)
    for r in rep:
        print(f"[gen_k3_ext] OK  {r}")
    OUT.write_text(s, encoding="utf-8", newline=NL)
    LAB.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_ext] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, headroom "
          f"{1_000_000 - len(s.encode())} B, sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

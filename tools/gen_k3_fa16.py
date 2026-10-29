"""gen_k3_fa16.py: kernels/lab/<out>.py = a layer file (default k3_gc_9) + FA16 (tools/fa16_patches.py: VFA verify-full softmax
exponentials partly on the FMA pipe, PEX of 32 per thread and tile; compile-time, shipped build without lab arms).

Same contract as gen_k3_att1.py: every anchor asserted unique, the text goes through gen_k3_m32.finish (strip + AST checks, ASCII,
nonce, judge-checker scan <= base, < 1 MB), the output is a finish() fixed point, refuses to overwrite.

    python -I tools/gen_k3_fa16.py --out=k3_fa16_1 [--vsm] [--pp] [--pvs] [--h16] [--pex=N] [--base=k3_gc_9] [--check]
"""
import hashlib
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
FLAGS = {a for a in sys.argv[1:] if a.startswith("--") and "=" not in a}
NL = chr(10)
import fa16_patches as FP  # noqa: E402

OUT_NAME = ARGS.get("out", "k3_fa16_lab")
OUT = ROOT / f"kernels/lab/{OUT_NAME}.py"
TAG = OUT_NAME.replace("_", "-")
PEX = int(ARGS.get("pex", "0"))
H16, PVS, PP, VSM = "--h16" in FLAGS, "--pvs" in FLAGS, "--pp" in FLAGS, "--vsm" in FLAGS


def main():
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        import gen_k3_m32 as GM  # noqa: E402
        stub = False
    except ModuleNotFoundError as ex:
        if ex.name != "sbb1_patches":
            raise
        sys.modules["sbb1_patches"] = types.ModuleType("sbb1_patches")
        import gen_k3_m32 as GM  # noqa: E402
        stub = True
    sys.argv = _argv
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_9')}.py"
    check = "--check" in FLAGS
    GM.check(base.exists(), f"{base.name} not found")
    GM.check(0 <= PEX <= 32, "pex in 0..32")
    if not check:
        GM.check(not OUT.exists(), f"{OUT.relative_to(ROOT)} exists; pick another --out")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    chk_base = b
    if not stub:
        try:
            chk_base = GM.G37.generate()[1]
        except (SystemExit, Exception) as ex:   # noqa: BLE001
            print(f"[gen_k3_fa16] note: k3_v37 chain not regenerable ({ex!r:.100}); checker baseline = {base.name}")
    opts = [f"PEX {PEX}"] * bool(PEX) + ["H16"] * H16 + ["PVS"] * PVS + ["PP (exp-phase ping-pong of the 2 consumer warpgroups)"] * PP + ["VSM (V^T per-thread constants in smem)"] * VSM
    nn = f"# nonce {TAG} ({base.stem} + FA16 VFA verify-full consumer: {', '.join(opts) or 'no-op'})"
    rep = [f"base {base.name} ({len(b.encode())} B)", f"fa16_patches.apply(pex={PEX}, h16={H16}, pvs={PVS}, pp={PP}, vsm={VSM}, lab=False)"]
    t = FP.apply(b.replace(bn, nn), pex=PEX, h16=H16, pvs=PVS, pp=PP, vsm=VSM, lab=False)
    s = GM.finish(t, nn, TAG, chk_base, rep)
    GM.check(GM.finish(s, nn, TAG, chk_base, []) == s, "output is not a finish() fixed point")
    for r in rep:
        print(f"[gen_k3_fa16] OK  {r}")
    sha = hashlib.sha1(s.encode()).hexdigest()[:12]
    if check:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[gen_k3_fa16] check OK {OUT.name} sha1 {sha}")
        return
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_fa16] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {sha})")


if __name__ == "__main__":
    main()

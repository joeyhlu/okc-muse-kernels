"""gen_k3_vfa3.py: kernels/lab/<out>.py = a layer file (default k3_gc_13) + fa16_patches (VSM optional) + VFA3 (tools/vfa3_patches.py:
V^T transpose on the producer warps, PVT) of the VFA verify-full kernel.
Same contract as gen_k3_vfa3.py: anchors asserted unique, gen_k3_m32.finish (strip + AST checks, ASCII, nonce, judge-checker scan
<= base, < 1 MB), finish() fixed point, refuses to overwrite.
--lab: lab build (VFA_DIAG arms 16 VSM, 17 NOVT bound, 18 / 19 / 20 PVT lag 2 / 1 / 3; _vfa_build(lab=True)).
    python -I tools/gen_k3_vfa3.py --out=k3_vfa3_1 [--pvt=0|1] [--lag=2] [--preg=40] [--creg=96] [--vsm] [--lab] [--base=k3_gc_13] [--check]
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
import vfa3_patches as V3  # noqa: E402

OUT_NAME = ARGS.get("out", "k3_vfa3_lab")
OUT = ROOT / f"kernels/lab/{OUT_NAME}.py"
TAG = OUT_NAME.replace("_", "-")
VSM, LAB, PVT = "--vsm" in FLAGS, "--lab" in FLAGS, ARGS.get("pvt", "1") == "1"
LAG, PREG, CREG = int(ARGS.get("lag", "2")), int(ARGS.get("preg", "40")), int(ARGS.get("creg", "96"))


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
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_13')}.py"
    check = "--check" in FLAGS
    GM.check(base.exists(), f"{base.name} not found")
    if not check:
        GM.check(not OUT.exists(), f"{OUT.relative_to(ROOT)} exists; pick another --out")
    b = base.read_text(encoding="utf-8")
    bn = b.split(NL)[1]
    chk_base = b
    if not stub:
        try:
            chk_base = GM.G37.generate()[1]
        except (SystemExit, Exception) as ex:   # noqa: BLE001
            print(f"[gen_k3_vfa3] note: k3_v37 chain not regenerable ({ex!r:.100}); checker baseline = {base.name}")
    opts = [f"PVT {'on' if PVT else 'off'} (V^T on the producer warps, lag {LAG}, regs {PREG} / {CREG})"] + ["VSM"] * VSM + ["lab arms"] * LAB
    nn = f"# nonce {TAG} ({base.stem} + VFA3 verify-full: {', '.join(opts)})"
    rep = [f"base {base.name} ({len(b.encode())} B)", f"fa16_patches.apply(vsm={VSM}, lab=False); vfa3_patches.apply(pvt={PVT}, lag={LAG}, preg={PREG}, creg={CREG}, lab={LAB})"]
    t = FP.apply(b.replace(bn, nn), vsm=VSM, lab=False)
    t = V3.apply(t, pvt=PVT, lag=LAG, preg=PREG, creg=CREG, lab=LAB)
    if LAB:
        t = V3._one(t, "def _vfa_build(lab=False):" + NL, "def _vfa_build(lab=True):" + NL, "vfa build lab")
    s = GM.finish(t, nn, TAG, chk_base, rep)
    GM.check(GM.finish(s, nn, TAG, chk_base, []) == s, "output is not a finish() fixed point")
    for r in rep:
        print(f"[gen_k3_vfa3] OK  {r}")
    sha = hashlib.sha1(s.encode()).hexdigest()[:12]
    if check:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[gen_k3_vfa3] check OK {OUT.name} sha1 {sha}")
        return
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vfa3] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {sha})")


if __name__ == "__main__":
    main()

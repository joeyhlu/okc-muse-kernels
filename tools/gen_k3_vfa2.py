"""gen_k3_vfa2.py: kernels/lab/<out>.py = a layer file (default k3_gc_10) + VFA2 (tools/vfa2_patches.py: VFA verify-full PDL entry,
VFA_PW) [+ FA16-VSM (tools/fa16_patches.py, V^T constants in smem)].

Same contract as gen_k3_fa16.py: every anchor asserted unique, the text goes through gen_k3_m32.finish (strip + AST checks, ASCII,
nonce, judge-checker scan <= base, < 1 MB), the output is a finish() fixed point, refuses to overwrite.
--lab: lab build (VFA_LAB arms incl. DIAG 16 = VSM; _vfa_build(lab=True)); --pw=0 ships VFA_PW False (default True).

    python -I tools/gen_k3_vfa2.py --out=k3_vfa2_1 [--vsm] [--lab] [--pw=0|1] [--base=k3_gc_10] [--check]
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
import vfa2_patches as VP  # noqa: E402

OUT_NAME = ARGS.get("out", "k3_vfa2_lab")
OUT = ROOT / f"kernels/lab/{OUT_NAME}.py"
TAG = OUT_NAME.replace("_", "-")
VSM, LAB, PW = "--vsm" in FLAGS, "--lab" in FLAGS, ARGS.get("pw", "1") == "1"


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
    base = ROOT / f"kernels/muse_decode_layer/{ARGS.get('base', 'k3_gc_10')}.py"
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
            print(f"[gen_k3_vfa2] note: k3_v37 chain not regenerable ({ex!r:.100}); checker baseline = {base.name}")
    opts = [f"PW {'on' if PW else 'off (knob)'} (VFA PDL entry: prologue TF, producer pre-wait on old-key tiles)"] + ["VSM"] * VSM + ["lab arms"] * LAB
    nn = f"# nonce {TAG} ({base.stem} + VFA2 verify-full: {', '.join(opts)})"
    rep = [f"base {base.name} ({len(b.encode())} B)", f"vfa2_patches.apply_pw(default={PW}); fa16_patches.apply(vsm={VSM}, lab={LAB})"]
    t = VP.apply_pw(b.replace(bn, nn), default=PW)
    if VSM or LAB:
        t = FP.apply(t, vsm=VSM, lab=LAB)
    if LAB:
        t = VP._one(t, "def _vfa_build(lab=False):" + NL, "def _vfa_build(lab=True):" + NL, "vfa build lab")
    s = GM.finish(t, nn, TAG, chk_base, rep)
    GM.check(GM.finish(s, nn, TAG, chk_base, []) == s, "output is not a finish() fixed point")
    for r in rep:
        print(f"[gen_k3_vfa2] OK  {r}")
    sha = hashlib.sha1(s.encode()).hexdigest()[:12]
    if check:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs")
        print(f"[gen_k3_vfa2] check OK {OUT.name} sha1 {sha}")
        return
    OUT.write_text(s, encoding="utf-8", newline=NL)
    print(f"[gen_k3_vfa2] wrote {OUT.relative_to(ROOT)} ({len(s.encode())} B, sha1 {sha})")


if __name__ == "__main__":
    main()

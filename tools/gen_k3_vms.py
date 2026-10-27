"""gen_k3_vms.py: kernels/muse_decode_layer/k3_vms_<n>.py = k3_m32_3 + the VMS one-launch MoE route at T <= MAXT
(tools/vms_patches.py, notes/study/vm_small.md).

Chain: tools/gen_k3_m32_3.py's k3_m32_3 pre-strip text (k3_m32_2 chain + SK1; its strip is asserted equal to k3_m32_3.py on
disk), the new nonce, vms_patches.apply_vms, gen_k3_m32.finish (strip_layer + strip_cxx with AST-equivalence checks, compiles,
ASCII, line 1, nonce, no duplicate names, judge-checker scan <= base, < 1 MB). Writes the file and a byte-identical box copy
kernels/lab/k3_vms_<n>_ab.py.

    python -I tools/gen_k3_vms.py [--n=2] [--maxt=16] [--fl=0] [--kl1] [--check]   (--kl1 -> k3_vms_<n>k.py)

--n=1 reproduces k3_vms_1 byte-identically from the frozen lab revision kernels/lab/vms_moe_lab_v1.py (maxt 16, no fl);
--n >= 2 uses the live lab kernels/lab/vms_moe_lab.py. Other builders: stack on k3_vms_1 with vms1_text() (see its docstring).
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
CHECK = "--check" in sys.argv
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
N = int(ARGS.get("n", 1))
MAXT = int(ARGS.get("maxt", 16))
FL = int(ARGS.get("fl", 0))
KL1 = "--kl1" in sys.argv
_argv, sys.argv = sys.argv, [sys.argv[0]]
import gen_k3_m32_3 as G3  # noqa: E402
sys.argv = _argv
import vms_patches  # noqa: E402

GM = G3.GM
SFX = "k" if KL1 else ""
OUT = ROOT / f"kernels/muse_decode_layer/k3_vms_{N}{SFX}.py"
LAB = ROOT / f"kernels/lab/k3_vms_{N}{SFX}_ab.py"
TAG = f"k3-vms-{N}{SFX}"
NONCE1 = ("# nonce k3-vms-1 (k3_m32_3 + VMS: one-launch MoE at T <= 16 (36 KB full-K units, continuous A -> B weight ring, "
          "per-job doneA, in-launch finalize), self-tested vs K2A with fallback)")
NONCE = NONCE1 if (N == 1 and not KL1) else (NONCE1.replace("k3-vms-1 (", f"{TAG} (KL1 legacy K2A strip + ") if N == 1 else
    f"# nonce {TAG} (k3_m32_3 + VMS: one-launch MoE at T <= {MAXT} (36 KB full-K units, continuous A -> B weight ring, "
    f"per-job doneA, in-launch finalize, flags {FL}{', KL1 legacy K2A strip' if KL1 else ''}), self-tested vs K2A with fallback)")


def m32_3_text():
    """k3_m32_3's pre-strip text, rebuilt exactly as tools/gen_k3_m32_3.py builds it"""
    v39, base, _ = GM.G37.generate()
    GM.check(GM.G37.V39 and v39 == GM.GD.V39.read_text(encoding="utf-8"), "gen_k3_v37 --v39 does not reproduce k3_v39.py")
    gd = G3.gd_patches.apply(v39)
    gd = gd.replace(GM.G37.NONCE, GM.GD.NONCE)
    GM.check(GM.strip_layer.strip(gd) == GM.GD1.read_text(encoding="utf-8"), "the gd chain does not reproduce k3_gd_1.py")
    path2, tag2, nonce2, px2 = GM.OUTS[2]
    t2 = G3.m32_patches.apply(gd, px=px2).replace(GM.GD.NONCE, nonce2)
    t = G3.sk1_patches.apply_sk1(t2.replace(nonce2, G3.NONCE))
    return t, base


def vms1_text(kl1=False):
    """Base for patch modules that stack on k3_vms_1: returns (pre-strip text of k3_vms_1, base, NONCE1).
    kl1=True also applies the KL1 legacy K2A strip (vms_patches.apply_kl1: ~100 KB more room; v3.3 -> r1 Marlin fallback).
    Usage: t, base, n1 = gen_k3_vms.vms1_text(kl1=True); t = my_patch(t.replace(n1, MY_NONCE));
    s = gen_k3_vms.GM.finish(t, MY_NONCE, MY_TAG, base, rep)   (MY_TAG must appear once, in MY_NONCE)."""
    t3, base = m32_3_text()
    t = vms_patches.apply_vms(t3.replace(G3.NONCE, NONCE1), maxt=16, fl=0, lab=vms_patches.LAB_V1)
    if kl1:
        t = vms_patches.apply_kl1(t)
    return t, base, NONCE1


def main():
    rep = []
    t3, base = m32_3_text()
    s3 = GM.finish(t3, G3.NONCE, G3.TAG, base, [])
    disk = G3.OUT.read_text(encoding="utf-8")
    nocom = lambda s: [ln for ln in s.split("\n") if not ln.strip().startswith("//")]  # noqa: E731
    GM.check(s3 == disk or nocom(s3) == nocom(disk), "the k3_m32_3 chain does not reproduce k3_m32_3.py on disk")
    rep.append(f"k3_m32_3 reproduced ({'byte-identical' if s3 == disk else 'up to // lines'}: {len(disk.encode())} B)")
    t = vms_patches.apply_vms(t3.replace(G3.NONCE, NONCE), maxt=MAXT, fl=FL, lab=vms_patches.LAB_V1 if N == 1 else None)
    rep.append(f"VMS applied (T <= {MAXT}): pre-strip {len(t.encode())} B")
    if KL1:
        t = vms_patches.apply_kl1(t, log=rep.append)
    s = GM.finish(t, NONCE, TAG, base, rep)
    for r in rep:
        print(f"[gen_k3_vms] OK  {r}")
    if CHECK:
        GM.check(OUT.exists() and OUT.read_text(encoding="utf-8") == s, f"{OUT.name} on disk differs (regenerate)")
        print(f"[gen_k3_vms] --check: {OUT.name} up to date")
        return 0
    OUT.write_text(s, encoding="utf-8", newline="\n")
    LAB.write_text(s, encoding="utf-8", newline="\n")
    print(f"[gen_k3_vms] wrote {OUT.relative_to(ROOT)} and {LAB.relative_to(ROOT)} ({len(s.encode())} B, "
          f"sha1 {hashlib.sha1(s.encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

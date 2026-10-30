"""Local sm_90a gate for tools/vxp_patches.py on mb_19's layer-VMS source (MB_LV_CUDA): patched-at-defaults SASS == base, then
regs / spills / instruction counts for VMS_L2H 0, VMS_XPF 1 and both with VMS_FX 1 (WSL nvcc 13.4, compile only).
    MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
        python "<repo>/tools/vxp_local_test.py" [src=mb_19]
"""
import pathlib
import re
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import vaq_local_test as T  # noqa: E402
import vxp_patches as P  # noqa: E402

SHIP = ["-DVMS_NS=3", "-DVMS_RO=1"]


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    s = (ROOT / f"kernels/muse_moe_w4a16/{opt.get('src', 'mb_19')}.py").read_text(encoding="utf-8")
    q = "MB_LV_CUDA = r'''"
    a = s.index(q) + len(q)
    src0 = s[a:s.index("'''", a)]
    src = P.apply(src0)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        rc0, ln0, b0 = T.sm90(src0, SHIP, td, "orig")
        rc1, ln1, b1 = T.sm90(src, SHIP, td, "def0")
        same = rc0 == 0 and rc1 == 0 and b0 == b1
        ok &= same
        print(f"[vxp] base: {' | '.join(ln0)}")
        print(f"[vxp] patched at defaults: SASS {'IDENTICAL' if same else 'DIFFERS'}")
        for nm, d in (("l2h0", ["-DVMS_L2H=0"]), ("xpf1", ["-DVMS_XPF=1"]), ("fx1l2h0", ["-DVMS_FX=1", "-DVMS_L2H=0"]),
                      ("fx1xpf1", ["-DVMS_FX=1", "-DVMS_XPF=1"])):
            rc, ln, b = T.sm90(src, SHIP + d, td, nm)
            spill = [x for x in ln if "spill" in x]
            good = rc == 0 and all(" 0 bytes spill stores" in x and " 0 bytes spill loads" in x for x in spill)
            ok &= good
            extra = "" if b is None else (f" | instr {b.count(chr(10)) + 1} (base {b0.count(chr(10)) + 1}) CCTL/prefetch "
                                          f"{T.count(b, r'CCTL|PREFETCH')} UBLKCP {T.count(b, 'UBLKCP')}")
            print(f"[vxp] {nm}: rc {rc} {'ok' if good else 'CHECK'} | {' | '.join(ln)}{extra}")
    print(f"[vxp] {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

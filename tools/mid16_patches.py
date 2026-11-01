"""R-mid16-1: alygg77's MOE_MID build (M <= 16, Apache-2.0, already vendored as our HYBLO source) as an in-layer arm at T = 16
on a REPAIRED stack (R-hyblo-1's +4.7 / +5.1 us MID kill ran with production guards tripped in every arm).

  python tools/mid16_patches.py [--base=kernels/lab/k3_vmsx_ship_rt.py] -> kernels/lab/k3_mid16_ab.py

Adds ('mid', 16, L2H 1 / CPF 0 / WSBLK 2 / SPLIT 1) to _HYBLO_KEYS (tools/hyblo_patches.py BUILDS) and its six self-tests.
HYBLO_T stays (1, 8) = ship dispatch; the A/B variant sets HYBLO_T = [1, 8, 16].
"""
import sys
from pathlib import Path

base = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--base=")), "kernels/lab/k3_vmsx_ship_rt.py"))
s = base.read_text(encoding="utf-8")


def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor x{s.count(old)}: {old[:80]!r}"
    s = s.replace(old, new)


sub("('tiny2', 8, ('-DOKC_L2H=1', '-DOKC_CPF=0', '-DOKC_WSBLK=1', '-DOKC_SPLIT=0')))",
    "('tiny2', 8, ('-DOKC_L2H=1', '-DOKC_CPF=0', '-DOKC_WSBLK=1', '-DOKC_SPLIT=0')), "
    "('mid', 16, ('-DOKC_L2H=1', '-DOKC_CPF=0', '-DOKC_WSBLK=2', '-DOKC_SPLIT=1')))")
sub("(8, 92, 'conc'), (8, 96, 'none'))",
    "(8, 92, 'conc'), (8, 96, 'none'), (9, 88, 'skewed'), (12, 89, 'normal'), (16, 90, 'skewed'), (16, 91, 'normal'), "
    "(16, 93, 'conc'), (16, 97, 'none'))")
first, rest = s.split("\n", 1)
assert first.startswith("# okc: problem=muse_decode_layer")
s = f"{first}\n# mid16 lab R-mid16-1 (tools/mid16_patches.py): alygg77 MOE_MID build added to HYBLO; HYBLO_T selects\n{rest}"
out = Path("kernels/lab/k3_mid16_ab.py")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(s)
print(out, out.stat().st_size)

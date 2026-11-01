"""R-sk2t-1: early PDL trigger for the sk2 qkv GEMV at T = 32 so the attention kernel (ks3 sliding / HYA full) can start its
pre-wait KV gather while qkv still streams (MEASURED nsys box_out/rt1: d32s ks3 starts 11.9 us = after sk2 qkv ends 10.8 us;
at d16s, where the GD qkv triggers early, ks3 starts at 2.4 us).

  python tools/sk2t_patches.py [--base=kernels/lab/k3_diet_rq128.py] -> kernels/lab/k3_sk2t_ab.py

sk2 `pdl` bit 16 (new) = trigger dependents right after the (no-op for the first kernel) griddepcontrol.wait instead of after the
main loop; the low 4 bits keep their meaning (weight stages pre-issued before the wait). Runtime arm: SK2_PDL["qkv"] = 16 | P.
SK2_PDL defaults unchanged ({"qkv": 0, "unpooler": 2}) = ship behaviour (same instructions executed: the new tests are on pdl).
"""
import sys
from pathlib import Path

base = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--base=")), "kernels/lab/k3_diet_rq128.py"))
s = base.read_text(encoding="utf-8")


def sub(old, new, n=1):
    global s
    assert s.count(old) == n, f"anchor x{s.count(old)} (want {n}): {old[:80]!r}"
    s = s.replace(old, new)


sub("const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;",
    "const int P = (CL == 1 && (pdl & 15) > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;")
# the second (all-thread) wait is followed by the producer branch; trigger right after it when bit 16 is set
old_wait2 = 'if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;\n__syncthreads();\nif (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");\n'
sub(old_wait2, old_wait2 + 'if (pdl & 16) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n')
sub('if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\nif constexpr (COMP) mbar_wait(cbar, 0);',
    'if (pdl && !(pdl & 16)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\nif constexpr (COMP) mbar_wait(cbar, 0);')
first, rest = s.split("\n", 1)
assert first.startswith("# okc: problem=muse_decode_layer")
s = f"{first}\n# sk2t lab R-sk2t-1 (tools/sk2t_patches.py): sk2 pdl bit 16 = early dependent trigger\n{rest}"
out = Path("kernels/lab/k3_sk2t_ab.py")
with open(out, "w", encoding="utf-8", newline="") as fh:
    fh.write(s)
print(out, out.stat().st_size)

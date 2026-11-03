"""SK2T lab file: early griddepcontrol.launch_dependents in sk2 (alygg77's vendored GEMV, notices kept) so the next kernel (KS3 at d32s,
HYA full at d32f) can become resident and run its pre-wait gathers under the qkv GEMV.

python tools/sk2t_patches.py [base=kernels/lab/k3_clean_ab.py] [out=kernels/lab/k3_sk2t_ab.py]
SK2_PDL bit 16 (new): trigger dependents at kernel entry. The low 4 bits keep their meaning (number of pre-wait stages; > 0 also sets the
programmatic launch attribute and the griddepcontrol.wait). SK2_PDL["qkv"] = 16 = early trigger WITHOUT programmatic launch of the first
kernel of the layer (nothing of the layer runs before the timed region starts). Defaults unchanged (qkv 0, unpooler 2) -> byte-identical
behaviour.
"""
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_clean_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_sk2t_ab.py")
s = base.read_text(encoding="utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:100])
    s = s.replace(a, b)


once("const int P = (CL == 1 && pdl > 0) ? min(pdl, min(NSTAGE, nst)) : 1;\nif (tid == 0) {",
     "const int P = (CL == 1 && (pdl & 15) > 0) ? min(pdl & 15, min(NSTAGE, nst)) : 1;\n"
     "if (pdl & 16) asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");   // SK2T: early trigger\n"
     "if (tid == 0) {")
a = 'if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");'
assert s.count(a) == 2, s.count(a)
s = s.replace(a, 'if (pdl & 15) asm volatile("griddepcontrol.wait;" ::: "memory");')
once('if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\nif constexpr (COMP) mbar_wait(cbar, 0);',
     'if ((pdl & 15) && !(pdl & 16)) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\nif constexpr (COMP) mbar_wait(cbar, 0);')
once("const int pdl = g_sk2_pdl;\nif (pdl > 0) {", "const int pdl = g_sk2_pdl;\nif ((pdl & 15) > 0) {")
out.write_text(s, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(s.encode())} B)")

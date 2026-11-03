"""QFP (timing-only probe, never shipped): KS1_TRIG bit 8 moves KS1's griddepcontrol.wait from before the q reads to just before the final
output store, so KS1's post-qkv work overlaps the qkv GEMV. Outputs of the probe arm are INVALID (stale q); it bounds what any q-first
design (Opus QFIRST) could save at d1s / v1s. Defaults (KS1_TRIG 1 / 2) are byte-identical in behaviour.
python tools/qfp_patches.py [base=kernels/lab/k3_clean_ab.py] [out=kernels/lab/k3_qfp_ab.py]"""
import sys
from pathlib import Path
base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_clean_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_qfp_ab.py")
s = base.read_text(encoding="utf-8")
def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
s = s.replace("if (p.trig == 2) pdl_trigger();", "if ((p.trig & 7) == 2) pdl_trigger();")   # KS1 and KS3: identical for trig <= 7
once("KS1P(1);\npdl_wait();\nif (p.trig == 1) pdl_trigger();", "KS1P(1);\nif (!(p.trig & 8)) pdl_wait();\nif ((p.trig & 7) == 1) pdl_trigger();")
once("vsl_fin1<DPT, UPR>(a, den, graw, p.eps, p.out", "if (p.trig & 8) pdl_wait();   // QFP probe\nvsl_fin1<DPT, UPR>(a, den, graw, p.eps, p.out")
once("__nv_bfloat16* op = p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64)", "if (p.trig & 8) pdl_wait();   // QFP probe\n__nv_bfloat16* op = p.out + (size_t)(tb + tg * TPG + (mr >> 3)) * (HQ * 64)")
out.write_text(s, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(s.encode())} B)")

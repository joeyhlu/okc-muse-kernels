"""VNSF lab file: VML phase-A split close without the redundant fence.acq_rel.gpu (Opus VML code review, Finding 1).

python tools/vnsf_patches.py [base=kernels/muse_decode_layer/k3_pure_c1.py] [out=kernels/lab/k3_vnsf_ab.py]
In seg<false, N> the finisher of a stream-K split runs bar_sync -> (thread 0) atom.acq_rel.gpu on the split counter -> bar_sync ->
fence.acq_rel.gpu -> __ldcg read-back of the partials (vse_take). The acquire RMW by thread 0 plus the CTA barrier already orders
the read-back after every contributor's release (contributors: partial stores / red.adds -> bar_sync -> atom.acq_rel.gpu), and every
read-back is an L2 (.cg) load, so the extra gpu-scope fence only costs latency (measured MEMBAR ~0.89 us per finishing split step,
vml_consumer_budget.md s5). VM_NOSF=1 drops that one fence; default 0 = byte-identical behaviour. New variants bf2nf / w72nf =
bf2 / w72 + -DVM_NOSF=1, self-tested in prepare when listed in VML_ABVARS; arms pick them per bucket through VML_BVAR at call time.
Numerics: identical (same loads, same adds, same order).
"""
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/muse_decode_layer/k3_pure_c1.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_vnsf_ab.py")
s = base.read_bytes().decode("utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:120])
    s = s.replace(a, b)


once("SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\nfence_acq_rel();\n#if VSE_HOLD\n",
     "SEGMK(const long long t_rb = smk.on ? gtime() : 0;)\n#if !VM_NOSF\nfence_acq_rel();\n#endif\n#if VSE_HOLD\n")
once("#ifndef VSE_PS\n#define VSE_PS 0\n", "#ifndef VM_NOSF\n#define VM_NOSF 0\n#endif\n#ifndef VSE_PS\n#define VSE_PS 0\n")
once("_VML={\"mod\":None,\"mods\":{},",
     "_VML_DEFS[\"bf2nf\"]=_VML_DEFS[\"bf2\"]+(\"-DVM_NOSF=1\",)  # VNSF lab\n"
     "_VML_DEFS[\"w72nf\"]=_VML_DEFS[\"w72\"]+(\"-DVM_NOSF=1\",)\n"
     "_VML={\"mod\":None,\"mods\":{},")
once("VML_ABVARS=()\n", "VML_ABVARS=('bf2nf',)  # VNSF lab: self-test bf2nf at the buckets listed in VML_BVAR\n")
once("VML_BVAR={256:'w72',128:'w72'}\n", "VML_BVAR={256:'w72',128:'w72',32:'bf2'}  # VNSF lab: bucket 32 listed so bf2nf is self-tested there\n")
out.write_bytes(s.encode("utf-8"))
print(f"wrote {out} ({len(s.encode())} B)")

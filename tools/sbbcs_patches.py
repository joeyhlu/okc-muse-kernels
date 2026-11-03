"""SBBCS lab file: SBB norm-boundary cluster size 8 / 4 (shipped 16) selectable per T at call time.

python tools/sbbcs_patches.py [base=kernels/lab/k3_clean_ab.py] [out=kernels/lab/k3_sbbcs_ab.py]
Measured (box_out/c81/kernels.txt): sbb exclusive 4.3 us at v1s / 3.9 at v1f vs 2.3 at d1s; R-lat-0: at T=8 the 128 sbb CTAs (8 rows x 16-CTA
clusters) land crammed on 32-48 SMs. Smaller clusters (8 CTAs x 128 threads, 4 x 256) are easier to place. SB_CS is a #define used
generically (threads = 8192 / (SB_CS * 8), partials[SB_CS], expect_tx (SB_CS - 1) * 16, cluster rank); the extra modules are the same
source with another SB_CS, compiled in prepare. SBB_CS1 / SBB_CS2 = {bucket: 16 | 8 | 4} pick the module for the attention boundary
(sbb1) and the T <= 8 final norm (VF sbb2). Defaults 16 everywhere = byte-identical behaviour. Reduction grouping changes with SB_CS
(not bitwise vs 16; same FP32 math).
"""
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_clean_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_sbbcs_ab.py")
s = base.read_text(encoding="utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:100])
    s = s.replace(a, b)


assert s.count("#define SB_CS 16") == 1
once(' _pt1_wait("sbb1")\n if _SBB1["mod"]is not None:\n  return _SBB1["mod"]\n from torch.utils.cpp_extension import load_inline\n',
     ' _pt1_wait("sbb1")\n from torch.utils.cpp_extension import load_inline\n if not _SBBCS:\n  _sbbcs_build(load_inline)   # SBBCS: also when PT1 prebuilt the main module\n'
     ' if _SBB1["mod"]is not None:\n  return _SBB1["mod"]\n')
once(' st["mod"].sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,rt,pw,qw,al,bt,pe,qe,',
     ' _sbbcs_mod(st["mod"],SBB_CS1,tokens).sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,rt,pw,qw,al,bt,pe,qe,')
once(' st["mod"].sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,st["e"],pw,qw,al,bt,pe,qe,isc|2)\n if("sbb2",tokens)not in _VF["seen"]:',
     ' _sbbcs_mod(st["mod"],SBB_CS2,tokens).sbb1_run(branch.contiguous(),stream.contiguous(),out,ns,st["e"],pw,qw,al,bt,pe,qe,isc|2)\n if("sbb2",tokens)not in _VF["seen"]:')
s = s.rstrip("\n") + '''

# SBBCS (lab): SBB cluster size per bucket. Selectors read at call / capture time.
SBB_CS1={1:16,8:16,16:16,32:16}
SBB_CS2={1:16,8:16}
_SBBCS={}
def _sbbcs_build(load_inline):
 for cs in (8,4):
  t1=time.time()
  src=SBB1_CU.replace("#define SB_CS 16",f"#define SB_CS {cs}")
  assert src!=SBB1_CU
  _SBBCS[cs]=load_inline(f"sbb1_k3t_cs{cs}",cpp_sources=SBB1_DECL,cuda_sources=src+SBB1_HOST,functions=["sbb1_run","vf_router32"],
   extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a"],verbose=False)
  print(f"[sbbcs] built SB_CS {cs} ({time.time()-t1:.0f}s)",flush=True)
def _sbbcs_mod(mod,sel,tokens):
 cs=sel.get(_k2a_bucket_of(tokens),16)
 if cs==16:
  return mod
 return _SBBCS[cs]
'''
out.write_text(s, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(s.encode())} B)")

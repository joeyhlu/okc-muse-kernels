"""TMAPG8 lab file: HYA bulk page-index prefetch (TMAPG_ON) beyond L == 1 (Opus API study notes/study/evo/att/opus_fullattn2_out.md).

python tools/tmapg_patches.py [base=kernels/lab/k3_clean_ab.py] [out=kernels/lab/k3_tmapg_ab.py]
Adds HYA_TMAPG (call-time, default 0 = shipped): 1 = TMAPG_ON also for FLAT kernels (verify full v16f / v32f), 2 = TMAPG_ON for every L.
Both variants are compiled in prepare (inside _hya_build) as separate modules; the call site picks one. Defaults are byte-identical.
"""
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_clean_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_tmapg_ab.py")
s = base.read_text(encoding="utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:100])
    s = s.replace(a, b)


ANCHOR = "constexpr bool TMAPG_ON = (L == 1);"
assert s.count(ANCHOR) == 1
once(' _HYA["build_s"]=time.time()-t0\n return _HYA["mod"]\n',
     ' _HYA["build_s"]=time.time()-t0\n _hya_tp_build(load_inline,tag)\n return _HYA["mod"]\n')
once(' _HYA["mod"].run(qkv,data["seq_lens"],data["cache_loc"],data["k_cache"],data["v_cache"],pt,data["k_scale"],',
     ' _hya_tp_pick().run(qkv,data["seq_lens"],data["cache_loc"],data["k_cache"],data["v_cache"],pt,data["k_scale"],')
s = s.rstrip("\n") + '''

# TMAPG8 (lab): bulk page-index prefetch for L = 8 producers. Selector read at call / capture time.
HYA_TMAPG=0
_HYA_TP={}
_HYA_TP_SRC={1:"constexpr bool TMAPG_ON = (L == 1) || FLAT;",2:"constexpr bool TMAPG_ON = true;"}
def _hya_tp_build(load_inline,tag):
 for m,rep in _HYA_TP_SRC.items():
  src=HYA_CUDA_SRC.replace("constexpr bool TMAPG_ON = (L == 1);",rep)
  assert src!=HYA_CUDA_SRC
  t0=time.time()
  _HYA_TP[m]=load_inline(name=f"hya_attn_tp{m}_{tag}",cpp_sources=HYA_CPP_SRC,cuda_sources=src,
   functions=["run"],extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr"],verbose=False)
  print(f"[tmapg] built mode {m} ({time.time()-t0:.0f}s)",flush=True)
def _hya_tp_pick():
 if HYA_TMAPG:
  return _HYA_TP[HYA_TMAPG]
 return _HYA["mod"]
'''
out.write_text(s, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(s.encode())} B)")

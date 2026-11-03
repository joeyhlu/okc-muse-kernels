"""HYAST lab file: deeper KV ring for the L=1 flat HYA kernel (decode full; Opus API study notes/study/evo/att/opus_decodefull_out.md).

python tools/tmapg_patches.py [base=kernels/lab/k3_clean_ab.py] [out=kernels/lab/k3_hyast_ab.py]
Adds HYA_ST1 (call-time, default 0 = shipped): 4 or 5 = KV stages of the L=1 flat kernel (decode full d16f / d32f).
Both variants are compiled in prepare (inside _hya_build) as separate modules; the call site picks one. Defaults are byte-identical.
"""
import sys
from pathlib import Path

base = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_clean_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/lab/k3_hyast_ab.py")
s = base.read_text(encoding="utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:100])
    s = s.replace(a, b)


ANCHOR = "static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;"
assert s.count(ANCHOR) == 1
once(' _HYA["build_s"]=time.time()-t0\n return _HYA["mod"]\n',
     ' _HYA["build_s"]=time.time()-t0\n _hya_st_build(load_inline,tag)\n return _HYA["mod"]\n')
once(' _HYA["mod"].run(qkv,data["seq_lens"],data["cache_loc"],data["k_cache"],data["v_cache"],pt,data["k_scale"],',
     ' _hya_st_pick().run(qkv,data["seq_lens"],data["cache_loc"],data["k_cache"],data["v_cache"],pt,data["k_scale"],')
s = s.rstrip("\n") + '''

# HYAST (lab): L=1 flat KV ring depth 4 or 5 (shipped 3). Selector read at call / capture time.
HYA_ST1=0
_HYA_ST={}
_HYA_ST_SRC={4:"static constexpr int STAGES = F ? 4 : 3;",5:"static constexpr int STAGES = F ? ((L == 1) ? 5 : 4) : 3;"}
def _hya_st_build(load_inline,tag):
 for m,rep in _HYA_ST_SRC.items():
  src=HYA_CUDA_SRC.replace("static constexpr int STAGES = F ? ((L == 1) ? 3 : 4) : 3;",rep)
  assert src!=HYA_CUDA_SRC
  t0=time.time()
  _HYA_ST[m]=load_inline(name=f"hya_attn_st{m}_{tag}",cpp_sources=HYA_CPP_SRC,cuda_sources=src,
   functions=["run"],extra_cuda_cflags=["-O3","-gencode=arch=compute_90a,code=sm_90a","--expt-relaxed-constexpr"],verbose=False)
  print(f"[hyast] built mode {m} ({time.time()-t0:.0f}s)",flush=True)
def _hya_st_pick():
 if HYA_ST1:
  return _HYA_ST[HYA_ST1]
 return _HYA["mod"]
'''
out.write_text(s, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(s.encode())} B)")

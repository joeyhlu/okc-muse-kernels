"""Ship form of TINYS bits 7 (doneA single release poller + grid-barrier last-arriver skip + gen load on warp 2) for a judge check.

python tools/tinys_ship.py [lab=kernels/lab/k3_tinys_ab.py] [out=kernels/muse_decode_layer/k3_tinys7_c1.py] [nonce=tinys7-c1]
The shipped tiny / tiny2 HYBLO builds compile _TINYS_CUDA with -DOKC_TINYS=7 instead of _HYBLO_CUDA (same source otherwise; bits 0 =
byte-identical to ship), and the TINYS A/B bank is emptied, so prepare builds the same number of modules as k3_pure_c1. combo96:
T<=8 load-geomean x1.0040 / x1.0045 vs ship (marginal rider).
"""
import sys
from pathlib import Path

lab = Path(sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_tinys_ab.py")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "kernels/muse_decode_layer/k3_tinys7_c1.py")
nonce = sys.argv[3] if len(sys.argv) > 3 else "tinys7-c1"
s = lab.read_bytes().decode("utf-8")


def once(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:120])
    s = s.replace(a, b)


once("# nonce pure-c1 (", f"# nonce {nonce} (TINYS bits 7 on tiny / tiny2; ")
once(" for k,_,var in _HYBLO_KEYS: # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)\n"
     "  fl=list(_HYBLO_FLAGS)+list(var)\n",
     " for k,_,var in _HYBLO_KEYS: # one nvcc at a time (R-hovl-1: parallel extra builds froze a container)\n"
     "  fl=list(_HYBLO_FLAGS)+list(var)+[\"-DOKC_TINYS=7\",\"-std=c++20\"]  # TINYS bits 7 (tools/tinys_ship.py)\n")
once("   tag=hashlib.sha1((_HYB_MOE_CPP+_HYBLO_CUDA+\" \".join(fl)).encode()).hexdigest()[:12]\n"
     "   _HYBLO[\"mods\"][k]=load_inline(name=f\"hyblo_{k}_{tag}\",cpp_sources=_HYB_MOE_CPP,cuda_sources=_HYBLO_CUDA,\n"
     "   functions=[\"okc_run\",\"okc_set_pdl\"],extra_cuda_cflags=fl,verbose=False)\n",
     "   tag=hashlib.sha1((_HYB_MOE_CPP+_TINYS_CUDA+\" \".join(fl)).encode()).hexdigest()[:12]\n"
     "   _HYBLO[\"mods\"][k]=load_inline(name=f\"hyblo_{k}_{tag}\",cpp_sources=_HYB_MOE_CPP,cuda_sources=_TINYS_CUDA,\n"
     "   functions=[\"okc_run\",\"okc_set_pdl\"],extra_cflags=[\"-std=c++20\"],extra_cuda_cflags=fl,verbose=False)\n")
once("TINYS_BUILD_BITS = (3, 7, 8, 15)  # lead: trimmed to the timed arms for combo96\n", "TINYS_BUILD_BITS = ()\n")
out.write_bytes(s.encode("utf-8"))
print(f"wrote {out} ({len(s.encode())} B)")

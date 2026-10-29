"""l2s_sass_probe.py (R-l2s-0): SASS probe for a pre-wait L2 prefetch in GD. Writes a copy of <layer.py> whose gd_kernel issues
prefetch.global.L2 over A.w32 just before pdl_wait() (mode loop = strided loop, line = 2 straight-line prefetches), then run the gate:
  python -I tools/l2s_sass_probe.py <layer.py> <out.py> loop|line
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" python "<repo>/tools/rpo_sass.py" <out.py> [OUTDIR]
rpo_sass prints "L2PF above A below B" per instance. Never runs on a GPU (A.w32 / A.sst2 are reused as pointer / byte count)."""
import sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
mode = sys.argv[3]
old = "  pdl_wait();\n  if constexpr (VF >= 1) {\n    if (A.marks && threadIdx.x == 0) A.marks[vf_c * 4 + 2]"
assert src.count(old) == 1
if mode == "loop":
    ins = ("  for (long long o_ = ((long long)blockIdx.x * NT + threadIdx.x) * 128; o_ < A.sst2; o_ += (long long)gridDim.x * NT * 128)\n"
           "    asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(reinterpret_cast<const char*>(A.w32) + o_));\n")
else:
    ins = ("  { const char* p_ = reinterpret_cast<const char*>(A.w32) + ((size_t)blockIdx.x * NT + threadIdx.x) * 128;\n"
           "    asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(p_));\n"
           "    asm volatile(\"prefetch.global.L2 [%0];\" :: \"l\"(p_ + (size_t)gridDim.x * NT * 128)); }\n")
src = src.replace(old, ins + old)
pathlib.Path(sys.argv[2]).write_text(src, encoding="utf-8")

"""Compile a file's K1 module for sm_90a with ptxas -v; print va1_main resource lines. python ax2_sm90.py <file.py> <outdir>"""
import ast
import pathlib
import re
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tree = ast.parse(path.read_text(encoding="utf-8"))
srcs = {}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("K1_CUDA_SRC",):
        srcs[node.targets[0].id] = node.value.value
cu = out / "k1.cu"
cu.write_text(srcs["K1_CUDA_SRC"], encoding="utf-8")
import torch.utils.cpp_extension as ce
inc = ce.include_paths(device_type="cuda") if "device_type" in ce.include_paths.__code__.co_varnames else ce.include_paths(cuda=True)
cmd = ["nvcc", "-std=c++20", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v", "--expt-relaxed-constexpr",
       "-c", str(cu), "-o", str(out / "k1.o"), "-D_GLIBCXX_USE_CXX11_ABI=1"]
for i in inc:
    cmd += ["-I", i]
import sysconfig
cmd += ["-I", sysconfig.get_paths()["include"]]
r = subprocess.run(cmd, capture_output=True, text=True)
log = r.stdout + r.stderr
(out / "ptxas.log").write_text(log)
print("rc", r.returncode)
lines = log.splitlines()
for i, l in enumerate(lines):
    if "Compiling entry function" in l and "va1_main" in l:
        name = re.search(r"'([^']+)'", l).group(1)
        print(name, "|", " ".join(x.strip() for x in lines[i + 1:i + 4] if "ptxas" in x and ("registers" in x or "spill" in x)))
if r.returncode:
    print("\n".join(lines[-40:]))

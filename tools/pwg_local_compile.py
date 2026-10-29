"""pwg_local_compile.py: compile-only ptxas check (sm_90a, WSL, no GPU) of GD unpooler / cat instances at T 16 / 32 with
pre-wait stage depth ST swept (R-pwg). Prints per instance: key, regs, spill bytes.
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/pwg_local_compile.py" kernels/muse_decode_layer/k3_gc_16.py "1,2,3,4"
"""
import os
import re
import subprocess
import sys
import tempfile
import pathlib
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
STS = [int(s, 0) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4").split(",")]
src = pathlib.Path(__file__).resolve().parent / "gdd_local_compile.py"
code = src.read_text().split("keys = mod._gd_keys()")[0]
exec(compile(code, str(src), "exec"))
keys = []
for (proj, b), rt in sorted(mod.GD_ROUTE.items(), key=str):
    if proj == "unpooler" and b in (16, 32):
        for st in STS:
            if (st & 15) + (st >> 4) <= rt[3]:
                keys.append(("unp", b, st, mod._gd_key(*mod._gd_nk(proj, b), rt[:4] + (st,) + rt[5:])))
for b, rt in sorted(mod.GD_CAT_ROUTE.items()):
    for st in STS:
        if (st & 15) + (st >> 4) <= rt[3]:
            keys.append(("cat", b, st, mod._gd_cat_key(rt[:4] + (st,) + rt[5:])))
ks = [k[3] for k in keys]
inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(ks))
info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(ks))
cuda = mod.GD_CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)
bd = tempfile.mkdtemp(prefix="pwgc_")
cu = os.path.join(bd, "gd.cu")
open(cu, "w").write("#include <torch/extension.h>\n" + cuda if "torch/extension.h" not in cuda else cuda)
import torch.utils.cpp_extension as ce
inc = sum([["-I", p] for p in ce.include_paths(device_type="cuda")], [])
import sysconfig
inc += ["-I", sysconfig.get_paths()["include"]]
r = subprocess.run(["nvcc", "-c", cu, "-o", os.path.join(bd, "gd.o"), "-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
                    "-Xptxas=-v", "-D_GLIBCXX_USE_CXX11_ABI=1", "--expt-relaxed-constexpr"] + inc, capture_output=True, text=True)
out = r.stderr.splitlines()
if r.returncode:
    print("\n".join(out[-15:]))
    sys.exit(1)
fn, rows = None, {}
for ln in out:
    m = re.search(r"Compiling entry function '(\S+)'", ln)
    if m:
        fn = m.group(1)
    m = re.search(r"(\d+) bytes spill stores", ln)
    if m and fn:
        rows.setdefault(fn, {})["sp"] = int(m.group(1))
    m = re.search(r"Used (\d+) registers", ln)
    if m and fn:
        rows.setdefault(fn, {})["r"] = int(m.group(1))
for tag, b, st, k in keys:
    mg = [f for f in rows if "gd_kernel" in f and ("Li%dELi%dELi%dELi%dELi%dELi%dELi%dELi%d" % k[1:9]) in f]
    v = rows[mg[0]] if mg else {}
    print(f"[pwgc] {tag} T{b} ST{st:#x} key {k}: regs {v.get('r')} spill {v.get('sp')}")

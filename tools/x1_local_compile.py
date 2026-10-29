"""x1_local_compile.py: sm_90a compile-only check of the VML module in a k3 file (WSL cudaenv; no H200 needed).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/x1_local_compile.py" <k3 file> [variant ...]

Execs only the VML source block of the file (from `_vml_E_LOC, _vml_TOPK` through `_vml_FLAGS = `), builds each _VML_DEFS variant
(default bf2 bf2x w72 w72x) with -Xptxas=-v, prints vm_main registers / spills, and dumps the SASS of each .so to
nothing (no cuobjdump in cudaenv): filter the ptxas -v lines with
  grep -A2 "Compiling entry.*vm_main" | grep -E "entry|registers|spill"
"""
import pathlib
import re
import subprocess
import sys
import threading
import time
import hashlib

import torch  # noqa: F401
from torch.utils.cpp_extension import load_inline, _get_build_directory

src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
lines = src.split("\n")
a = next(i for i, l in enumerate(lines) if l.startswith("_vml_E_LOC, _vml_TOPK"))
b = next(i for i, l in enumerate(lines) if l.startswith("_vml_FLAGS = "))
g = {"torch": torch, "hashlib": hashlib, "time": time}
exec("\n".join(lines[a:b + 1]), g)
d0 = next(i for i, l in enumerate(lines) if l.startswith("_VML_DEFS = "))
d1 = next(i for i, l in enumerate(lines) if l.startswith("_VML = {"))
exec("\n".join(lines[d0:d1]), g)
for l in lines[d1:d1 + 8]:
    if l.startswith("_VML_DEFS["):
        exec(l, g)
names = sys.argv[2:] or ["bf2", "bf2x", "w72", "w72x"]
res = {}


def job(nm):
    t0 = time.time()
    fl = g["_vml_FLAGS"] + list(g["_VML_DEFS"][nm]) + ["-Xptxas=-v"]
    tag = hashlib.sha1((g["_vml_CPP"] + g["_vml_CUDA"] + " ".join(fl)).encode()).hexdigest()[:10]
    name = f"x1cc_{nm}_{tag}"
    try:
        load_inline(name=name, cpp_sources=g["_vml_CPP"], cuda_sources=g["_vml_CUDA"], functions=["vml_run", "vml_cfg"],
                    extra_cuda_cflags=fl, verbose=True)
        res[nm] = ("ok", time.time() - t0, name)
    except Exception as ex:  # noqa: BLE001
        res[nm] = ("FAIL", time.time() - t0, repr(ex)[-1500:])


ths = [threading.Thread(target=job, args=(nm,)) for nm in names]
for t in ths:
    t.start()
for t in ths:
    t.join()
for nm in names:
    st, dt, x = res[nm]
    if st != "ok":
        print(f"[x1-cc] {nm}: FAIL {dt:.0f}s {x}")
        continue
    print(f"[x1-cc] {nm}: ok {dt:.0f}s")

"""gdd_local_compile.py: compile-only check (sm_90a, WSL, no GPU) of a layer file's GD module (gd_kernel instances of its routes).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/gdd_local_compile.py" kernels/lab/k3_gdd_1.py
Prints the instance keys, ptxas spill warnings (if any) and the build time.
"""
import importlib.abc
import importlib.util
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
path = ROOT / sys.argv[1]


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):   # local box lacks sglang / sgl_kernel: names only
    def find_spec(self, name, p, target=None):
        if name.split(".")[0] not in sys.stdlib_module_names and not name.startswith(("torch", "numpy", "_")):
            return importlib.util.spec_from_loader(name, self, is_package=True)
        return None

    def create_module(self, spec):
        import types

        class M(types.ModuleType):
            def __getattr__(self, a):
                if a.startswith("__"):
                    raise AttributeError(a)
                return type(a, (), {"__init__": lambda *x, **y: None, "__call__": lambda *x, **y: None,
                                    "__getattr__": lambda s, b: None})
        return M(spec.name)

    def exec_module(self, m):
        m.__path__ = []


import importlib.abc  # noqa: E402

sys.meta_path.append(_Stub())   # last: only modules nothing else finds
spec = importlib.util.spec_from_file_location("gdd_lab", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
keys = mod._gd_keys()
import os  # noqa: E402
if os.environ.get("GDDC_KEYS"):   # e.g. "(0, 2, 1, 2,|(0, 4, 1, 2," : keep only keys whose str() contains one of these
    keys = [k for k in keys if any(s in str(k) for s in os.environ["GDDC_KEYS"].split("|"))]
for k in keys:
    print("[gddc] key", k)
inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(keys))
cuda = mod.GD_CUDA_SRC.replace("@INST@", inst).replace("@INFO@", info)
from torch.utils.cpp_extension import load_inline  # noqa: E402

t0 = time.time()
bd = tempfile.mkdtemp(prefix="gddc_")
fns = ["gd_run", "gd_info", "gd_unpack", "gdr_run", "gdr_info", "gdc_run", "gd_unpack_cat"]
if "gd_marks" in mod.GD_CPP_SRC:
    fns += ["gd_marks", "gd_marks_log"]
if "void gdb_run(" in mod.GD_CPP_SRC:   # BND (tools/gen_k3_bnd.py)
    fns += ["gdb_run"]
load_inline(name="gddc", cpp_sources=mod.GD_CPP_SRC, cuda_sources=cuda, functions=fns, build_directory=bd,
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-warn-spills"] + (["-Xptxas=-v"] if os.environ.get("GDDC_V") else []),
            verbose=bool(os.environ.get("GDDC_V")))
print(f"[gddc] OK build {time.time() - t0:.0f} s ({len(keys)} instances)")

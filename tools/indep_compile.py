"""Local sm_90a compile check without nvcc or torch: NVRTC (CUDA C++ -> PTX for compute_90a) + ptxas -v (registers,
spills, smem per kernel). For lab files that keep their device code in a torch-free string (kernels/lab/vks_probe.py
DEV_SRC, kernels/lab/mxs_probe.py DEV_SRC).

  pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12   (ptxas, libnvrtc, cuda headers)
  python3 -I tools/indep_compile.py kernels/lab/vks_probe.py 'vks::vks_kernel<0>' ... [-D NAME=VAL ...]

Kernel names are instantiated through NVRTC name expressions (template kernels need them to be emitted).
"""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import subprocess
import sys
import tempfile

NV = None
for base in sys.path:
    if os.path.isdir(os.path.join(base, "nvidia", "cuda_nvrtc")):
        NV = os.path.join(base, "nvidia")
        break


STDINT = """#pragma once
typedef signed char int8_t; typedef short int16_t; typedef int int32_t; typedef long long int64_t;
typedef unsigned char uint8_t; typedef unsigned short uint16_t; typedef unsigned int uint32_t;
typedef unsigned long long uint64_t; typedef unsigned long long uintptr_t; typedef long long intptr_t;
"""


def _lib():
    so = glob.glob(os.path.join(NV, "cuda_nvrtc", "lib", "libnvrtc.so*"))
    return ctypes.CDLL(sorted(so)[0])


def nvrtc_ptx(src: str, names: list[str], defines: list[str]) -> tuple[str, dict[str, str]]:
    L = _lib()
    prog = ctypes.c_void_p()
    rc = L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"lab.cu", 0, None, None)
    assert rc == 0, f"nvrtcCreateProgram {rc}"
    for n in names:
        assert L.nvrtcAddNameExpression(prog, n.encode()) == 0, n
    inc = os.path.join(NV, "cuda_runtime", "include")
    shim = tempfile.mkdtemp(prefix="nvrtc_shim_")
    with open(os.path.join(shim, "stdint.h"), "w") as fh:   # NVRTC ships no libc headers
        fh.write(STDINT)
    opts = [b"--gpu-architecture=compute_90a", b"-std=c++17", b"-default-device", f"-I{inc}".encode(),
            f"-I{shim}".encode(), b"-lineinfo"]
    opts += [f"-D{d}".encode() for d in defines]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = L.nvrtcCompileProgram(prog, len(opts), arr)
    sz = ctypes.c_size_t()
    L.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
    log = ctypes.create_string_buffer(sz.value)
    L.nvrtcGetProgramLog(prog, log)
    if rc != 0:
        raise RuntimeError("NVRTC failed:\n" + log.value.decode(errors="replace"))
    if log.value.strip():
        print(log.value.decode(errors="replace"))
    L.nvrtcGetPTXSize(prog, ctypes.byref(sz))
    ptx = ctypes.create_string_buffer(sz.value)
    L.nvrtcGetPTX(prog, ptx)
    low = {}
    for n in names:
        p = ctypes.c_char_p()
        L.nvrtcGetLoweredName(prog, n.encode(), ctypes.byref(p))
        low[n] = p.value.decode()
    return ptx.value.decode(), low


def ptxas(ptx: str) -> str:
    exe = os.path.join(NV, "cuda_nvcc", "bin", "ptxas")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "lab.ptx")
        with open(f, "w") as fh:
            fh.write(ptx)
        r = subprocess.run([exe, "-arch=sm_90a", "-v", "-warn-spills", "-o", os.path.join(d, "lab.cubin"), f],
                           capture_output=True, text=True)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            raise RuntimeError("ptxas failed:\n" + out)
        return out


def main(argv):
    path, rest = argv[0], argv[1:]
    names = [a for a in rest if not a.startswith("-D")]
    defines = [a[2:] for a in rest if a.startswith("-D")]
    spec = importlib.util.spec_from_file_location("labmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = mod.DEV_SRC
    ptx, low = nvrtc_ptx(src, names, defines)
    out = ptxas(ptx)
    inv = {v: k for k, v in low.items()}
    cur = None
    for line in out.splitlines():
        if "Compiling entry function" in line:
            m = line.split("'")[1] if "'" in line else line
            cur = inv.get(m, m)
            print(f"== {cur}")
        elif cur and ("Used" in line or "spill" in line or "stack" in line or "warning" in line.lower()):
            print("   " + line.strip())
        elif "warning" in line.lower() or "error" in line.lower():
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])

"""vfa_compile.py: compile-only sm_90a check of the VFA CUDA source (tools/vfa_patches.py): registers / spills / smem per kernel
instantiation and every ptxas C75xx (wgmma serialization) warning.

Run in WSL through tools/local/cudaenv.sh (nvcc 13.4 + torch headers), from the repo root:
    python kernels/lab/vfa_compile.py <work dir> [--ship]      (--ship: the shipped build without the DIAG probe variants)
Exit 1 on a compile error, any spill, or a C7515 warning.
"""
import importlib.util
import pathlib
import re
import subprocess
import sys
import sysconfig

from torch.utils import cpp_extension as ce

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    work = pathlib.Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("vfa_patches", ROOT / "tools/vfa_patches.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    src = m.cuda_source(lab="--ship" not in sys.argv)
    alt = next((a[6:] for a in sys.argv if a.startswith("--src=")), None)   # an edited copy of the source (experiments)
    if alt:
        src = pathlib.Path(alt).read_text(encoding="utf-8")
    cu = work / "vfa.cu"
    cu.write_text(src)
    inc = [f"-I{p}" for p in ce.include_paths(device_type="cuda")] + [f"-I{sysconfig.get_paths()['include']}"]
    cmd = (["nvcc", "-c", str(cu), "-o", str(work / "vfa.o"), "-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a",
            "-Xptxas=-v", "-lineinfo", "-DTORCH_EXTENSION_NAME=vfachk", "-DTORCH_API_INCLUDE_EXTENSION_H", "--compiler-options", "-fPIC"]
           + list(ce.COMMON_NVCC_FLAGS) + inc)
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    (work / "vfa.log").write_text(log)
    if p.returncode:
        print(log[-8000:])
        print("COMPILE FAILED")
        return 1
    bad = False
    name = ""
    for ln in log.splitlines():
        mm = re.search(r"Compiling entry function '(\S+)'", ln)
        if mm:
            name = mm.group(1)
            short = re.sub(r"^_ZN3vfa8vfa_mainILi(\d)ELi(\d)ELb(\d)EEEvNS_6ParamsE", r"vfa_main<ST \1, DIAG \2, O32 \3>", name)
            print(f"-- {short}")
        if "Used" in ln and "registers" in ln:
            print("   " + ln.strip())
            sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if sp and (int(sp.group(1)) or int(sp.group(2))):
                bad = True
        if "bytes stack frame" in ln:
            print("   " + ln.strip())
            sp = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if sp and (int(sp.group(1)) or int(sp.group(2))):
                bad = True
        if "C75" in ln:
            print("   WARN " + ln.strip()[:300])
            if "C7515" in ln:
                bad = True
    print("SPILL/C7515" if bad else "CLEAN")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

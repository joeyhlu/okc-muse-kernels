"""sm_90a compile-only check of the PX lab (kernels/lab/px_moe_lab.py): ptxas registers / spills / local memory per
kernel, plus a SASS count of STL / LDL and RED (the v4 reds) per function.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/px_sass.py"
"""
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "kernels/muse_moe_w4a16"))
TRITON_BIN = pathlib.Path("/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin")


def main():
    import px_moe_lab as L
    cu = L.device_source()
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "px.cu"
        p.write_text(cu)
        cub = pathlib.Path(td) / "px.cubin"
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v",
                            "-Xptxas=-warn-spills", "-o", str(cub), str(p)], capture_output=True, text=True)
        err = r.stderr
        if r.returncode != 0:
            print(err[-6000:])
            print(r.stdout[-2000:])
            sys.exit(1)
        cur = None
        for line in err.splitlines():
            m = re.search(r"Compiling entry function '(\w+)'", line)
            if m:
                cur = m.group(1)
            m = re.search(r"Used (\d+) registers", line)
            if m and cur and any(k in cur for k in ("k1_kernel36", "k2_kernel36", "k2_px", "k3_px")):
                print(f"{cur[:60]:60s} {line.strip()}")
            if "spill" in line and cur and any(k in cur for k in ("k1_kernel36", "k2_kernel36", "k2_px", "k3_px")):
                if "0 bytes spill stores, 0 bytes spill loads" not in line:
                    print(f"{cur[:60]:60s} {line.strip()}")
        s = subprocess.run([str(TRITON_BIN / "cuobjdump"), "-sass", str(cub)], capture_output=True, text=True).stdout
    for f in re.split(r"\n\s+Function : ", s)[1:]:
        name = f.split("\n", 1)[0].strip()
        if not any(k in name for k in ("k1_kernel36", "k2_kernel36", "k2_px", "k3_px")):
            continue
        lines = f.splitlines()
        cnt = lambda pat: sum(1 for l in lines if re.search(pat, l))  # noqa: E731
        nins = sum(1 for l in lines if re.match(r"\s+/\*[0-9a-f]{4}\*/", l))
        print(f"{name[:70]:70s} {nins} instr | STL {cnt(r'\bSTL\b')} LDL {cnt(r'\bLDL\b')} | RED {cnt(r'\bRED')} "
              f"HMMA {cnt(r'\bHMMA')} SHFL {cnt(r'\bSHFL')} CCTL {cnt(r'\bCCTL')} MEMBAR {cnt(r'\bMEMBAR')}")


if __name__ == "__main__":
    main()

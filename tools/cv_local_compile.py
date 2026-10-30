"""cv_local_compile.py: sm_90a compile-only gate for kernels/lab/t8/c8v2_probe.py (R-cv-1 step 0, no H200).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/cv_local_compile.py" <out_dir>

Prints ptxas regs / spills per instance and the MEMBAR / fence opcode counts per kernel. Gate: the LL instances (LL=1) carry no
MEMBAR.*.GPU / MEMBAR.SC outside the final exit block, and no spills.
"""
import collections
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab/t8"))
import c8v2_probe as C  # noqa: E402

CUOBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
INST = [(8, 4096, 1, 0), (8, 4096, 1, 1), (8, 4096, 1, 2), (7, 32768, 1, 0), (7, 32768, 1, 1), (7, 32768, 1, 2)]


def main():
    od = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cv_local")
    od.mkdir(parents=True, exist_ok=True)
    src = C.CUDA_SRC
    dev = src[:src.index("template <typename K> static void set_smem")]
    dev = re.sub(r"#include <(torch|ATen|c10)[^>]*>\n", "", dev)
    dev += "\n".join(f"template __global__ void c8f_probe<{a}, {b}, {c}, {d}>(const __grid_constant__ PF p);" for a, b, c, d in INST)
    dev += "\n}  // namespace c8v\n"
    cu = od / "c8v2.cu"
    cu.write_text(dev)
    cub = od / "c8v2.cubin"
    r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-o", str(cub), str(cu)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        sys.exit(1)
    for line in r.stderr.splitlines():
        if "Compiling entry" in line or "Used" in line or "spill" in line:
            print(line.strip()[:170])
    sass = subprocess.run([CUOBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    (od / "c8v2.sass").write_text(sass)
    fn, cnt = None, collections.defaultdict(collections.Counter)
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1)
            continue
        mm = re.match(r"\s+/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if fn and mm:
            op = mm.group(1)
            if op.startswith(("MEMBAR", "FENCE", "ERRBAR", "CCTL", "ATOMG", "RED", "STG", "LDG")):
                cnt[fn][op] += 1
    for f, c in cnt.items():
        print(f"{f[:60]}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()

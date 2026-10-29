"""vfa3_local.py: sm_90a compile-only check of a layer file's VFA kernel (VFA_CUDA_SRC device part): ptxas regs / spills per
vfa_main<ST, DIAG, true> instantiation + SASS instruction counts of the consumer hot loop are not attempted (use cuobjdump by hand).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vfa3_local.py" kernels/lab/k3_vfa3_lab1.py 0,17,18 [4]
"""
import ast
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    path = ROOT / sys.argv[1]
    diags = [int(x) for x in sys.argv[2].split(",")]
    st = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tree = ast.parse(path.read_text(encoding="utf-8"))
    src = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "VFA_CUDA_SRC")
    src = src.replace("#if VFA_LAB", "#if 1")
    if len(sys.argv) > 4:   # preg,creg override of the VFA3 setmaxnreg pair
        pr, cr = sys.argv[4].split(",")
        src = re.sub(r'#define VFA_PVT_PREG "\d+"', f'#define VFA_PVT_PREG "{pr}"', src)
        src = re.sub(r'#define VFA_PVT_CREG "\d+"', f'#define VFA_PVT_CREG "{cr}"', src)
    cut = src.index("template <int ST, int DIAG, bool O32>\nstatic void vfa_go(")
    dev = src[:cut] + "}  // namespace vfa\n"
    dev = re.sub(r"#include <(torch|ATen|c10)[^>]*>\n", "", dev)
    dev += "".join(f"template __global__ void vfa::vfa_main<{st}, {d}, true>(const __grid_constant__ vfa::Params);\n" for d in diags)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "vfa.cu"
        p.write_text(dev)
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-Xptxas=-warn-spills", "-o",
                            str(pathlib.Path(td) / "vfa.cubin"), str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-4000:])
            sys.exit(1)
        cur = None
        for line in r.stderr.splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", line)
            if m:
                dm = re.search(r"vfa_mainILi\d+ELi(\d+)E", m.group(1))
                cur = dm.group(1) if dm else m.group(1)[:40]
            if "spill" in line or "Used" in line or "warning" in line.lower():
                print(f"DIAG {cur}: {line.strip()[:160]}")
        sass = subprocess.run(["/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump", "-sass", str(pathlib.Path(td) / "vfa.cubin")], capture_output=True, text=True).stdout
        fn = None
        cnt = {}
        for line in sass.splitlines():
            m = re.search(r"Function : (\S+)", line)
            if m:
                dm = re.search(r"vfa_mainILi\d+ELi(\d+)E", m.group(1))
                fn = dm.group(1) if dm else "?"
                cnt[fn] = 0
            elif fn and re.match(r"\s+/\*[0-9a-f]{4}\*/", line):
                cnt[fn] += 1
                if any(k in line for k in ("STL", "LDL", "USETMAXREG", "EXIT", "SYNCS.ARRIVE", "BAR.SYNC", "LDSM")):
                    print(f"  {fn} #{cnt[fn]}: {line.strip()[:90]}")
        print("SASS instr per DIAG:", cnt)


if __name__ == "__main__":
    main()

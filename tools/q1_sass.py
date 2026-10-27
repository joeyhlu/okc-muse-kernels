"""sm_90a compile-only check of the Q1 lab kernel: registers, spills, local memory use and where it sits (SASS).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/q1_sass.py" [module=q1_moe_lab] [arch=sm_90a]
"""
import importlib
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
TRITON_BIN = pathlib.Path("/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin")


def main():
    modname = sys.argv[1] if len(sys.argv) > 1 else "q1_moe_lab"
    arch = sys.argv[2] if len(sys.argv) > 2 else "sm_90a"
    lab = importlib.import_module(modname)
    src = getattr(lab, "Q1_CUDA", None) or getattr(lab, "Q2_CUDA", None) or lab.Z1_CUDA
    ns = "q1" if hasattr(lab, "Q1_CUDA") else ("q2" if hasattr(lab, "Q2_CUDA") else "z1")
    stop = {"q1": "template <int NST>\nconstexpr int smem_bytes", "q2": "constexpr int smem_bytes()", "z1": "void launch("}[ns]
    head = src[:src.index(f"namespace {ns} {{")]
    body = src[src.index(f"namespace {ns} {{"):src.index(stop)]
    head = "\n".join(l for l in head.splitlines() if "torch/" not in l and "ATen" not in l and "c10/" not in l)
    inst = "".join(f"template __global__ void q1_kernel<{n}>(const Args A);\n" for n in (3, 4)) if ns == "q1" else ""
    cu = head + "\n" + body + inst + f"}}  // namespace {ns}\n"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "q1.cu"
        p.write_text(cu)
        cub = pathlib.Path(td) / "q1.cubin"
        r = subprocess.run(["nvcc", f"-arch={arch}", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v",
                            "-o", str(cub), str(p)], capture_output=True, text=True)
        print(r.stderr.strip()[-4000:])
        if r.returncode != 0:
            print(r.stdout[-4000:])
            sys.exit(1)
        s = subprocess.run([str(TRITON_BIN / "cuobjdump"), "-sass", str(cub)], capture_output=True, text=True).stdout
    funcs = re.split(r"\n\s+Function : ", s)
    for f in funcs[1:]:
        name = f.split("\n", 1)[0].strip()
        lines = f.splitlines()
        stl = [l for l in lines if re.search(r"\bSTL\b", l)]
        ldl = [l for l in lines if re.search(r"\bLDL\b", l)]
        nins = sum(1 for l in lines if re.match(r"\s+/\*[0-9a-f]{4}\*/", l))
        print(f"{name}: {nins} instr, STL {len(stl)}, LDL {len(ldl)}")
        for l in (stl + ldl)[:12]:
            print("   ", l.strip()[:120])


if __name__ == "__main__":
    main()

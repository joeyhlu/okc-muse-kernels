"""sm_90a compile-only check of the SK lab arms (kernels/lab/sk_lab.py): ptxas registers / spills per MoE kernel.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/sk_sass.py" [arm ...]
"""
import concurrent.futures as cf
import os
import pathlib
import re
import subprocess
import sysconfig
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
KEYS = ("k1_kernel", "k2_kernel", "p0_kernel")


def one(arm, td):
    import sk_lab as L
    from torch.utils.cpp_extension import include_paths
    cu, _ = (L.source_all if os.environ.get("SK_ALLVIDS") else L.source)(arm)
    p = pathlib.Path(td) / f"sk_{arm}.cu"
    p.write_text(cu)
    inc = sum((["-I", d] for d in include_paths("cuda")), [])
    r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v",
                        "-Xptxas=-warn-spills", "-std=c++20", *inc, "-I", sysconfig.get_paths()["include"], "-o", str(p.with_suffix(".cubin")), str(p)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return arm, None, r.stderr[-5000:]
    out, cur = [], None
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            cur = m.group(1)
        m = re.search(r"Used (\d+) registers", line)
        if m and cur and any(k in cur for k in KEYS):
            out.append(f"{cur[:70]:70s} {line.strip()[:120]}")
        if cur and any(k in cur for k in KEYS) and "spill" in line:
            out.append(f"{'':70s} {line.strip()}")
    return arm, out, ""


def main():
    import sk_lab as L
    arms = sys.argv[1:] or list(L.ARMS)
    with tempfile.TemporaryDirectory() as td, cf.ThreadPoolExecutor(len(arms)) as ex:
        for arm, out, err in ex.map(lambda a: one(a, td), arms):
            print(f"==== {arm}")
            print(err if out is None else "\n".join(out))


if __name__ == "__main__":
    main()

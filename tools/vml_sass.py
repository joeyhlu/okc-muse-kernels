"""sm_90a compile-only check of the VML lab kernels (kernels/lab/vml_*.py): ptxas registers / spills / wgmma
serialization warnings per kernel, optional SASS opcode counts.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vml_sass.py" probe [--sass]
"""
import collections
import importlib
import pathlib
import re
import subprocess
import sys
import sysconfig
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "probe"
    L = importlib.import_module(f"vml_{which}")
    from torch.utils.cpp_extension import include_paths
    src = L.VP_CUDA if hasattr(L, "VP_CUDA") else L.CUDA
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / f"vml_{which}.cu"
        p.write_text(src)
        inc = sum((["-I", d] for d in include_paths("cuda")), [])
        cub = p.with_suffix(".cubin")
        r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v",
                            "-Xptxas=-warn-spills", "-std=c++20", *inc, "-I", sysconfig.get_paths()["include"], "-o",
                            str(cub), str(p)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-6000:])
            sys.exit(1)
        cur = None
        for line in r.stderr.splitlines():
            m = re.search(r"Compiling entry function '(\w+)'", line)
            if m:
                cur = m.group(1)
            if "warning" in line.lower() or "Used" in line or "spill" in line:
                print(f"{(cur or '')[:60]:60s} {line.strip()[:200]}")
        if "--sass" in sys.argv:
            s = subprocess.run(["/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump", "-sass", str(cub)], capture_output=True, text=True).stdout
            fn, cnt = None, collections.defaultdict(collections.Counter)
            for line in s.splitlines():
                m = re.search(r"Function : (\S+)", line)
                if m:
                    fn = m.group(1)
                m = re.search(r"/\*[0-9a-f]{4,}\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", line)
                if m and fn:
                    cnt[fn][m.group(2).split(".")[0]] += 1
            for fn, c in cnt.items():
                print(fn[:80], dict(c.most_common(30)))


if __name__ == "__main__":
    main()

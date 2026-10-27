"""sm_90a compile-only check of a generated layer's K2A33 (all gate vids) and K1 CUDA sources: ptxas registers / spills per
kernel, side by side for two layer files (e.g. k3_m32_2 vs k3_m32_3).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/sk1_sass.py" kernels/muse_decode_layer/k3_m32_2.py kernels/muse_decode_layer/k3_m32_3.py
"""
import ast
import concurrent.futures as cf
import pathlib
import re
import subprocess
import sys
import sysconfig
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sources(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") in ("K2A33_KERNEL_SRC", "K1_CUDA_SRC") for t in node.targets):
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name in ("_k2a33_wg_specs", "_k2a36_gate_cases", "_k2a36_gate_attrs"):
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    k33 = (ns["K2A33_KERNEL_SRC"].replace("@CASES3@", ns["_k2a36_gate_cases"]()).replace("@ATTRS3@", ns["_k2a36_gate_attrs"]())
           .replace("@WGSPECS@", ns["_k2a33_wg_specs"]()))
    return {"k2a33": k33, "k1": ns["K1_CUDA_SRC"]}


def compile_one(tag, src, td):
    from torch.utils.cpp_extension import include_paths
    p = pathlib.Path(td) / f"{tag}.cu"
    p.write_text(src)
    inc = sum((["-I", d] for d in include_paths("cuda")), []) + ["-I", sysconfig.get_paths()["include"]]
    r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-std=c++20",
                        *inc, "-o", str(p.with_suffix(".cubin")), str(p)], capture_output=True, text=True)
    if r.returncode != 0:
        return tag, None, r.stderr[-4000:]
    out, cur, spill = {}, None, ""
    for line in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '(\w+)'", line)
        if m:
            cur = m.group(1)
        if "bytes stack frame" in line:
            spill = re.sub(r"\s+", " ", line.strip().replace("ptxas info    : ", ""))
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            out[cur] = f"{m.group(1)} regs | {spill}"
    return tag, out, ""


def main():
    files = sys.argv[1:]
    with tempfile.TemporaryDirectory() as td, cf.ThreadPoolExecutor(4) as ex:
        jobs = []
        for i, f in enumerate(files):
            for k, s in sources(ROOT / f).items():
                jobs.append(ex.submit(compile_one, f"{k}_{i}", s, td))
        res = {}
        for j in jobs:
            tag, out, err = j.result()
            if out is None:
                print(f"==== {tag} FAILED\n{err}")
                sys.exit(1)
            res[tag] = out
    for k in ("k2a33", "k1"):
        a, b = res[f"{k}_0"], res.get(f"{k}_1", {})
        print(f"==== {k}: {files[0]} vs {files[1] if len(files) > 1 else '-'}")
        for name in sorted(set(a) | set(b)):
            same = "same" if a.get(name) == b.get(name) else "DIFF"
            print(f"  {same:4s} {name[:64]:64s} | {a.get(name, '-')}  ||  {b.get(name, '-')}")


if __name__ == "__main__":
    main()

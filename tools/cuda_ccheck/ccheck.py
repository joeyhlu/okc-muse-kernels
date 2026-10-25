#!/usr/bin/env python3
"""ccheck.py -- laptop compile check for a kernel file's CUDA source, no GPU and no torch: nvcc 13.0 from NVIDIA's pip
wheels + stub torch headers (tools/cuda_ccheck/torch_stub). Prints ptxas -v registers / stack / spills per kernel, and
optionally dumps SASS and compares it with another file's (constant-bank offsets normalized).

  python3 tools/cuda_ccheck/ccheck.py kernels/muse_attention_decode/k1_v9c.py [--src CUDA_SRC] [--arch sm_90|sm_90a]
                                      [-DK1_CLUSTER=0 ...] [--sass] [--same-sass-as kernels/.../k1_v4.py]

One-time toolchain (~150 MB, Linux or WSL; pip picks the platform wheels), into $OKC_CUDA (default ~/.cache/okc_cuda13):
  mkdir -p ~/.cache/okc_cuda13 && cd ~/.cache/okc_cuda13
  pip download --no-deps "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-runtime==13.0.*" "nvidia-cuda-cccl==13.0.*" \\
      "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-cuobjdump==13.2.51" nvidia-cuda-nvdisasm
  for w in *.whl; do python3 -m zipfile -e "$w" .; done; chmod +x nvidia/cu13/bin/* nvidia/cu13/nvvm/bin/*
(13.0.88 = the judge image's nvcc; cuobjdump / nvdisasm have no 13.0 wheel, the newer ones read 13.0 cubins.)
Not a substitute for tools/box/sass_gate.py on the pod (real torch headers, the judge's exact flags), but it catches
compile errors and register / spill regressions before any GPU time. Flags default to k1's own build
(-O3 -lineinfo -Xptxas=-v, compute_90/sm_90) plus torch's default -D__CUDA_NO_HALF* set.
"""
import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def cuda_root():
    base = os.environ.get("OKC_CUDA", os.path.expanduser("~/.cache/okc_cuda13"))
    root = os.path.join(base, "nvidia", "cu13")
    if not os.path.isfile(os.path.join(root, "bin", "nvcc")):
        sys.exit(f"ccheck: no nvcc under {root}; see the one-time toolchain steps in {__file__}")
    return root


def cuda_src(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    env = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            env[n.targets[0].id] = n.value.value
    if not names:
        names = [k for k in env if re.search(r"(^|_)SRC$", k) and "CPP" not in k.split("_")]
        if len(names) != 1:
            sys.exit(f"ccheck: pick the CUDA source with --src (candidates: {names})")
    return "\n".join(env[n] for n in names)


def compile_one(path, names, arch, defines, out_dir):
    root = cuda_root()
    tag = os.path.basename(path).replace(".py", "")
    cu = os.path.join(out_dir, tag + ".cu")
    with open(cu, "w") as f:
        f.write("#include <cuda.h>\n#include <cuda_runtime.h>\n" + cuda_src(path, names))
    cubin = os.path.join(out_dir, tag + ".cubin")
    env = dict(os.environ, PATH=os.pathsep.join([os.path.join(root, "bin"), os.path.join(root, "nvvm", "bin"),
                                                 os.environ.get("PATH", "")]))
    cmd = [os.path.join(root, "bin", "nvcc"), "-cubin", "-O3", "-lineinfo", "-Xptxas=-v",
           f"-gencode=arch=compute_{arch[3:]},code={arch}", "-std=c++17", "--expt-relaxed-constexpr",
           "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
           "-D__CUDA_NO_HALF2_OPERATORS__", "-I", os.path.join(HERE, "torch_stub"), "-I", os.path.join(root, "include"),
           "-I", os.path.join(root, "include", "cccl"), *defines, "-o", cubin, cu]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    if r.returncode:
        print(log[-8000:])
        sys.exit(f"ccheck: nvcc failed for {path}")
    return cubin, log, env


def report(log):
    rows, cur, frame = [], None, None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur, frame = m.group(1), None
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur:
            frame = tuple(int(x) for x in m.groups())
        m = re.search(r"Used (\d+) registers", line)
        if m and cur:
            try:
                nm = subprocess.run(["c++filt", cur], capture_output=True, text=True).stdout.strip() or cur
            except OSError:
                nm = cur
            rows.append((nm, int(m.group(1)), frame or (0, 0, 0)))
        if re.search(r"\b(warning|advisory)\b", line, re.I):
            print("  ptxas:", line.strip())
    for nm, regs, (fr, st, ld) in rows:
        print(f"  {nm[:64]:64s} {regs:4d} regs  stack {fr:4d} B  spill st/ld {st}/{ld} B")
    return rows


def sass(cubin, env):
    r = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("ccheck: cuobjdump failed (install the cuobjdump + nvdisasm wheels): " + r.stderr[-400:])
    txt = re.sub(r"/\*[0-9a-f]{4,}\*/|/\* 0x[0-9a-f]+ \*/", "", r.stdout)
    return "\n".join(l.strip() for l in txt.splitlines() if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--src", nargs="*", default=[])
    ap.add_argument("--arch", default="sm_90")
    ap.add_argument("--sass", action="store_true", help="write <out>/<name>.sass")
    ap.add_argument("--same-sass-as", default="", help="compare SASS with this file's (constant offsets normalized)")
    ap.add_argument("--out", default="")
    a, rest = ap.parse_known_args()
    defines = [x for x in rest if x.startswith("-D")]
    out = a.out or tempfile.mkdtemp(prefix="ccheck_")
    os.makedirs(out, exist_ok=True)
    print(f"ccheck {a.kernel} {a.arch} {' '.join(defines)}")
    cubin, log, env = compile_one(a.kernel, a.src, a.arch, defines, out)
    report(log)
    if a.sass or a.same_sass_as:
        s1 = sass(cubin, env)
        if a.sass:
            open(cubin.replace(".cubin", ".sass"), "w").write(s1)
            print(f"  SASS: {cubin.replace('.cubin', '.sass')}")
        if a.same_sass_as:
            out2 = tempfile.mkdtemp(prefix="ccheck_ref_")
            cubin2, log2, _ = compile_one(a.same_sass_as, a.src, a.arch, [], out2)
            s2 = sass(cubin2, env)
            norm = lambda s: re.sub(r"c\[0x0\]\[0x[0-9a-f]+\]", "c[0x0][*]", s)   # noqa: E731
            same = norm(s1) == norm(s2)
            print(f"  SASS {'IDENTICAL' if same else 'DIFFERS'} to {a.same_sass_as} (constant-bank offsets normalized)")
            if not same:
                sys.exit(1)


if __name__ == "__main__":
    main()

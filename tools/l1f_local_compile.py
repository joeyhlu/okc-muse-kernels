"""l1f_local_compile.py: sm_90a compile-only check of the RE GD block (WSL nvcc 13.4, no H200).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/l1f_local_compile.py" kernels/lab/k3_l1f_0.py [kernels/muse_decode_layer/k3_gc_19.py]

Builds GD_CUDA_SRC of the lab file (and of the base) with the pooler (1,8,16) and GC (2,8,18) instances, prints ptxas
registers / spills per gd_kernel instance, and checks that every RE = 0 instance has the same PTX as the base's (same ptxas -> same SASS).
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
INST = [(0, 1, 1, 8, 16, 2, 1, 2, 1), (0, 2, 1, 8, 18, 2, 1, 2, 0, 2), (0, 4, 1, 8, 18, 1, 1, 2, 0, 2)]


def gd_src(path):
    s = (ROOT / path).read_text(encoding="utf-8")
    for n in ast.parse(s).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "GD_CUDA_SRC":
            c = ast.literal_eval(n.value)
            inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(INST))
            info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(INST))
            return c.replace("@INST@", inst).replace("@INFO@", info)
    raise SystemExit("GD_CUDA_SRC not found")


def build(src, tag, td):
    cu = pathlib.Path(td) / f"{tag}.cu"
    cu.write_text(src, encoding="utf-8")
    cub = pathlib.Path(td) / f"{tag}.cubin"
    r = subprocess.run(["nvcc", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-cubin", "-Xptxas=-v", "-o", str(cub), str(cu)],
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-6000:])
        raise SystemExit(f"{tag}: nvcc failed")
    regs = {}
    fn = None
    for ln in r.stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", ln)
        if m:
            fn = m.group(1)
        m = re.search(r"Used (\d+) registers", ln)
        if m and fn:
            sp = re.search(r"(\d+) bytes spill stores", r.stderr[r.stderr.find(fn):])
            regs[fn] = (int(m.group(1)), ln.strip())
    ptx = pathlib.Path(td) / f"{tag}.ptx"
    r2 = subprocess.run(["nvcc", "-O3", "-gencode=arch=compute_90a,code=sm_90a", "-ptx", "-o", str(ptx), str(cu)],
                        capture_output=True, text=True)
    if r2.returncode:
        raise SystemExit(f"{tag}: nvcc -ptx failed {r2.stderr[-2000:]}")
    funcs, cur = {}, None
    for ln in ptx.read_text().splitlines():
        m = re.match(r"\.visible \.entry (\S+)\(", ln) or re.match(r"\.entry (\S+)\(", ln)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur is not None:
            ln2 = re.sub(r"_param_0\[\d+\]", "_param_0[N]", ln.replace(cur, "F"))
            ln2 = re.sub(r"\$L__BB\d+_", "$L__BB_", ln2)
            if ln2.strip().startswith("// .globl"):
                continue
            ln2 = re.sub(r"_ZZN2gd9gd_kernelI\S+?EEEvNS_4ArgsEE", "LOCAL", ln2)
            funcs[cur].append(ln2)
    return regs, funcs, r.stderr


def demangle(names):
    r = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True)
    return dict(zip(names, r.stdout.splitlines()))


def key(dm):
    """gd_kernel<...> template args as a tuple, RE default 0 appended / dropped."""
    m = re.search(r"gd_kernel<(.*)>\(", dm)
    if not m:
        return None
    a = tuple(int(x.strip()) for x in m.group(1).split(","))
    a = a + (0,) * (12 - len(a))
    return a


def main():
    lab = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "kernels/muse_decode_layer/k3_gc_19.py"
    with tempfile.TemporaryDirectory() as td:
        r1, f1, e1 = build(gd_src(lab), "lab", td)
        r0, f0, e0 = build(gd_src(base), "base", td)
    d1, d0 = demangle(list(f1)), demangle(list(f0))
    k1 = {key(d1[n]): n for n in f1 if key(d1[n])}
    k0 = {key(d0[n]): n for n in f0 if key(d0[n])}
    for k, n in sorted(k1.items()):
        rr = r1.get(n, (None, "?"))
        print(f"[lab]  {k}: {rr[1]}")
    for k, n in sorted(k0.items()):
        rr = r0.get(n, (None, "?"))
        print(f"[base] {k}: {rr[1]}")
    same = True
    for k, n0 in sorted(k0.items()):
        n1 = k1.get(k)
        if n1 is None:
            print(f"[cmp] {k}: missing in lab")
            same = False
            continue
        a, b = f0[n0], f1[n1]
        h0, h1 = hashlib.sha1("\n".join(a).encode()).hexdigest()[:10], hashlib.sha1("\n".join(b).encode()).hexdigest()[:10]
        if a != b and "--diff" in sys.argv:
            import difflib
            print(chr(10).join(list(difflib.unified_diff(a, b, lineterm="", n=1))[:60]))
        print(f"[cmp] {k}: base {len(a)} ptx lines {h0}, lab {len(b)} ptx lines {h1} -> {'SAME' if a == b else 'DIFF'}")
        same &= a == b
    sp = [ln for ln in e1.splitlines() if "spill" in ln and not re.search(r"\b0 bytes spill stores, 0 bytes spill loads", ln)]
    print(f"[re_local_compile] RE = 0 instances PTX-identical to base: {same}; non-zero spill lines in lab: {len(sp)}")
    for ln in sp:
        print("   ", ln.strip())
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

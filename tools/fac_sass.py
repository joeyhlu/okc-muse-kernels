"""fac_sass.py: sm_90a SASS of a layer file's VA1 (va1_main<4>) and VFA (vfa_main<4, DIAG, true>) kernels, for the
FAC consumer cycle budget. Writes the full SASS to <out_dir>/{va,vfa}.sass and prints regs / spills plus an opcode
histogram of each consumer hot loop (the backward-branch loop that contains the most HMMA / HGMMA). <= 20 rows per kernel.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/fac_sass.py" kernels/muse_decode_layer/k3_gc_21.py <out_dir> [va|vfa|both]
"""
import ast
import collections
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUOBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"


def src_of(tree, name):
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == name)


def strip(dev):
    return re.sub(r"#include <(torch|ATen|c10)[^>]*>\n", "", dev)


def build(name, dev, od):
    cu = od / f"{name}.cu"
    cu.write_text(dev)
    cub = od / f"{name}.cubin"
    r = subprocess.run(["nvcc", "-arch=sm_90a", "-cubin", "-O3", "--expt-relaxed-constexpr", "-Xptxas=-v", "-o", str(cub), str(cu)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        sys.exit(1)
    for line in r.stderr.splitlines():
        if ("Used" in line or "spill" in line) and ("bytes stack" not in line or "spill" in line):
            if "0 bytes spill stores, 0 bytes spill loads" in line:
                continue
            print(f"[{name}] {line.strip()[:150]}")
    sass = subprocess.run([CUOBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    (od / f"{name}.sass").write_text(sass)
    return sass


def loops(sass, fn_pat):
    ins, on = [], False
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            on = re.search(fn_pat, m.group(1)) is not None
            continue
        if on:
            mm = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
            if mm:
                ins.append((int(mm.group(1), 16), mm.group(2).strip()))
    return ins


def op_of(text):
    t = re.sub(r"^@!?U?P\w+\s+", "", text)
    return t.split()[0] if t else "?"


def report(name, ins, key):
    addr = {a: i for i, (a, _) in enumerate(ins)}
    cands = []
    for i, (a, t) in enumerate(ins):
        m = re.search(r"BRA\s+(?:`\(\.L_x_\d+\)\s*)?.*?0x([0-9a-f]+)", t)
        if op_of(t) == "BRA" and m:
            tgt = int(m.group(1), 16)
            if tgt < a and tgt in addr:
                j = addr[tgt]
                body = ins[j:i + 1]
                nk = sum(1 for _, x in body if key in op_of(x))
                cands.append((nk, -(i - j), j, i))
    need = max((c[0] for c in cands), default=0)
    need = 32 if key == "HMMA" and need >= 32 else need
    cands = sorted([c for c in cands if c[0] >= need], key=lambda c: c[3] - c[2]) + sorted([c for c in cands if c[0] < need], reverse=True)
    if not cands:
        print(f"[{name}] no loop found")
        return
    nk, _, j, i = cands[0]
    body = ins[j:i + 1]
    h = collections.Counter(op_of(t).split(".")[0] for _, t in body)
    print(f"[{name}] hot loop SASS [{j}..{i}] = {len(body)} instr ({key}: {nk}); total kernel {len(ins)}")
    print(f"[{name}] " + ", ".join(f"{k} {v}" for k, v in h.most_common(24)))
    inner = [c for c in cands[1:6] if c[2] >= j and c[3] <= i]
    for c in inner:
        print(f"[{name}]   inner loop [{c[2]}..{c[3]}] len {c[3] - c[2] + 1}: " + " | ".join(op_of(t) for _, t in ins[c[2]:c[3] + 1])[:150])


def main():
    path = ROOT / sys.argv[1]
    od = pathlib.Path(sys.argv[2])
    od.mkdir(parents=True, exist_ok=True)
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if which in ("va", "both"):
        s = src_of(tree, "K1_CUDA_SRC")
        dev = strip(s[: s.index("}  // namespace k1") + len("}  // namespace k1")]) + "\n"
        if "template <int ST, int CV = 0>" in s:   # R-fac lab files: report every consumer variant
            dev += "".join(f"template __global__ void k1::va1_main<4, {c}>(const k1::Params);\n" for c in (0, 1, 2))
            sass = build("va", dev, od)
            for c in (0, 1, 2):
                report(f"va cv{c}", loops(sass, rf"va1_mainILi4ELi{c}E"), "HMMA")
        else:
            dev += "template __global__ void k1::va1_main<4>(const k1::Params);\n"
            sass = build("va", dev, od)
            report("va", loops(sass, r"va1_main"), "HMMA")
    if which in ("vfa", "both"):
        s = src_of(tree, "VFA_CUDA_SRC").replace("#if VFA_LAB", "#if 1")
        cut = s.index("template <int ST, int DIAG, bool O32>\nstatic void vfa_go(")
        dev = strip(s[:cut]) + "}  // namespace vfa\n"
        dev += "template __global__ void vfa::vfa_main<4, 0, true>(const __grid_constant__ vfa::Params);\n"
        sass = build("vfa", dev, od)
        report("vfa", loops(sass, r"vfa_main"), "GMMA")


if __name__ == "__main__":
    main()

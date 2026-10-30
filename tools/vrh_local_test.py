"""Local sm_90a gate for VM_RH (tools/vrh_patches.py): w72 / bf2 vm_main cubins of the base layer's VML vs the patched source at
VM_RH 0 (SASS must be identical) and w72 at VM_RH 1 (regs / spills / MEMBAR + RED counts). WSL nvcc 13.4 via tools/local/cudaenv.sh.
    python tools/vrh_local_test.py [layer=kernels/lab/k3_b23_1.py]
"""
import ast
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import vrh_patches as P  # noqa: E402

OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
INCS = ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>")


def vml(text):
    ns = {}
    NLc = chr(10)
    a = text.index("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096" + NLc)
    b = text.index(NLc + "VML_ENABLE = True" + NLc, a)
    exec(compile("import hashlib, time" + NLc + "import torch" + NLc + text[a:b], "vml_block", "exec"), ns)   # noqa: S102
    i = text.index(NLc + "_VML_DEFS = {")
    defs = ast.literal_eval(text[i + len(NLc + "_VML_DEFS = "):text.index("}" + NLc, i) + 1])
    cu = ns["_vml_CUDA"]
    cu = cu[:cu.index("void vml_run(")]
    for inc in INCS:
        cu = cu.replace(inc, "")
    return cu, ns["_vml_FLAGS"], defs


def compile_(src, flags, td, nm):
    k = pathlib.Path(td) / f"{nm}.cu"
    k.write_text(src, encoding="utf-8")
    cub = pathlib.Path(td) / f"{nm}.cubin"
    fl = [f for f in flags if not f.startswith("-gencode")]
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(k)], capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode:
        return None, log[-3000:]
    sass = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    body = "\n".join(ln for ln in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln))
    info = [ln.strip() for ln in log.splitlines() if "vm_main" in ln or "Used" in ln or "spill" in ln]
    return body, info


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    t0 = (ROOT / opt.get("layer", "kernels/lab/k3_b23_1.py")).read_text(encoding="utf-8")
    t1 = P.apply(t0)
    cu0, fl0, defs = vml(t0)
    cu1, _, _ = vml(t1)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for b in ("w72", "bf2"):
            s0, i0 = compile_(cu0, fl0 + list(defs[b]), td, f"{b}0")
            s1, i1 = compile_(cu1, fl0 + list(defs[b]), td, f"{b}1")
            same = s0 is not None and s0 == s1
            ok &= same
            print(f"[vrh] {b} base vs patched VM_RH 0: SASS {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
        s2, i2 = compile_(cu1, fl0 + list(defs["w72"]) + ["-DVM_RH=1"], td, "w72rh")
        if s2 is None:
            print("[vrh] w72 VM_RH 1 FAILED:\n" + i2)
            return 1
        main_info = [x for x in i2 if "Used" in x or "spill" in x]
        print("[vrh] w72 VM_RH 1 ptxas: " + " | ".join(main_info[-6:]), flush=True)
        for nm, s in (("base", s0 if False else compile_(cu0, fl0 + list(defs["w72"]), td, "w72b")[0]), ("rh", s2)):
            print(f"[vrh]   {nm}: MEMBAR {len(re.findall('MEMBAR', s))} RED {len(re.findall(r'RED\.', s))} "
                  f"LDS.STRONG/ACQ {len(re.findall(r'LDS.*STRONG|LD\.E.*STRONG', s))} instr {s.count(chr(10)) + 1}", flush=True)
    print(f"[vrh] {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

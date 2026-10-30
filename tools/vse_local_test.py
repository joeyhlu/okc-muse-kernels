"""Local sm_90a gate for tools/vse_patches.py (R-vse): vm_main cubins of a file's VML (layer or MoE-board file) vs the patched
source. Gates: VSE_HOLD 0 SASS identical (bf2, w72; also the VM_MKW 64 marked builds compile); VSE_HOLD 1 / 3 regs <= base, 0 new
spills; MEMBAR / SYNCS counts. Run in WSL (tools/local/cudaenv.sh, nvcc 13.4):
    python tools/vse_local_test.py [file=kernels/lab/k3_rider3_1.py] [holds=1,3]
"""
import pathlib
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import vse_patches as P  # noqa: E402

OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
INCS = ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>")
DQ = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")
TAIL = ("-DVM_TA=1", "-DVM_DEFER=1", "-DVCP_DPI=0", "-DVCP_DPH=0", "-DVCP_PK=1", "-DSFX=1")
DEFS = {"bf2": DQ + ("-DVM_RT=1", "-DVM_FT=1", "-DVM_FTR=1") + TAIL,
        "w72": DQ + ("-DVM_WG5=2", "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72", "-DVM_RT=1", "-DVM_FTR=1",
                     "-DVM_LD=2") + TAIL}


def vml(text):
    nl = chr(10)
    a = text.index("_vml_E_LOC, _vml_TOPK, _vml_KDIM = 32, 8, 4096" + nl)
    f = text.index(nl + "_vml_FLAGS = [", a)
    b = text.index(nl, f + 1)
    ns = {}
    exec(compile("import hashlib, time" + nl + "import torch" + nl + text[a:b], "vml_block", "exec"), ns)   # noqa: S102
    cu = ns["_vml_CUDA"]
    cu = cu[:cu.index("void vml_run(")]
    for inc in INCS:
        cu = cu.replace(inc, "")
    return cu, ns["_vml_FLAGS"]


def compile_(src, flags, td, nm):
    k = pathlib.Path(td) / f"{nm}.cu"
    k.write_text(src, encoding="utf-8")
    cub = pathlib.Path(td) / f"{nm}.cubin"
    fl = [f for f in flags if not f.startswith("-gencode")]
    r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(k)], capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode:
        return None, log[-3000:]
    sass = subprocess.run([OBJ, "-sass", "-fun", "vm_main", str(cub)], capture_output=True, text=True).stdout
    if "vm_main" not in sass:
        sass = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
    body = "\n".join(ln for ln in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln))
    lines = log.splitlines()
    info = ""
    for i, ln in enumerate(lines):   # ptxas -v: "Compiling entry function '..vm_main..'" then "Used N registers" / spill lines
        if "Compiling entry function" in ln and "vm_main" in ln:
            info = " | ".join(x.strip() for x in lines[i + 1:i + 5] if "spill" in x or "Used" in x)
    return body, info


def stats(s):
    return (f"instr {s.count(chr(10)) + 1} MEMBAR {len(re.findall('MEMBAR', s))} SYNCS.ARRIVE {len(re.findall(r'SYNCS\.ARRIVE', s))} "
            f"BAR.SYNC {len(re.findall(r'BAR\.SYNC', s))}")


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    t0 = (ROOT / opt.get("file", "kernels/lab/k3_rider3_1.py")).read_text(encoding="utf-8")
    holds = [int(h) for h in opt.get("holds", "1,3").split(",") if h]
    cu0, fl0 = vml(t0)
    cu1, _ = vml(P.apply(t0))
    jobs = {}
    for b in ("bf2", "w72"):
        jobs[f"{b}0"] = (cu0, fl0 + list(DEFS[b]))
        jobs[f"{b}1"] = (cu1, fl0 + list(DEFS[b]))
        jobs[f"{b}m"] = (cu1, fl0 + list(DEFS[b]) + ["-DVM_MKW=64"])
        for h in holds:
            jobs[f"{b}h{h}"] = (cu1, fl0 + list(DEFS[b]) + [f"-DVSE_HOLD={h}"])
            jobs[f"{b}h{h}m"] = (cu1, fl0 + list(DEFS[b]) + [f"-DVSE_HOLD={h}", "-DVM_MKW=64"])
        if opt.get("ps"):
            jobs[f"{b}ps"] = (cu1, fl0 + list(DEFS[b]) + ["-DVSE_PS=1"])
            jobs[f"{b}psm"] = (cu1, fl0 + list(DEFS[b]) + ["-DVSE_PS=1", "-DVM_MKW=64"])
    ok = True
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(8) as ex:
        res = dict(zip(jobs, ex.map(lambda kv: compile_(kv[1][0], kv[1][1], td, kv[0]), jobs.items())))
        for b in ("bf2", "w72"):
            s0, i0 = res[f"{b}0"]
            s1, _ = res[f"{b}1"]
            same = s0 is not None and s0 == s1
            ok &= same
            print(f"[vse] {b} base vs patched VSE_HOLD 0: SASS {'IDENTICAL' if same else 'DIFFERS'}; base {i0}; {stats(s0 or '')}", flush=True)
            for nm in [f"{b}m"] + [f"{b}h{h}" for h in holds] + [f"{b}h{h}m" for h in holds] + ([f"{b}ps", f"{b}psm"] if opt.get("ps") else []):
                s, i = res[nm]
                if s is None:
                    ok = False
                    print(f"[vse] {nm} FAILED:\n{i}", flush=True)
                    continue
                print(f"[vse]   {nm}: {i}; {stats(s)}", flush=True)
    print(f"[vse] {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

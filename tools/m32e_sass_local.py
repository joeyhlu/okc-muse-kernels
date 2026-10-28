"""m32e_sass_local.py: device-only sm_90a cubin of kernels/lab/m32e_moe.py variants in WSL + SASS opcode counts (no torch build,
no GPU): cuts the host wrapper (vml_run / vml_cfg) and the torch includes, runs nvcc -cubin -Xptxas=-v, then cuobjdump -sass.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/m32e_sass_local.py" OUTDIR variant[,variant...] [--dump]

Prints per variant: ptxas registers / spills of vm_main and the opcode counts (R2UR, IMAD.MOV, UIADD3, UMOV, VIADD, HGMMA,
STL, LDL, total) of vm_main; --dump writes OUTDIR/<variant>.sass (full) for HGMMA-neighbourhood reading.
"""
import collections
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))

BASE = ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")   # k3_gc_4's VML variant 'bf2'
VARS = {"bf2": BASE, "ud": BASE + ("-DVM_UD=1",), "u2": BASE + ("-DVM_UD2=1",), "u3": BASE + ("-DVM_UD2=2",),
        "u4": BASE + ("-DVM_UD2=3",), "u5": BASE + ("-DVM_UD2=4",),
        "u4l": BASE + ("-DVM_UD2=3", "-DVM_UD=1"), "u5l": BASE + ("-DVM_UD2=4", "-DVM_UD=1")}
W5 = lambda m, p, c, g: BASE + (f"-DVM_WG5={m}", f"-DVM_PREG={p}", f"-DVM_CREG={c}", f"-DVM_GV={g}")   # noqa: E731
VARS.update({"w1g8": W5(1, 56, 112, 8), "w1": W5(1, 32, 112, 2), "w2": W5(2, 32, 112, 2), "w2p24": W5(2, 24, 112, 2),
             "w2g8": W5(2, 32, 112, 8), "w3": W5(3, 32, 112, 2), "w2c120": W5(2, 24, 120, 2),
             "w2j72": W5(2, 32, 112, 2) + ("-DVM_JN=72", "-DVM_XSN=72"),
             "w2j80": W5(2, 32, 112, 2) + ("-DVM_JN=80", "-DVM_NX=2", "-DVM_XSN=80"), "w2g1": W5(2, 32, 112, 1),
             "w2j72g1": W5(2, 32, 112, 1) + ("-DVM_JN=72", "-DVM_XSN=72"),
             "w2j96": W5(2, 32, 112, 2) + ("-DVM_JN=96", "-DVM_NX=2", "-DVM_XSN=96"),
             "w2j112": W5(2, 32, 112, 2) + ("-DVM_JN=112", "-DVM_NX=2", "-DVM_XSN=112")})
KEYS = ["R2UR", "IMAD.MOV", "UIADD3", "UMOV", "VIADD", "HGMMA", "REDUX", "SHFL", "STL", "LDL"]
OBJ = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"


def device_src():
    import m32e_moe as V
    src = V.CUDA
    cut = src.index("void vml_run(")
    src = src[:cut]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src, [f for f in V.FLAGS if not f.startswith("-gencode")]


def counts(sass):
    cur, per = None, collections.defaultdict(collections.Counter)
    for ln in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", ln)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_.]+)", ln)
        if cur and m:
            op = m.group(1)
            per[cur]["total"] += 1
            for k in KEYS:
                if op == k or op.startswith(k + "."):
                    per[cur][k] += 1
    return per


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2].split(",")
    dump = "--dump" in sys.argv
    src, flags = device_src()
    cu = out / "vml_dev.cu"
    cu.write_text(src)
    res = {}

    def job(nm):
        cub = out / f"{nm}.cubin"
        cmd = ["nvcc", "-cubin", "-arch=sm_90a", *flags, *VARS[nm], "-Xptxas=-v", "-o", str(cub), str(cu)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        info = [ln for ln in (r.stdout + r.stderr).splitlines() if ("vm_main" in ln or "Used" in ln or "spill" in ln
                                                                    or "error" in ln or "C75" in ln)]
        if r.returncode:
            res[nm] = ("FAIL", "\n".join((r.stdout + r.stderr).splitlines()[-30:]), None)
            return
        s = subprocess.run([OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        if dump:
            (out / f"{nm}.sass").write_text(s)
        res[nm] = ("ok", "\n".join(info), counts(s))

    ths = [threading.Thread(target=job, args=(nm,)) for nm in names]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for nm in names:
        st, info, per = res[nm]
        print(f"[m32e-sass] {nm}: {st}")
        if st != "ok":
            print(info)
            continue
        main_lines = [ln for ln in info.splitlines() if "Used" in ln or "spill" in ln or "C75" in ln]
        print("\n".join("   " + ln.strip()[:160] for ln in main_lines[-6:]))
        for fn, c in per.items():
            if "vm_main" in fn:
                print(f"   {fn[:40]} total {c['total']} " + " ".join(f"{k} {c[k]}" for k in KEYS), flush=True)


if __name__ == "__main__":
    main()

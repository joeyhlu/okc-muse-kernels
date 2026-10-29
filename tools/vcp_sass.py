"""vcp_sass.py: ptxas regs / SASS opcode counts of vm_main for tools/box/vcp_run.py arms (k3_gc_10 VML w72 + VCP
patches), device-only cubin in WSL (no torch build). Two nvcc at a time.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/mef_sass.py" OUTDIR arm[,arm...] [--dump]
"""
import pathlib
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/box")]
import m32e_sass_local as S  # noqa: E402
import vcp_run as M  # noqa: E402


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2].split(",")
    ns = M.extract_ns()
    src = M.patch(ns["_vml_CUDA"])
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    cu = out / "mef_dev.cu"
    cu.write_text(src)
    res, sem = {}, threading.Semaphore(2)

    def job(nm):
        with sem:
            cub = out / f"{nm}.cubin"
            src_a, fl = M.arm_src_flags(ns, nm)
            src_a = src_a[:src_a.index("void vml_run(")]
            for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
                src_a = src_a.replace(inc, "")
            cu = out / f"{nm}.cu"
            cu.write_text(src_a)
            d = [f for f in fl if not f.startswith("-gencode")]
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *d, "-Xptxas=-v", "-o", str(cub), str(cu)],
                               capture_output=True, text=True)
            if r.returncode:
                res[nm] = ("FAIL", (r.stdout + r.stderr)[-1500:], None)
                return
            info = [ln.strip()[:150] for ln in (r.stdout + r.stderr).splitlines() if "Used" in ln or "spill" in ln or "serializ" in ln]
            s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            if "--dump" in sys.argv:
                (out / f"{nm}.sass").write_text(s)
            ser = any("serializ" in ln for ln in info)
            dep = {k: s.count("WARPGROUP.DEPBAR.LE gsb0, 0x" + k) for k in "0123"}
            info = [("SERIALIZED " if ser else "") + "depbar " + "/".join(str(dep[k]) for k in "0123")] + [ln for ln in info if "Used" in ln]
            res[nm] = ("ok", info, S.counts(s))
    ths = [threading.Thread(target=job, args=(nm,)) for nm in names]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for nm in names:
        st, info, per = res[nm]
        if st != "ok":
            print(f"[vcp-sass] {nm} FAIL {info}")
            continue
        for fn, c in per.items():
            if "vm_main" in fn:
                print(f"[vcp-sass] {nm:7s} total {c['total']} HGMMA {c['HGMMA']} STL {c['STL']} LDL {c['LDL']} | {info[0]} | {info[-1][-60:]}", flush=True)


if __name__ == "__main__":
    main()

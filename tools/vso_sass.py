"""vso_sass.py: ptxas regs / spills / SASS counts of vm_main for R-vso arms (k3_vso_<n> VML), device-only cubin in WSL.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vso_sass.py" LAB OUTDIR var[,var...]     (var 'ship:w72' = gc_16 source)
Two nvcc at a time.
"""
import ast
import pathlib
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
import m32e_sass_local as S  # noqa: E402


def ns_of(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    ns = {}
    for node in tree.body:
        names = [node.name] if isinstance(node, ast.FunctionDef) else []
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)] + \
                    [t.value.id for t in node.targets if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)]
        if any(n.startswith("_vml_") or n.startswith("_VSO") or n == "_VML_DEFS" for n in names):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    return ns


def main():
    lab, out = ROOT / sys.argv[1], pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    nsl, nss = ns_of(lab), ns_of(ROOT / "kernels/muse_decode_layer/k3_gc_16.py")
    res, sem = {}, threading.Semaphore(2)

    def job(nm):
        with sem:
            ns = nss if nm.startswith("ship:") else nsl
            v = nm.split(":")[-1]
            src = ns["_vml_CUDA"]
            src = src[:src.index("void vml_run(")]
            for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
                src = src.replace(inc, "")
            tag = nm.replace(":", "_")
            cu, cub = out / f"{tag}.cu", out / f"{tag}.cubin"
            cu.write_text(src)
            fl = [f for f in list(ns["_vml_FLAGS"]) + list(ns["_VML_DEFS"][v]) if not f.startswith("-gencode")]
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(cu)],
                               capture_output=True, text=True)
            if r.returncode:
                res[nm] = ("FAIL", (r.stdout + r.stderr)[-1200:], None)
                return
            info = [ln.strip()[:160] for ln in (r.stdout + r.stderr).splitlines()
                    if ("vm_main" in ln or "Used" in ln or "spill" in ln or "serializ" in ln)]
            s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            res[nm] = ("ok", info, S.counts(s))
    ths = [threading.Thread(target=job, args=(nm,)) for nm in sys.argv[3].split(",")]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for nm in sys.argv[3].split(","):
        st, info, per = res[nm]
        if st != "ok":
            print(f"[vso-sass] {nm} FAIL {info}")
            continue
        used = [ln for ln in info if "Used" in ln]
        spill = sorted({ln[-70:] for ln in info if "spill" in ln and "0 bytes spill" not in ln})
        for fn, c in per.items():
            if "vm_main" in fn:
                print(f"[vso-sass] {nm:12s} total {c['total']} HGMMA {c['HGMMA']} STL {c['STL']} LDL {c['LDL']} | "
                      f"{used[0][-50:] if used else ''} | spills {spill[:2]}", flush=True)


if __name__ == "__main__":
    main()

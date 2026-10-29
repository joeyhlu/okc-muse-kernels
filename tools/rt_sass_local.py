"""rt_sass_local.py: sm_90a cubin of a layer file's VML vm_main (VM_RT 0 vs 1, bf2 / w72 defines) in WSL: ptxas registers /
spills and SASS opcode counts, no torch build, no GPU (tools/m32e_sass_local.py's method on the layer's own _vml_ source).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/rt_sass_local.py" OUTDIR [layer=kernels/muse_decode_layer/k3_gc_7.py]
"""
import ast
import pathlib
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import m32e_sass_local as MS  # noqa: E402

VARS = {"bf2": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_FSW=1", "-DVM_SYNC7=1", "-DVM_NOB9=1")}
VARS["w72"] = VARS["bf2"] + ("-DVM_WG5=2", "-DVM_PREG=32", "-DVM_CREG=112", "-DVM_GV=2", "-DVM_JN=72", "-DVM_XSN=72")


def layer_cuda(path):
    t = (ROOT / path).read_text(encoding="utf-8")
    ns = {}
    for n in ast.parse(t).body:
        if isinstance(n, ast.FunctionDef) and n.name in ("_vml_wg_spec", "_vml_wg_spec64"):
            exec(compile(ast.Module([n], []), path, "exec"), ns)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in (
                "_vml_CUDA_T", "_vml_JMAX", "_vml_SLOTS", "_vml_FIXW", "_vml_FLAGS"):
            ns[n.targets[0].id] = eval(compile(ast.Expression(n.value), path, "eval"), dict(ns))
    src = (ns["_vml_CUDA_T"].replace("@WGSPECS@", "".join(ns["_vml_wg_spec"](n) + ns["_vml_wg_spec64"](n) for n in range(8, 129, 8)))
           .replace("@JMAX@", str(ns["_vml_JMAX"])).replace("@SLOTS@", str(ns["_vml_SLOTS"])).replace("@FIXW@", str(ns["_vml_FIXW"])))
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src, [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")]


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    kv = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    src, flags = layer_cuda(kv.get("layer", "kernels/muse_decode_layer/k3_gc_7.py"))
    cu = out / "vml_layer.cu"
    cu.write_text(src)
    jobs = [(v, rt) for v in VARS for rt in (0, 1)]
    res = {}

    def job(v, rt):
        cub = out / f"{v}_{rt}.cubin"
        cmd = ["nvcc", "-cubin", "-arch=sm_90a", *flags, *VARS[v], f"-DVM_RT={rt}", "-Xptxas=-v", "-o", str(cub), str(cu)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            res[(v, rt)] = ("FAIL", (r.stdout + r.stderr)[-3000:], None)
            return
        s = subprocess.run([MS.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        (out / f"{v}_{rt}.sass").write_text(s)
        info = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if "Used" in ln or "spill" in ln]
        res[(v, rt)] = ("ok", "\n".join(info), MS.counts(s))
    for i in range(0, len(jobs), 2):   # at most 2 compiles at once (WSL)
        ths = [threading.Thread(target=job, args=j) for j in jobs[i:i + 2]]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    for (v, rt) in jobs:
        st, info, per = res[(v, rt)]
        print(f"[rt-sass] {v} VM_RT={rt}: {st}")
        if st != "ok":
            print(info)
            continue
        print("\n".join("   " + ln[:170] for ln in info.splitlines()[-4:]))
        for fn, c in per.items():
            if "vm_main" in fn:
                print(f"   {fn[:30]} total {c['total']} " + " ".join(f"{k} {c[k]}" for k in MS.KEYS), flush=True)


if __name__ == "__main__":
    main()

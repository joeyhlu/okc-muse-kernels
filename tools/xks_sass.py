"""xks_sass.py: ptxas regs / spills and SASS counts (HMMA, LDSM, BAR, STL / LDL) of vm_main for tools/box/xks_run.py arms
(k3_gc_18 VML + tools/xks_patches.py), device-only cubin in WSL (no torch build, no GPU). Two nvcc at a time.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/xks_sass.py" OUTDIR var:arm[,var:arm...]      (var = bf2 / w72)
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/box")]
import m32e_sass_local as S  # noqa: E402
import xks_patches as X  # noqa: E402

LAYER = ROOT / "kernels/muse_decode_layer/k3_gc_18.py"
ARMS = {"ship": None, "base": (), "xn": ("-DXKS_XNULL=1",), "nw": ("-DXKS_NOW=1",), "xnnw": ("-DXKS_XNULL=1", "-DXKS_NOW=1"),
        "xks": ("-DXKS=1",), "xksnw": ("-DXKS=1", "-DXKS_NOW=1")}


def extract_ns():
    tree = ast.parse(LAYER.read_text(encoding="utf-8"))
    ns = {"hashlib": hashlib, "time": time}
    for node in tree.body:
        names = []
        if isinstance(node, ast.FunctionDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(n.startswith("_vml_") or n == "_VML_DEFS" for n in names):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(LAYER), "exec"), ns)
            except Exception:
                pass
    return ns


def dev(src):
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    items = [x.split(":") for x in sys.argv[2].split(",")]
    ns = extract_ns()
    srcs = {"ship": out / "ship_dev.cu", "pat": out / "xks_dev.cu"}
    srcs["ship"].write_text(dev(ns["_vml_CUDA"]))
    srcs["pat"].write_text(dev(X.patch(ns["_vml_CUDA"])))
    res, sem = {}, threading.Semaphore(2)

    def job(var, arm):
        with sem:
            cub = out / f"{var}_{arm}.cubin"
            fl = [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")] + list(ns["_VML_DEFS"][var]) + list(ARMS[arm] or ())
            cu = srcs["ship" if ARMS[arm] is None else "pat"]
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, "-Xptxas=-v", "-o", str(cub), str(cu)], capture_output=True, text=True)
            if r.returncode:
                res[(var, arm)] = ("FAIL", [ln[-300:] for ln in (r.stdout + r.stderr).splitlines() if "error" in ln][:6], None, None)
                return
            info = [ln.strip()[:150] for ln in (r.stdout + r.stderr).splitlines() if "vm_main" in ln or "Used" in ln or "spill" in ln]
            s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
            (out / f"{var}_{arm}.sass").write_text(s)
            body = s[s.index("vm_main"):] if "vm_main" in s else s
            body = body[:body.index("Function :", 10)] if "Function :" in body[10:] else body
            ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z0-9_]+)", body)
            cnt = {k: sum(1 for o in ops if o == k) for k in ("HMMA", "LDSM", "BAR", "STL", "LDL", "STS", "LDS")}
            cnt["total"] = len(ops)
            res[(var, arm)] = ("ok", info, cnt, hashlib.sha1(re.sub(r"/\*[0-9a-f]{4}\*/", "", body).encode()).hexdigest()[:10])
    ths = [threading.Thread(target=job, args=tuple(it)) for it in items]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for var, arm in items:
        st, info, cnt, h = res[(var, arm)]
        if st != "ok":
            print(f"[xks-sass] {var}:{arm} FAIL {info}")
            continue
        regs = " / ".join(i.split("Used ")[1][:16] for i in info if "Used" in i)
        spill = [i for i in info if "spill" in i and " 0 bytes spill stores" not in i]
        print(f"[xks-sass] {var}:{arm:5s} {cnt} sha {h} | regs {regs} | {spill[:2]}", flush=True)


if __name__ == "__main__":
    main()

"""vrb_sass.py: device-only sm_90a cubins of the VRB arms (tools/gen_k3_vrb.py, k3_vrb_0 VML source) in WSL: ptxas regs /
spills of vm_main + vm_fin, SASS opcode totals, SASS hash (== k3_gc_14 for the 'ship' arms). Two nvcc at a time.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vrb_sass.py" OUTDIR bf2:ship,bf2:0,bf2:c,... [--dump]
arm = var:codes, codes 'ship' = the k3_gc_14 source itself, '0' = k3_vrb_0 with no rider flag.
"""
import ast
import hashlib
import pathlib
import re
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
import gen_k3_vrb as G  # noqa: E402
import m32e_sass_local as S  # noqa: E402


def ns_of(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    ns = {"hashlib": hashlib}
    for node in tree.body:
        names = [node.name] if isinstance(node, ast.FunctionDef) else (
            [t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign) else [])
        if any(n.startswith("_vml_") or n == "_VML_DEFS" for n in names):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
            except Exception:
                pass
    return ns


def dev(ns):
    src = ns["_vml_CUDA"]
    src = src[:src.index("void vml_run(")]
    for inc in ("#include <torch/extension.h>", "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAException.h>"):
        src = src.replace(inc, "")
    return src, [f for f in ns["_vml_FLAGS"] if not f.startswith("-gencode")]


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    arms = sys.argv[2].split(",")
    nss = {"ship": ns_of(G.BASE), "vrb": ns_of(ROOT / "kernels/lab/k3_vrb_0.py")}
    cus = {}
    for k, ns in nss.items():
        src, fl = dev(ns)
        (out / f"{k}.cu").write_text(src)
        cus[k] = (out / f"{k}.cu", fl, ns["_VML_DEFS"])
    res, sem = {}, threading.Semaphore(2)

    def job(arm):
        var, codes = arm.split(":")
        cu, fl, D = cus["ship" if codes == "ship" else "vrb"]
        ex = () if codes in ("ship", "0") else G.defs(var, codes)
        cub = out / f"{var}_{codes}.cubin"
        with sem:
            r = subprocess.run(["nvcc", "-cubin", "-arch=sm_90a", *fl, *D[var], *ex, "-Xptxas=-v", "-o", str(cub), str(cu)],
                               capture_output=True, text=True)
        if r.returncode:
            res[arm] = "FAIL\n" + "\n".join((r.stdout + r.stderr).splitlines()[-12:])
            return
        txt = r.stdout + r.stderr
        regs = {}
        fn = None
        for ln in txt.splitlines():
            m = re.search(r"Compiling entry function '(\S+)'", ln)
            if m:
                fn = "main" if "vm_main" in m.group(1) else ("fin" if "vm_fin" in m.group(1) else None)
            m = re.search(r"Used (\d+) registers", ln)
            if m and fn:
                regs[fn] = int(m.group(1))
            m = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ln)
            if m and fn and (int(m.group(1)) or int(m.group(2))):
                regs[fn + "_spill"] = m.group(0)
        s = subprocess.run([S.OBJ, "-sass", str(cub)], capture_output=True, text=True).stdout
        if "--dump" in sys.argv:
            (out / f"{var}_{codes}.sass").write_text(s)
        per = S.counts(s)
        tot = {("main" if "vm_main" in f else "fin"): c["total"] for f, c in per.items() if "vm_main" in f or "vm_fin" in f}
        body = re.sub(r"/\*[0-9a-f]{4}\*/", "", s)
        h = hashlib.sha1("\n".join(ln for ln in body.splitlines() if "Function" in ln or ln.strip().startswith("/*")
                                   or re.match(r"\s+[A-Z@]", ln)).encode()).hexdigest()[:10]
        res[arm] = f"regs {regs} sass {tot} hash {h}"

    ths = [threading.Thread(target=job, args=(a,)) for a in arms]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for a in arms:
        print(f"[vrb-sass] {a:10s} {res[a]}", flush=True)


if __name__ == "__main__":
    main()

"""m32e_layer_compile.py: compile-only check (WSL, sm_90a, ptxas -v) of the VML block of a generated layer file in every variant
it routes (VML_VAR + VML_BVAR values), plus CPU checks of the m32e glue (cost / pool argument lists, route variant choice).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/m32e_layer_compile.py" kernels/lab/k3_m32e_1_ab.py
"""
import hashlib
import pathlib
import runpy
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    path = ROOT / sys.argv[1]
    src = path.read_text(encoding="utf-8")
    nl = chr(10)
    lines = src.split(nl)
    a = next(i for i, ln in enumerate(lines) if ln.startswith("_vml_E_LOC"))
    b = next(i for i, ln in enumerate(lines) if ln.startswith("VML_ENABLE = True"))
    glue = [ln for ln in lines[b:b + 40] if ln.startswith(("VML_COST = ", "VML_VAR = ", "VML_BVAR = ", "VML_POOL = ",
                                                              "VML_DPFLAGS = ", "_VML_DEFS = ", '             "d2"'))]
    ns = {}
    exec(nl.join(["import hashlib", "import time", "import torch"] + lines[a:b] + glue), ns)   # the layer's VML block only
    from torch.utils.cpp_extension import load_inline
    variants = (ns["VML_VAR"],) + tuple(dict.fromkeys(v for v in ns["VML_BVAR"].values() if v != ns["VML_VAR"]))
    res = {}

    def job(v):
        t0 = time.time()
        try:
            fl = ns["_vml_FLAGS"] + list(ns["_VML_DEFS"][v]) + ["-Xptxas=-v"]
            tag = hashlib.sha1((ns["_vml_CPP"] + ns["_vml_CUDA"] + " ".join(fl)).encode()).hexdigest()[:10]
            mod = load_inline(name=f"m32el_{v}_{tag}", cpp_sources=ns["_vml_CPP"], cuda_sources=ns["_vml_CUDA"],
                              functions=["vml_run", "vml_cfg"], extra_cuda_cflags=fl, verbose=True)
            cfg = tuple(mod.vml_cfg())
            cost = ns["_vml_cost_ext"](mod, ns["VML_COST"][256])
            pool = ns["_vml_pool_ext"](mod, ns["VML_POOL"].get(256, (0, 0)))
            res[v] = ("ok", time.time() - t0, cfg, len(cost), pool)
        except Exception as ex:   # noqa: BLE001
            res[v] = ("FAIL", time.time() - t0, repr(ex)[-2000:])
    ths = [threading.Thread(target=job, args=(v,)) for v in variants]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for v in variants:
        print(f"[m32e-lc] variant {v!r}: {res[v]}", flush=True)
    print(f"[m32e-lc] VML_BVAR {ns['VML_BVAR']} VML_POOL {ns['VML_POOL']} VML_DPFLAGS {ns['VML_DPFLAGS']} defs "
          f"{ {k: ns['_VML_DEFS'][k] for k in variants} }", flush=True)


if __name__ == "__main__":
    main()

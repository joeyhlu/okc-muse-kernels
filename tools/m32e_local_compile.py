"""m32e_local_compile.py: compile-only check of kernels/lab/m32e_moe.py variants for sm_90a in WSL (no H200 needed).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/m32e_local_compile.py" [variant ...]

Variants: base (= VML v13 'fs' settings), wide (WIDE_DEFS), wide_mk (wide + VM_MKW=64), base_mk (VM_MKW=64). Prints ptxas
registers / spill lines for vm_main / vm_fin and the module's vml_cfg(). Builds run in parallel threads (like PT1).
"""
import pathlib
import re
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import m32e_patches as P  # noqa: E402

VARS = {"base": ("-DVM_TRICKS_DQ=0",), "wide": ("-DVM_TRICKS_DQ=0",) + P.WIDE_DEFS,
        "wide_mk": ("-DVM_TRICKS_DQ=0", "-DVM_MKW=64") + P.WIDE_DEFS, "base_mk": ("-DVM_TRICKS_DQ=0", "-DVM_MKW=64"),
        "wg5a": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1"), "wg5b": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1") + P.WIDE_DEFS,
        "wg5c": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1", "-DVM_JN=96", "-DVM_NX=2", "-DVM_XSN=112"),
        "wg5d": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1", "-DVM_PREG=24", "-DVM_CREG=120") + P.WIDE_DEFS,
        "wg5e": ("-DVM_WG5=1",),
        "y1": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1", "-DVM_PREG=32", "-DVM_CREG=120", "-DVM_GV=4"),
        "y2": ("-DVM_WG5=1", "-DVM_PREG=32", "-DVM_CREG=120", "-DVM_GV=4"),
        "y3": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1", "-DVM_PREG=32", "-DVM_CREG=120", "-DVM_GV=4", "-DVM_JN=72", "-DVM_XSN=72"),
        "dp": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1"), "dpm": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1", "-DVM_MKW=64"),
        "dp2": ("-DVM_TRICKS_DQ=0", "-DVM_DP=1", "-DVM_DPI=2"), "bm": ("-DVM_TRICKS_DQ=0", "-DVM_MKW=64"),
        "sc": ("-DVM_TRICKS_DQ=0", "-DVM_SC=1"), "scm": ("-DVM_TRICKS_DQ=0", "-DVM_SC=1", "-DVM_MKW=64"),
        "cb": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1"), "cd": ("-DVM_TRICKS_DQ=0", "-DVM_DEFER2=1"),
        "cbd": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_DEFER2=1"),
        "ps": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_PS=1"), "psm": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_PS=1", "-DVM_MKW=64"),
        "ad": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_AD=1"), "adm": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_AD=1", "-DVM_MKW=64"),
        "gf": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_GSM=1", "-DVM_FSW=1"),
        "gfm": ("-DVM_TRICKS_DQ=0", "-DVM_BRED=1", "-DVM_GSM=1", "-DVM_FSW=1", "-DVM_MKW=64"),
        "y4": ("-DVM_TRICKS_DQ=0", "-DVM_WG5=1", "-DVM_PREG=32", "-DVM_CREG=120", "-DVM_GV=4", "-DVM_JN=80", "-DVM_NX=2", "-DVM_XSN=80")}


def main():
    import io
    import contextlib
    import m32e_moe as V
    names = sys.argv[1:] or ["base", "wide"]
    res = {}

    def job(nm):
        t0 = time.time()
        buf = io.StringIO()
        try:
            from torch.utils.cpp_extension import load_inline
            fl = V.FLAGS + list(VARS[nm]) + ["-Xptxas=-v"]
            import hashlib
            tag = hashlib.sha1((V.CPP + V.CUDA + " ".join(fl)).encode()).hexdigest()[:10]
            mod = load_inline(name=f"m32e_{nm}_{tag}", cpp_sources=V.CPP, cuda_sources=V.CUDA, functions=["vml_run", "vml_cfg"],
                              extra_cuda_cflags=fl, verbose=True)
            res[nm] = ("ok", time.time() - t0, tuple(mod.vml_cfg()))
        except Exception as ex:   # noqa: BLE001
            res[nm] = ("FAIL", time.time() - t0, repr(ex)[-3000:])

    ths = [threading.Thread(target=job, args=(nm,)) for nm in names]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for nm in names:
        print(f"[m32e-cc] {nm}: {res[nm][0]} in {res[nm][1]:.0f} s: {res[nm][2]}", flush=True)


if __name__ == "__main__":
    main()

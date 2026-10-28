"""mb_local_test.py: local checks of a generated MoE board loop file (kernels/muse_moe_w4a16/mb_<n>.py), no H200 needed.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      env TORCH_CUDA_ARCH_LIST=9.0a python "<repo>/tools/mb_local_test.py" <n> [builds]

1. compiles every MB_BUILDS variant for sm_90a (parallel threads, like prepare), with ptxas -v: registers / spills per kernel;
2. CPU checks of the glue: bucket map, board-like and layer-like input builders (shapes, distinct ids per token, local counts
   equal the embedded layer profiles), arm parsing, cost lookup.
"""
import pathlib
import re
import runpy
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    do_builds = len(sys.argv) <= 2 or sys.argv[2] != "nobuild"
    path = ROOT / f"kernels/muse_moe_w4a16/mb_{n}.py"
    ns = runpy.run_path(str(path))
    import torch
    from torch.utils.cpp_extension import load_inline

    # ---- 2. glue checks on CPU
    bk = ns["_mb_bucket"]
    assert [bk(m) for m in (1, 4, 5, 8, 12, 13, 16, 24, 25, 32, 33, 48, 49, 128, 192, 193, 256, 257, 320)] == \
        [1, 1, 8, 8, 8, 16, 16, 16, 32, 32, 32, 32, 128, 128, 128, 256, 256, 256, 256]
    assert bk(321) is None
    for bs in (1, 8, 16, 32, 128, 256):
        for dist in ("normal", "skewed", "hot"):
            d = ns["_mb_board_input"](bs, 3, "cpu", dist)
            ids = d["topk_ids"]
            assert ids.shape == (bs, 8) and ids.dtype == torch.int32
            assert all(len(set(r.tolist())) == 8 for r in ids), "duplicate expert in a token row"
            assert int(ids.min()) >= 0 and int(ids.max()) < 256
            loc = torch.bincount(ids.flatten().long(), minlength=256)
            per_gpu = loc.view(8, 32).sum(1)
            assert int(per_gpu.argmax()) == 0 or int(per_gpu[0]) == int(per_gpu.max()), "busiest GPU is not 0-31"
            assert torch.allclose(d["topk_weights"].sum(-1), torch.ones(bs), atol=1e-5)
        for i, c in enumerate(ns["MB_LAYER_PROFILES"].get(bs, [])):
            d = ns["_mb_layer_input"](bs, c, 200 + i, "cpu")
            ids = d["topk_ids"]
            assert all(len(set(r.tolist())) == 8 for r in ids), ("dup", bs, c)
            cnt = torch.bincount(ids.flatten().long(), minlength=256)[:32]
            got = sorted([int(v) for v in cnt if v > 0], reverse=True)
            assert got == sorted(c, reverse=True), (bs, c, got)
    for rep, specs in ns["MB_ARMS"].items():
        for s in specs:
            a = ns["_mb_arm"](s)
            assert a["b"] in ns["MB_BUILDS"], s
            if a["kind"] == "vml" and a["ck"] == "L":
                assert rep in ns["MB_LAYER_COST"], (rep, s)
    print(f"[mb_local] glue checks PASS (layer profiles {[(k, len(v)) for k, v in ns['MB_LAYER_PROFILES'].items()]})", flush=True)

    if not do_builds:
        return
    # ---- 1. sm_90a compile-only builds (nvcc -c, ptxas -v), one subprocess per variant in parallel
    import subprocess
    import sysconfig
    from torch.utils import cpp_extension as ce
    inc = []
    for d in ce.include_paths(device_type="cuda") + [sysconfig.get_paths()["include"]]:
        inc += ["-I", d]
    nvcc = str(pathlib.Path(ce.CUDA_HOME) / "bin" / "nvcc")
    tmp = pathlib.Path("/tmp/mb_local")
    tmp.mkdir(exist_ok=True)
    names = list(ns["MB_BUILDS"]) if len(sys.argv) <= 2 else sys.argv[2].split(",")
    procs = {}
    for nm in names:
        b = ns["MB_BUILDS"][nm]
        if b[0] == "vms":
            flags = ["-O3", "-DVMS_G=132", f"-DVMS_NS={b[1]}", f"-DVMS_MAXT={b[2]}", f"-DVMS_FL={b[3]}",
                     f"-DVMS_PAD={b[4] if len(b) > 4 else 0}", f"-DVMS_WT={b[5] if len(b) > 5 else 0}",
                     "-gencode=arch=compute_90a,code=sm_90a"]
            src = ns["MB_VMS_CUDA"]
        else:
            flags = [f for f in ns["_vml_FLAGS"] if not f.startswith("-Xptxas")] + list(b[1])
            src = ns["_vml_CUDA"]
        cu = tmp / f"{nm}.cu"
        cu.write_text(src)
        cmd = [nvcc, "-c", str(cu), "-o", str(tmp / f"{nm}.o"), "-std=c++20", "-Xptxas=-v", "--expt-relaxed-constexpr",
               "-DTORCH_EXTENSION_NAME=mbl", "-D_GLIBCXX_USE_CXX11_ABI=1"] + flags + inc
        procs[nm] = cmd
    done = {}

    def run_one(nm):
        t0 = time.time()
        pr = subprocess.run(procs[nm], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        done[nm] = (pr, pr.stdout, time.time() - t0)
    pending = list(procs)
    while pending:   # at most 3 nvcc at a time (WSL memory)
        batch, pending = pending[:3], pending[3:]
        ths = [threading.Thread(target=run_one, args=(nm,)) for nm in batch]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    for nm in procs:
        pr, log, dt = done[nm]
        info = re.findall(r"Compiling entry function '(\S+)'.*?(\d+) bytes stack frame, (\d+) bytes spill stores, "
                          r"(\d+) bytes spill loads.*?Used (\d+) registers", log, flags=re.S)
        short = [(f[0][-22:], f"regs {f[4]} spill {f[2]}/{f[3]}") for f in info]
        st = "ok" if pr.returncode == 0 else f"FAIL rc {pr.returncode}"
        print(f"[mb_local] build {nm} {ns['MB_BUILDS'][nm]}: {st} {dt:.0f}s {short}", flush=True)
        warn = [ln for ln in log.splitlines() if "warning" in ln.lower() or "error" in ln.lower()]
        if pr.returncode != 0:
            print(log[-3000:], flush=True)
        elif warn:
            print("   " + chr(10).join(warn[:6]), flush=True)


if __name__ == "__main__":
    main()

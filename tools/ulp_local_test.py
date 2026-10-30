"""ulp_local_test.py: local (RTX 3070, sm_86) correctness check of the batched / NN cuBLASLt plan added by tools/ulp_patches.py.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/ulp_local_test.py" [kernels/lab/k3_ulp_1.py]

Extracts LT_CPP from the generated layer file, builds it, and runs lt_plan_b with the HEURISTIC's first algorithm (algo 66 /
nvjet is sm_90 only, so no timing and no pin here) for (nn, batch) in {0, 1} x {1, 2, 4} on the unpooler shape (N 8192, K 4096)
at T = 128 / 256, against torch.mm. This validates the layouts (A transposed storage, batch offsets, shared x, C slices) and
that lt_plan (old signature) is unchanged; the sm_90 pin itself is validated by the layer's prepare self-test on the H200.
"""
import re
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_ulp_1.py"
src = open(path, encoding="utf-8").read()
m = re.search(r"^LT_CPP = r'''(.*?)'''", src, re.S | re.M)
assert m, "LT_CPP not found"
cpp = m.group(1)
import glob
import os
from torch.utils.cpp_extension import CUDA_HOME, load_inline

dirs = [os.path.join(CUDA_HOME, d) for d in ("lib64", "lib", os.path.join("targets", "x86_64-linux", "lib"))] if CUDA_HOME else []
tdir = os.path.dirname(torch.__file__)
dirs += glob.glob(os.path.join(os.path.dirname(tdir), "nvidia", "*", "lib")) + [os.path.join(tdir, "lib")]
flags = ["-lcublasLt"]
for d in dirs:
    libs = sorted(glob.glob(os.path.join(d, "libcublasLt.so*")))
    if libs:
        lib = "-lcublasLt" if any(x.endswith("libcublasLt.so") for x in libs) else "-l:" + os.path.basename(libs[-1])
        flags = ["-L" + d, lib, "-Wl,-rpath," + d]
        break
inc = glob.glob(os.path.join(os.path.dirname(tdir), "nvidia", "*", "include"))
mod = load_inline(name="ulp_local_test_lt", cpp_sources=cpp, functions=["lt_plan", "lt_plan_b", "lt_cfg", "lt_make", "lt_run"],
                  extra_ldflags=flags, extra_include_paths=inc + ([os.path.join(CUDA_HOME, "include")] if CUDA_HOME else []),
                  with_cuda=True, verbose=False)
print("[ulp-local] built", flush=True)

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(5)
N, K = 8192, 4096
w = (torch.randn((N, K), generator=g, device=dev) * 0.02).to(torch.bfloat16)
wt = w.t().contiguous()
ws = torch.empty((32 << 20,), dtype=torch.uint8, device=dev)
ok = True
for T in (128, 256):
    x = torch.randn((T, K), generator=g, device=dev).to(torch.bfloat16)
    ref = torch.mm(x, w.t())
    for nn in (0, 1):
        for bt in (1, 2, 4):
            code = int(mod.lt_plan_b(T, N, K, 0, nn, bt, 32 << 20, 32))
            pid, cnt = code >> 8, code & 255
            if cnt == 0:
                print(f"[ulp-local] T={T} nn={nn} batch={bt}: heuristic returned 0 algos (sm_86)", flush=True)
                continue
            y = torch.full((T, N), float("nan"), dtype=torch.bfloat16, device=dev)
            try:
                mod.lt_run(pid, 0, wt if nn else w, x, y, 0.0, ws)
                torch.cuda.synchronize()
            except Exception as ex:  # noqa: BLE001
                print(f"[ulp-local] T={T} nn={nn} batch={bt}: run raised {str(ex)[:120]}", flush=True)
                ok = False
                continue
            fin = bool(torch.isfinite(y).all())
            e = float((y.float() - ref.float()).norm() / ref.float().norm()) if fin else float("inf")
            good = fin and e < 1e-2
            ok &= good
            print(f"[ulp-local] T={T} nn={nn} batch={bt}: {cnt} heuristic algos, finite {fin}, rel vs torch.mm {e:.2e} -> "
                  f"{'ok' if good else 'FAIL'}", flush=True)
    # the old entry point is unchanged
    code = int(mod.lt_plan(T, N, K, 0, 32 << 20, 32))
    y = torch.full((T, N), float("nan"), dtype=torch.bfloat16, device=dev)
    mod.lt_run(code >> 8, 0, w, x, y, 0.0, ws)
    torch.cuda.synchronize()
    e = float((y.float() - ref.float()).norm() / ref.float().norm())
    ok &= e < 1e-2
    print(f"[ulp-local] T={T} old lt_plan: rel {e:.2e}", flush=True)
print("[ulp-local] " + ("ALL PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)

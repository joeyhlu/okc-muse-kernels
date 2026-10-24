"""k2b_v1_build.py: box helper (CPU only, no GPU work). Builds K2b v1's CUDA extension exactly as prepare() does and
reports the build time and ptxas resource usage per variant (registers, spills, smem, wgmma serialisation notes).
  python3 kernels/muse_prefill_layer/k2b_v1_build.py [--lab] [--cold] [--configs a,b]
--cold: unique extension name + CCACHE_DISABLE=1 (what a fresh judge sandbox pays for Gate A).
Not a submission file: uses os.environ for the cache switch.
"""
import argparse
import os
import pathlib
import re
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", action="store_true")
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--configs", default="")
    ap.add_argument("--src", default="k2b_v1.py")
    ap.add_argument("--exact", action="store_true", help="same name/flags as prepare() (warms the judge-clone build)")
    args = ap.parse_args()
    src = (HERE / args.src).read_text()
    g = {"torch": torch}
    pure = src.split("# ===== K2B PURE BEGIN =====")[1].split("# ===== K2B PURE END =====")[0].split("\n", 1)[1]
    exec(compile(pure, args.src + ":pure", "exec"), g)
    cfgs = g["k2b_all_configs"](args.lab)
    if args.configs:
        cfgs = {k: cfgs[k] for k in args.configs.split(",")}
    variants = g["k2b_variants"](cfgs)
    cu = g["k2b_gen_source"](variants)
    name = "k2b_v1_build_%d" % (int(time.time() * 1000) % 10 ** 9) if args.cold else "k2b_v1_build_warm"
    if args.cold:
        os.environ["CCACHE_DISABLE"] = "1"
    flags = ["-O3", "--expt-relaxed-constexpr", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-warn-spills",
             "-Xptxas=-v"]
    if args.exact:   # mirror _k2b1_build
        import hashlib
        name = "k2b_v1_" + hashlib.sha1((g["K2B_CPP_SRC"] + cu).encode()).hexdigest()[:10]
        flags = flags if args.lab else flags[:-1]
    t0 = time.time()
    import io
    import contextlib
    buf = io.StringIO()
    try:
        mod = load_inline(name=name, cpp_sources=g["K2B_CPP_SRC"], cuda_sources=cu, functions=["run", "smem_bytes"],
                          extra_cuda_cflags=flags, verbose=True)
        ok = True
    except Exception as ex:  # noqa: BLE001
        ok = False
        print("BUILD FAILED:", str(ex)[-6000:])
    dt = time.time() - t0
    print(f"[build] {len(variants)} variants ({', '.join(cfgs)}): {'ok' if ok else 'FAILED'} in {dt:.1f}s "
          f"(cold={args.cold})", flush=True)
    if ok:
        for i, v in enumerate(variants):
            print(f"[build] vid {i} Cfg{v}: smem {mod.smem_bytes(i)}")


if __name__ == "__main__":
    main()

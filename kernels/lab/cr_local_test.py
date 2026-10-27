"""Local checks of the CR-K1 lab kernel (kernels/lab/cr_k1_lab.py).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/kernels/lab/cr_local_test.py" [sm_86|sm_90a]

sm_86 (RTX 3070, cp.async producer, 1-slot ring): correctness on routings at bs 1 / 3 / 8 / 32 / 128 / 256 (incl. a
multi-pass hot expert) vs the FP32 reference, error flags, self-reset counters, repeat-call agreement.
sm_90a: compile only, ptxas registers / spills (-Xptxas -v).
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import cr_k1_lab as L  # noqa: E402
import nvfp4  # noqa: E402


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
    if arch == "sm_90a":
        import os
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        from torch.utils.cpp_extension import load_inline
        try:
            load_inline(name="cr_k1_sm90_probe", cpp_sources=L.CR_CPP, cuda_sources=L.CR_CUDA,
                        functions=["cr_run", "cr_attrs"],
                        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-v,-warn-spills",
                                           "-gencode=arch=compute_90a,code=sm_90a"], verbose=True)
        except Exception as ex:  # noqa: BLE001  (loading a sm_90a-only module on the 3070 may fail after the build)
            print("load:", repr(ex)[:300])
        return
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5252)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous()}
    mod, dt = L.cr_build(arch="sm_86")
    print(f"built {dt:.0f}s attrs {list(mod.cr_attrs())}")
    ws, gsc = L.cr_prepack(w["w13"], w["w13_sf"], w["w13_gs"])
    G = torch.cuda.get_device_properties(0).multi_processor_count
    st = L.CrState(mod, ws, gsc, G, dev)
    allok = True
    for bs, seed in ((1, 11), (1, 12), (3, 13), (8, 14), (32, 15), (128, 16), (256, 17), ("L32", 0), ("L16", 2)):
        if isinstance(bs, str):
            d = L.cr_case_counts(int(bs[1:]), L.CR_LAYER_COUNTS[int(bs[1:])][seed], 5150 + seed, dev)
        else:
            d = L.cr_case(bs, 660000 + seed, dev)
        for kappa in (0.22, 0.30):
            P = L.cr_plan(d, G, kappa=kappa)
            rel, worst, err, cl, nan, inter = L.cr_check(st, d, w, P, L.cr_dequant)
            inter1 = inter.clone()
            rel2, _, err2, cl2, nan2, inter2 = L.cr_check(st, d, w, P, L.cr_dequant)
            rep = float((inter1.float() - inter2.float()).norm() / inter1.float().norm().clamp(min=1e-30))
            # null mode leaves counters untouched
            ok = rel <= 1e-2 and worst <= 5e-2 and err[0] == 0 and cl == 0 and not nan and rep <= 1e-4 and err2[0] == 0
            allok &= ok
            print(f"bs={bs} kappa={kappa}: experts {len(P['experts'])} tokens {P['cnts'][:6]} units {P['nunits']} "
                  f"R {P['R']} | rel {rel:.2e} worst {worst:.2e} err {err} cnt-clean {cl == 0} nan {nan} repeat {rep:.1e} "
                  f"-> {'OK' if ok else 'FAIL'}")
    mod8, _ = L.cr_build(arch="sm_86", ncw=8, ngmax=6)
    print(f"V attrs {list(mod8.cr_attrs())}")
    st8 = L.CrState(mod8, ws, gsc, 2 * G, dev)
    for bs, seed in ((1, 21), (8, 22), (128, 16), (256, 17), ("L32", 1)):
        if isinstance(bs, str):
            d = L.cr_case_counts(int(bs[1:]), L.CR_LAYER_COUNTS[int(bs[1:])][seed], 5150 + seed, dev)
        else:
            d = L.cr_case(bs, 660000 + seed, dev)
        P = L.cr_plan(d, 2 * G, kappa=0.22, tpass=48, nb=32)
        rel, worst, err, cl, nan, _ = L.cr_check(st8, d, w, P, L.cr_dequant)
        ok = rel <= 1e-2 and worst <= 5e-2 and err[0] == 0 and cl == 0 and not nan
        allok &= ok
        print(f"V bs={bs}: experts {len(P['experts'])} units {P['nunits']} | rel {rel:.2e} worst {worst:.2e} err {err} "
              f"cnt-clean {cl == 0} nan {nan} -> {'OK' if ok else 'FAIL'}")
    print("ALL OK" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()

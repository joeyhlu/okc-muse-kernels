"""Local test of the CR board file kernels/muse_moe_w4a16/cr_2.py on the RTX 3070 (sm_86 build, no PDL effect):
prepare() (incl. its own self-test), then judge-style checks on the board's routing law at several bs, repeat calls and
a CUDA-graph replay with changed inputs.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/kernels/lab/cr_2_test.py" [sm_86|sm_90a]
"""
import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
import nvfp4  # noqa: E402


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
    spec = importlib.util.spec_from_file_location("cr2_board", ROOT / "kernels/muse_moe_w4a16/cr_2.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    B.CR2_ARCH = arch
    if arch == "sm_90a":
        import os
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        from torch.utils.cpp_extension import load_inline
        try:
            load_inline(name="cr2_sm90_probe", cpp_sources=B.CR2_CPP, cuda_sources=B.CR2_CUDA,
                        functions=["cr2_call", "cr2_attrs"],
                        extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-Xptxas=-v,-warn-spills", "-DCR_NGMAX=8",
                                           "-DCR_NCW=16", "-DCR_NST=4", "-gencode=arch=compute_90a,code=sm_90a"],
                        verbose=True)
        except Exception as ex:  # noqa: BLE001
            print("load:", repr(ex)[:200])
        return
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5252)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(32,))
    w2, w2_sf, gs2 = nvfp4.random_weights(4096, 4096, generator=g, device=dev, leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
         "w2_gs": gs2.expand(32).contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device=dev)).to(torch.bfloat16)}
    st = B.prepare(w)
    allok = True
    for bs in (1, 2, 8, 16, 32, 64, 128, 256):
        d = B._cr2_case(bs, 4242 + bs, dev)
        ref = B._cr2_ref(d, w)
        out = B.kernel_fn(d, st)
        out2 = B.kernel_fn(d, st)
        torch.cuda.synchronize()
        rel = float((out.float() - ref).norm() / ref.norm())
        rep = float((out.float() - out2.float()).norm() / out.float().norm())
        ok = rel <= 1e-2 and int(st["err"][0]) == 0 and int(st["cnt"].abs().sum()) == 0 and rep <= 1e-3
        allok &= ok
        print(f"bs={bs}: rel {rel:.3e} repeat {rep:.1e} err {st['err'].tolist()} -> {'OK' if ok else 'FAIL'}")
    # graph replay with changed inputs (same shape): the plan is on device, so a captured graph must follow new routing
    bs = 32
    d = B._cr2_case(bs, 1, dev)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        B.kernel_fn(d, st)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            outg = B.kernel_fn(d, st)
    torch.cuda.synchronize()
    for seed in (2, 3):
        d2 = B._cr2_case(bs, seed, dev)
        d["hidden_states"].copy_(d2["hidden_states"]); d["topk_ids"].copy_(d2["topk_ids"]); d["topk_weights"].copy_(d2["topk_weights"])
        gr.replay()
        torch.cuda.synchronize()
        ref = B._cr2_ref(d2, w)
        rel = float((outg.float() - ref).norm() / ref.norm())
        ok = rel <= 1e-2
        allok &= ok
        print(f"graph replay changed inputs seed {seed}: rel {rel:.3e} -> {'OK' if ok else 'FAIL'}")
    print("ALL OK" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()

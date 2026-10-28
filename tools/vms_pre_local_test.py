"""Local check of VMS-PRE (tools/vms_pre.py) on the RTX 3070: tail_kernel2 (sm_86 copy) + VMS v1 vs tail_kernel2 + VMS-PRE
(own routing from the logits partials, VMS_ABS_NORM = 0): global ids bitwise, output rel < 1e-4, near-ties, BF16 / FP32.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vms_pre_local_test.py"
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import vms_abs  # noqa: E402
import vms_abs_local_test as AT  # noqa: E402
import vms_moe_lab as V  # noqa: E402
import vms_pre  # noqa: E402
import vms_local_test as VT  # noqa: E402


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    t = (ROOT / "kernels/lab/vms_moe_lab_v1.py").read_text(encoding="utf-8")
    cuda0 = t.split("CUDA_SRC = r'''", 1)[1].split("'''", 1)[0].replace(
        "  // ---- routing (warp 0, redundant per CTA): pairs sorted by expert, jobs of <= 8 pairs" + chr(10), "")
    cpp0 = t.split("CPP_SRC = r'''", 1)[1].split("'''", 1)[0]
    cuda, cpp = vms_pre.pre_sources(cuda0, cpp0)
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=2", "-gencode=arch=compute_86,code=sm_86"]
    m1 = load_inline(name="vmspre_v1_86", cpp_sources=cpp, cuda_sources=cuda, functions=["run", "attrs"], extra_cuda_cflags=flags)
    mp = load_inline(name="vmspre_pre_86", cpp_sources=cpp, cuda_sources=cuda, functions=["run_abs", "attrs"],
                     extra_cuda_cflags=flags + ["-DVMS_ABS=1", "-DVMS_ABS_NORM=0"])
    tm = AT.tail_module()
    print(f"[pre] built: v1 {list(m1.attrs())} pre {list(mp.attrs())}")
    w = VT.make_weights(dev)
    pk = V.prepack(w)
    sc = V.scratch(dev)
    g0 = torch.Generator(device=dev).manual_seed(9)
    nwx = (1.0 + 0.1 * torch.randn((4096,), generator=g0, device=dev)).to(torch.bfloat16)
    bias = (0.05 * torch.randn((256,), generator=g0, device=dev)).float()
    tida = torch.zeros((16, 8), dtype=torch.int32, device=dev)
    ok_all = True
    for T, seed, kind, ydt, ns in ((1, 1, "normal", torch.bfloat16, 8), (1, 2, "normal", torch.float32, 8), (2, 3, "tie", torch.bfloat16, 8),
                                   (5, 4, "normal", torch.float32, 8), (8, 5, "tie", torch.bfloat16, 8), (8, 6, "normal", torch.float32, 1),
                                   (8, 7, "tie", torch.float32, 8)):
        g = torch.Generator(device=dev).manual_seed(seed)
        y = torch.randn((T, 4096), generator=g, device=dev).to(ydt).contiguous()
        lg = (torch.randn((ns, T, 256), generator=g, device=dev) * 0.5).contiguous()
        ls = int(torch.randint(0, 8, (1,), generator=g, device=dev).item()) * 32
        b = bias.clone()
        if kind == "tie":
            lg[:, :, ls:ls + 40] = 0.7
            b[ls:ls + 40] = b[ls]
        lst = torch.tensor([ls], dtype=torch.int32, device=dev)
        for k in range(2):
            xn = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
            tid = torch.empty((T, 8), dtype=torch.int32, device=dev)
            lid = torch.empty((T, 8), dtype=torch.int32, device=dev)
            tm.v8_tail(y, lg, 0, -1, lg.stride(1), ns, lg.stride(0), nwx, b, lst, xn, tw, tid, lid, 1e-6, 0.1767766952966369, 1.0, 1e-15, 2)
            ref = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            m1.run(xn, lid, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"], ref,
                   sc["nomarks"], T, 1e-8, 0, 0)
            out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            sc["ctr"][2] = 0
            mp.run_abs(xn, xn, lg, b, lst, nwx, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"],
                       sc["ctr"], out, sc["nomarks"], T, 1e-8, 1e-6, 0.1767766952966369, 1.0, 1e-15, 0, tida)
            torch.cuda.synchronize()
            ids_ok = bool(torch.equal(tida[:T], tid))
            rel = float((out.float() - ref.float()).norm() / ref.float().norm())
            err = int(sc["ctr"][2])
            ok = ids_ok and rel < 1e-4 and err == 0
            ok_all &= ok
            print(f"[pre] T={T} {kind:6s} y={str(ydt)[6:]:8s} ns={ns} call {k}: ids {ids_ok} rel {rel:.1e} err {err} -> {'ok' if ok else 'FAIL'}")
    print(f"[pre] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

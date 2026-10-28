"""Local check of VMS-ABS (tools/vms_abs.py) on the RTX 3070 (sm_86): tail_kernel2 (k3_slim_3's V8 tail source, PDL
instructions removed for sm_86) + VMS v1 vs one VMS-ABS launch on the same inputs: bitwise outputs and xn, incl. near-tie
top-8 cases, BF16 and FP32 pooler outputs, T = 1 / 2 / 5 / 8, repeated calls.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vms_abs_local_test.py"
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import vms_abs  # noqa: E402
import vms_moe_lab as V  # noqa: E402
import vms_local_test as VT  # noqa: E402


def tail_module():
    from torch.utils.cpp_extension import load_inline
    s = (ROOT / "kernels/muse_decode_layer/k3_slim_3.py").read_text(encoding="utf-8")
    src = {}
    for n in ast.parse(s).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("V8_TAIL_CPP_SRC", "V8_TAIL_CUDA_SRC"):
            src[n.targets[0].id] = n.value.value
    cuda = src["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    cuda = cuda.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    return load_inline(name="vmsabs_tail86", cpp_sources=src["V8_TAIL_CPP_SRC"], cuda_sources=cuda,
                       functions=["v8_tail", "v8_router32"], extra_cuda_cflags=["-O3"], verbose=False)


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    cuda, cpp = vms_abs.lab_sources()
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=2", "-gencode=arch=compute_86,code=sm_86"]
    m1 = load_inline(name="vmsabs_v1_86", cpp_sources=cpp, cuda_sources=cuda, functions=["run", "attrs"], extra_cuda_cflags=flags)
    ma = load_inline(name="vmsabs_abs_86", cpp_sources=cpp, cuda_sources=cuda, functions=["run_abs", "attrs"],
                     extra_cuda_cflags=flags + ["-DVMS_ABS=1"])
    tm = tail_module()
    print(f"[abs] built: v1 {list(m1.attrs())} abs {list(ma.attrs())}")
    w = VT.make_weights(dev)
    pk = V.prepack(w)
    sc = V.scratch(dev)
    xnb = torch.zeros((16, 4096), dtype=torch.bfloat16, device=dev)
    g0 = torch.Generator(device=dev).manual_seed(9)
    nwx = (1.0 + 0.1 * torch.randn((4096,), generator=g0, device=dev)).to(torch.bfloat16)
    bias = (0.05 * torch.randn((256,), generator=g0, device=dev)).float()
    ok_all = True
    for T, seed, kind, ydt, ns in ((1, 1, "normal", torch.bfloat16, 8), (1, 2, "normal", torch.float32, 8), (2, 3, "tie", torch.bfloat16, 8),
                                   (5, 4, "normal", torch.float32, 8), (8, 5, "tie", torch.bfloat16, 8), (8, 6, "normal", torch.float32, 1),
                                   (8, 7, "tie", torch.float32, 8), (3, 8, "normal", torch.bfloat16, 1)):
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
            tm.v8_tail(y, lg, 0, -1, lg.stride(1), ns, lg.stride(0), nwx, b, lst, xn, tw, tid, lid, 1e-6, 0.1767766952966369,
                       1.0, 1e-15, 2)
            ref = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            m1.run(xn, lid, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"], ref,
                   sc["nomarks"], T, 1e-8, 0, 0)
            ref2 = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            m1.run(xn, lid, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"], ref2,
                   sc["nomarks"], T, 1e-8, 0, 0)
            torch.cuda.synchronize()
            v1self = bool(torch.equal(ref, ref2))
            out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            tida = torch.empty((T, 8), dtype=torch.int32, device=dev)
            sc["ctr"][2] = 0
            ma.run_abs(xnb, y, lg, b, lst, nwx, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"],
                       sc["ctr"], out, sc["nomarks"], T, 1e-8, 1e-6, 0.1767766952966369, 1.0, 1e-15, 0, tida)
            torch.cuda.synchronize()
            same = bool(torch.equal(out, ref))
            xs = bool(torch.equal(xnb[:T], xn))
            err = int(sc["ctr"][2])
            dv = xnb[8:].reshape(-1).view(torch.int32)
            ids_ok = bool(torch.equal(dv[:T * 8], lid.reshape(-1))) and bool(torch.equal(tida, tid))
            tw_ok = bool(torch.equal(dv[128:128 + T * 8].view(torch.float32), tw.reshape(-1)))
            rel = float((out.float() - ref.float()).norm() / ref.float().norm())
            ok = xs and ids_ok and tw_ok and rel < 1e-4 and err == 0
            same = f"{same} ids {ids_ok} tw {tw_ok}"
            ok_all &= ok
            print(f"[abs] T={T} {kind:6s} y={str(ydt)[6:]:8s} ns={ns} call {k}: out bitwise {same} xn bitwise {xs} err {err} "
                  f"local routes {int((lid >= 0).sum())} v1-vs-v1 bitwise {v1self} max|d| {float((out.float() - ref.float()).abs().max()):.3e} rel {float((out.float() - ref.float()).norm() / ref.float().norm()):.2e} -> {'ok' if ok else 'FAIL'}")
    print(f"[abs] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

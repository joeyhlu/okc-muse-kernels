"""Local ($0) timing of VMS base vs VMS-NTL build variants on the RTX 3070 (sm_86), same routings, same weights
(tools/ntl_local_test.py helpers). Variants: -DVMS_NTL_NI (routing in a __noinline__ function) x -DVMS_NTL_XN (0 = no xn
transform: timing only, 1 = at the job switch, 2 = after the job's first weight wait). Prints per T the median us per call
(CUDA events, 200 calls) and the NTL - base delta, plus registers; a correctness spot check vs base (XN 1 / 2 only).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/ntl_local_time.py" kernels/lab/k3_ntl_5.py
"""
import statistics
import sys

import torch

import ntl_local_test as NT

VARIANTS = (("ni0xn1", 0, 1), ("ni1xn1", 1, 1), ("ni0xn2", 0, 2), ("ni1xn2", 1, 2), ("ni1xn0", 1, 0))


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    src, ns = NT.lab_parts()
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=2", "-DVMS_RO=1", "-gencode=arch=compute_86,code=sm_86"]
    mb = load_inline(name="ntlt_base86", cpp_sources=src["VMS_CPP_SRC"], cuda_sources=src["VMS_CUDA_SRC"],
                     functions=["run", "attrs"], extra_cuda_cflags=flags)
    mods = {}
    for name, ni, xn in VARIANTS:
        mods[name] = load_inline(name=f"ntlt_{name}_86", cpp_sources=src["VMS_CPP_SRC"] + src["_NTL_VMS_CPP"],
                                 cuda_sources=src["VMS_CUDA_SRC"], functions=["run", "run_ntl", "attrs"],
                                 extra_cuda_cflags=flags + ["-DVMS_NTL=1", f"-DVMS_NTL_NI={ni}", f"-DVMS_NTL_XN={xn}"])
    print(f"[ntl-time] regs base {list(mb.attrs())[:2]} " + " ".join(f"{k} {list(m.attrs())[:2]}" for k, m in mods.items()), flush=True)
    cuda = src["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    cuda = cuda.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    tm = load_inline(name="ntl_tail86", cpp_sources=src["V8_TAIL_CPP_SRC"], cuda_sources=cuda, functions=["v8_tail", "v8_router32"],
                     extra_cuda_cflags=["-O3"])
    w = NT.VT.make_weights(dev)
    E = w["w13"].shape[0]
    pk = ns["_vms_prepack_expert"]
    wA = torch.stack([pk(w["w13"][e], w["w13_sf"][e], True) for e in range(E)]).contiguous()
    wB = torch.stack([pk(w["w2"][e], w["w2_sf"][e], False) for e in range(E)]).contiguous()
    gsA = (w["w13_gs"].float().reshape(E) * float(2 ** 62)).contiguous()
    gsB = (w["w2_gs"].float().reshape(E) * float(2 ** 62)).contiguous()
    nwp = (1.0 + w["post_norm_w"].float()).contiguous()
    del w
    inter = torch.zeros((128, 4096), dtype=torch.bfloat16, device=dev)
    ybuf = torch.zeros((128, 4096), dtype=torch.float32, device=dev)
    ssv = torch.zeros((256,), dtype=torch.float32, device=dev)
    ctr = torch.zeros((72,), dtype=torch.int32, device=dev)
    nom = torch.empty((0,), dtype=torch.int64, device=dev)
    g0 = torch.Generator(device=dev).manual_seed(9)
    xw = (1.0 + 0.1 * torch.randn((4096,), generator=g0, device=dev)).to(torch.bfloat16)
    bias = (0.05 * torch.randn((256,), generator=g0, device=dev)).float()
    ssum = torch.zeros((32,), dtype=torch.float32, device=dev)
    tw_o = torch.zeros((128,), dtype=torch.float32, device=dev)
    lid_o = torch.zeros((128,), dtype=torch.int32, device=dev)
    eps = 1e-5
    for T in (1, 8, 16):
        gg = torch.Generator(device=dev).manual_seed(500 + T)
        y = (torch.randn((T, 4096), generator=gg, device=dev) * 0.7).to(torch.bfloat16).contiguous()
        lg = (torch.randn((8, T, 256), generator=gg, device=dev) * 0.5).contiguous()
        lst = torch.tensor([96], dtype=torch.int32, device=dev)
        ss0 = y.float().pow(2).sum(-1)
        xn = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
        tid = torch.empty((T, 8), dtype=torch.int32, device=dev)
        lid = torch.empty((T, 8), dtype=torch.int32, device=dev)
        tm.v8_tail(y, lg, 0, -1, lg.stride(1), 8, lg.stride(0), xw, bias, lst, xn, tw, tid, lid, eps, NT.MULT, 1.0, 1e-15, 2)
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        tid2 = torch.empty((T, 8), dtype=torch.int32, device=dev)

        def base():
            mb.run(xn, lid, tw, wA, wB, gsA, gsB, nwp, inter, ybuf, ssv, ctr, out, nom, T, 1e-8, 0, 0)

        def ntl(m):
            m.run_ntl(y, lg, bias, lst, ssum, xw, tid2, tw_o, lid_o, wA, wB, gsA, gsB, nwp, inter, ybuf, ssv, ctr, out, nom, T,
                      1e-8, eps, NT.MULT, 1.0, 1e-15, 0)

        def timeit(fn, refill):
            ts = []
            for i in range(240):
                if refill:
                    ssum[:T].copy_(ss0)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn()
                b.record()
                b.synchronize()
                if i >= 40:
                    ts.append(a.elapsed_time(b) * 1000.0)
            return statistics.median(ts)

        base()
        ref = out.clone()
        tb = timeit(base, False)
        row = [f"T={T:2d} local {int((lid >= 0).sum()):3d} base {tb:6.1f}"]
        for name, m in mods.items():
            ssum[:T].copy_(ss0)
            ntl(m)
            torch.cuda.synchronize()
            rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
            tn = timeit(lambda: ntl(m), True)
            row.append(f"{name} {tn:6.1f} ({tn - tb:+5.1f}{'' if name.endswith('xn0') else f', rel {rel:.0e}'})")
        print("[ntl-time] " + " | ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

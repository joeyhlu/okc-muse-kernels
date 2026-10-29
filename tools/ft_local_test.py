"""ft_local_test.py (R-ft-0 step 0, $0): FAST-TAIL tail_kernel3 vs the ship's tail_kernel2 on the local RTX 3070 (sm_86;
griddepcontrol removed for the local build). Reference = k3_gc_20's own V8 tail source (unhardened tail_kernel2, pdl 2);
arms = the lab file's tail with pdl 2 (hardened tail2), 4 (tail3 RW 2), 8 (tail3 RW 8). Checks xn / tw / tid / lid bitwise.
Cases: T = 1..16, 32, 33, 64, 128, 256 x value kinds (normal, exact ties, 1-ulp near ties, hot experts, duplicated biases)
x input variants (bf16 y + 8 partials = the live path, bf16 + 2D logits, FP32 y with logits inside, FP32 y + partials)
x local_start 0 / 96 / 224, two calls each.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/ft_local_test.py" [kernels/lab/k3_ft_1.py]
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_ft_1.py")
BASE = ROOT / "kernels/muse_decode_layer/k3_gc_20.py"


def _srcs(path):
    out = {}
    for n in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("V8_TAIL_CPP_SRC", "V8_TAIL_CUDA_SRC"):
            out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def _build(path, name):
    from torch.utils.cpp_extension import load_inline
    s = _srcs(path)
    cu = s["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    cu = cu.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    return load_inline(name=name, cpp_sources=s["V8_TAIL_CPP_SRC"], cuda_sources=cu, functions=["v8_tail"],
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"], verbose=False)


def run(mod, y, logits, nw, bias, ls, flag):
    T, dev = y.shape[0], y.device
    xn = torch.full((T, 4096), 7.0, dtype=torch.bfloat16, device=dev)
    tw = torch.full((T, 8), -1.0, device=dev)
    tid = torch.full((T, 8), -9, dtype=torch.int32, device=dev)
    lid = torch.full((T, 8), -9, dtype=torch.int32, device=dev)
    ns, sst = 1, 0
    if logits is None:
        lg, la, lb, lld = y, 4096, 4096 + 256, y.stride(0)
    elif logits.dim() == 3:
        lg, la, lb, lld, ns, sst = logits, 0, -1, logits.stride(1), logits.shape[0], logits.stride(0)
    else:
        lg, la, lb, lld = logits, 0, -1, logits.stride(0)
    mod.v8_tail(y, lg, la, lb, lld, ns, sst, nw, bias, ls, xn, tw, tid, lid, 1e-5, 0.1767766952966369, 1.0, 1e-15, flag)
    return xn, tw, tid, lid


def main():
    dev = torch.device("cuda")
    ref = _build(BASE, "ft_ref86")
    lab = _build(LAB, "ft_lab86_" + str(abs(hash(LAB.read_text(encoding="utf-8"))) % 10**8))
    gen = torch.Generator(device=dev).manual_seed(777)
    nw = (1 + 0.1 * torch.randn(4096, generator=gen, device=dev)).to(torch.bfloat16)
    bias0 = 0.05 * torch.randn(256, generator=gen, device=dev)
    fails, n, ties = {2: 0, 4: 0, 8: 0}, 0, 0
    Ts = list(range(1, 17)) + [32, 33, 64, 128, 256]
    for T in Ts:
        for kind in ("normal", "exact", "ulp", "hot", "dupbias"):
            ls = torch.tensor([[0, 96, 224][(T + len(kind)) % 3]], dtype=torch.int32, device=dev)
            bias = bias0
            lg = torch.randn((T, 256), generator=gen, device=dev) * 3.0
            part = torch.randn((8, T, 256), generator=gen, device=dev) * 1.1
            if kind == "exact":
                lg, bias = torch.round(lg), torch.zeros_like(bias0)
            elif kind == "ulp":   # neighbours 1 ulp apart and equal pairs, bias 0
                lg = torch.round(lg)
                lg[:, 1::4] = torch.nextafter(lg[:, 0::4], torch.full_like(lg[:, 0::4], 1e9))
                lg[:, 2::4] = lg[:, 0::4]
                bias = torch.zeros_like(bias0)
            elif kind == "hot":
                lg[:, 5] += 40.0
                lg[:, 77] += 40.0
                lg[:, 200:208] += 25.0
            elif kind == "dupbias":
                lg, bias = torch.round(lg * 2) / 2, torch.round(bias0 * 40) / 40
            if kind != "normal":
                part = (lg.unsqueeze(0).repeat(8, 1, 1) * 0.125).contiguous()
            bias = bias.contiguous()
            ties += int((torch.sigmoid(part.sum(0) * 0.1767766952966369) + bias).sort(1, descending=True).values[:, 6:9]
                        .diff(1).eq(0).any(1).sum())
            pool = torch.randn((T, 4096), generator=gen, device=dev).to(torch.bfloat16)
            y32 = torch.randn((T, 4096 + 512), generator=gen, device=dev)
            y32[:, 4096:4352] = lg
            y32[:, 4352:] = 0.0 if kind != "normal" else 0.25 * lg
            for yy, ll in ((pool, part), (pool, lg.contiguous()), (y32, None), (y32, part)):
                r = [a.clone() for a in run(ref, yy, ll, nw, bias, ls, 2)]
                for f in (2, 4, 8):
                    for rep in range(2):
                        g = run(lab, yy, ll, nw, bias, ls, f)
                        ok = all(bool(torch.equal(a, b)) for a, b in zip(r, g))
                        if not ok:
                            fails[f] += 1
                            if fails[f] <= 3:
                                print(f"FAIL flag {f} T={T} {kind} y {yy.dtype} logits {None if ll is None else tuple(ll.shape)}: "
                                      f"tid eq {torch.equal(r[2], g[2])} tw eq {torch.equal(r[1], g[1])} xn eq {torch.equal(r[0], g[0])}")
                        n += 1
    torch.cuda.synchronize()
    print(f"[ft_local_test] {LAB.name}: {n} runs, rows with ties at ranks 7-9: {ties}; fails per flag {fails} -> "
          f"{'ALL PASS' if not any(fails.values()) else 'FAIL'}")


if __name__ == "__main__":
    main()

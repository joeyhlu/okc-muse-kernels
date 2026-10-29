"""Local test of UNF (tools/gen_k3_unf.py) on the layer's VMS source, RTX 3070 (sm_86: cp.async ring NS 1, rows copied sync).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/unf_local_test.py" [layer=kernels/lab/k3_unf_1.py] [n=200]

sm_86 smem (99 KB) caps the slice at n rows (<= 8 per CTA on 46 SMs). Builds base (VMS_UNF 0) and UNF (AT 1, AT 0) from the
same text. Checks per case and call: y bitwise = base y, f vs FP32(y . W^T) rel < 4e-3, uo untouched outside [0, m) x [0, n),
err flag 0, ctr[4] == 64 * ctr[0]; graph replays with changed inputs.
"""
import ast
import hashlib
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vms_moe_lab_v1 as V  # noqa: E402
from vms_local_test import make_weights, make_input  # noqa: E402


def seg(L, name):
    a = L.index(name + ' = r"""') + len(name + ' = r"""')
    return L[a:L.index('"""', a)]


def build(L, G, unf, n, at, name):
    from torch.utils.cpp_extension import load_inline
    cu, cpp = seg(L, "VMS_CUDA_SRC"), seg(L, "VMS_CPP_SRC")
    fns = ["run", "attrs"]
    if unf:
        cpp = seg(L, "UNF_CPP_SRC") if 'UNF_CPP_SRC = r"""' in L else cpp + seg(L, "UNF_CPP_SRC")
        fns = ["run", "run_unf", "attrs"]
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=1", "-DVMS_RO=1", f"-DVMS_UNF={unf}", f"-DVMS_UNF_N={n}", f"-DVMS_UNF_AT={at}",
             "-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
    tag = hashlib.sha1((cu + cpp + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=cu, functions=fns, extra_cuda_cflags=flags)


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    lay = opt.get("layer", "kernels/lab/k3_unf_1.py")
    n = int(opt.get("n", 200))
    L = (ROOT / lay).read_text(encoding="utf-8")
    for nd in ast.parse(L).body:
        if isinstance(nd, ast.Assign) and getattr(nd.targets[0], "id", None) == "VMS_POS":
            V.POS = ast.literal_eval(nd.value)
    L2 = L.replace('UNF_CPP_SRC = VMS_CPP_SRC + r"""', 'UNF_CPP_SRC = r"""' + seg(L, "VMS_CPP_SRC"), 1)
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    m0 = build(L2, G, 0, n, 1, "unf_l0")
    mu = {at: build(L2, G, 1, n, at, f"unf_l1a{at}") for at in (1, 0)}
    print(f"[unf] sm_86 builds ok: base {list(m0.attrs())} unf AT1 {list(mu[1].attrs())} AT0 {list(mu[0].attrs())}")
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    g = torch.Generator(device=dev).manual_seed(77)
    uw = (torch.randn((n, 4096), generator=g, device=dev) * 0.02).to(torch.bfloat16).contiguous()
    uld = n + 40
    sc0 = V.scratch(dev)
    ok_all = True

    def run_unf(mod, sc, x, ids, tw, uo, out=None):
        mm = x.shape[0]
        if out is None:
            out = torch.empty((mm, 4096), dtype=torch.bfloat16, device=dev)
        mod.run_unf(x, ids, tw, pk["wA"], pk["wB"], pk["gsA"], pk["gsB"], pk["nw"], sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"],
                    out, sc["nomarks"], mm, 1e-8, 0, 0, uw, uo, uld)
        return out

    def chk(y, yb, uo, T, sc):
        ref = y.float() @ uw.float().t()
        ry, _ = V.rel_err(y, yb)
        f = uo[:T, :n].float()
        rel = float((f - ref).norm() / ref.norm().clamp_min(1e-30))
        outside = bool(uo[T:].any()) or bool(uo[:, n:].any())
        c = sc["ctr"]
        err, coh = int(c[2]), int(c[4]) == 64 * int(c[0])
        good = ry < 1e-3 and rel < 4e-3 and not outside and err == 0 and coh and bool(torch.isfinite(f).all())
        return good, f"y vs base {ry:.1e} (bitwise {torch.equal(y, yb)}) f rel {rel:.2e} outside {'0' if not outside else 'DIRTY'} err {err} ctr {'ok' if coh else 'BAD'}"

    cases = [(1, 1, "normal"), (1, 2, "hot"), (2, 3, "normal"), (5, 5, "skew"), (8, 7, "hot"), (8, 8, "normal"), (3, 12, "nolocal"),
             (1, 13, "nolocal")]
    for at in (1, 0):
        sc = V.scratch(dev)
        for T, seed, kind in cases:
            x, ids, tw = make_input(T, seed, dev, kind)
            if kind == "nolocal" and T == 1:
                ids.fill_(-1)
            yb = V.call(m0, pk, sc0, x, ids, tw).clone()
            if at == 1 and kind != "nolocal":
                yb2 = V.call(m0, pk, sc0, x, ids, tw)
                torch.cuda.synchronize()
                print(f"[unf] base vs base T={T} {kind}: bitwise {torch.equal(yb, yb2)} rel {V.rel_err(yb2, yb)[0]:.1e}")
            for k in range(3):
                uo = torch.zeros((8, uld), dtype=torch.bfloat16, device=dev)
                y = run_unf(mu[at], sc, x, ids, tw, uo)
                torch.cuda.synchronize()
                good, msg = chk(y, yb, uo, T, sc)
                ok_all &= good
                if k == 0 or not good:
                    print(f"[unf] AT{at} T={T} {kind:7s} call {k}: {msg} -> {'ok' if good else 'FAIL'}")
        for T in (1, 8):
            x, ids, tw = make_input(T, 100 + T, dev, "hot")
            out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            uo = torch.zeros((8, uld), dtype=torch.bfloat16, device=dev)
            run_unf(mu[at], sc, x, ids, tw, uo, out)
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                run_unf(mu[at], sc, x, ids, tw, uo, out)
            for r in range(3):
                x2, ids2, tw2 = make_input(T, 200 + 10 * T + r, dev, "normal" if r == 1 else "hot")
                x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
                gr.replay()
                torch.cuda.synchronize()
                yb = V.call(m0, pk, sc0, x2, ids2, tw2)
                torch.cuda.synchronize()
                good, msg = chk(out, yb, uo, T, sc)
                ok_all &= good
                print(f"[unf] AT{at} graph T={T} replay {r}: {msg} -> {'ok' if good else 'FAIL'}")
    print(f"[unf] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

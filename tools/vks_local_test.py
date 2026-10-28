"""Local test of VMS-KS (tools/vks_patches.py) on the layer's VMS source, RTX 3070 (sm_86: cp.async ring, NS from argv).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vks_local_test.py" [ns=1] [er=0] [layer=kernels/muse_decode_layer/k3_gc_4.py]

Builds base (VMS_KS 0) and KS (VMS_KS 1) from the same patched text, prepack / refs from kernels/lab/vms_moe_lab_v1.py.
Checks per case and call: rel vs FP32 (< 6e-3, row < 3e-2), KS vs base rel (< 1e-3), err flag 0, barrier counter coherent
(ctr[1] == ctr[0] * G), every fixup counter back at 0; graph replays with changed inputs (hot / normal).
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vms_moe_lab_v1 as V  # noqa: E402
import vks_patches as K  # noqa: E402
from vms_local_test import make_weights, make_input  # noqa: E402

FX_ROWS = 16 * 384 * 2 * 128 * 4 // 4096   # fixup partials behind ybuf: [16 J1 jobs][384 pairs][2 h][4][32] float4


def layer_src(path):
    L = (ROOT / path).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    b = L.index('"""', a)
    c = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""') if 'VMS_CPP_SRC = r"""' in L else None
    cpp = L[c:L.index('"""', c)] if c else None
    return L[a:b], cpp


def layer_pos(path):
    import ast
    for n in ast.parse((ROOT / path).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "VMS_POS":
            return ast.literal_eval(n.value)
    raise KeyError("VMS_POS")


def build(src, cpp, G, ns, ks, er, name):
    from torch.utils.cpp_extension import load_inline
    import hashlib
    flags = ["-O3", f"-DVMS_G={G}", f"-DVMS_NS={ns}", f"-DVMS_KS={ks}", f"-DVMS_ER={er}", "-Xptxas=-warn-spills",
             "-gencode=arch=compute_86,code=sm_86"]
    tag = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run", "attrs"],
                       extra_cuda_cflags=flags)


def scratch(dev):
    sc = V.scratch(dev)
    sc["ybuf"] = torch.zeros((V.MAXP + FX_ROWS, V.HD), dtype=torch.float32, device=dev)
    sc["ctr"] = torch.zeros((8 + V.MAXJ + 16 * 384,), dtype=torch.int32, device=dev)
    return sc


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ns, er = int(opt.get("ns", 1)), int(opt.get("er", 0))
    lay = opt.get("layer", "kernels/muse_decode_layer/k3_gc_5.py")
    src0, cpp = layer_src(lay)
    V.POS = layer_pos(lay)   # the layer's VMS bit layout (UNP L2 on k3_gc_5) for the prepack
    if cpp is None:
        cpp = V.CPP_SRC
    src = K.apply_ks(src0)
    assert K.check_ks0(src0, src)
    if er:
        a = "  const int gen = S.gen;\n  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n"
        assert src.count(a) == 1
        src = src.replace(a, "  const int gen = S.gen;\n#if VMS_ER\n  if (c >= max(1, p.m * 4)) {\n"
                             "    if (tid == 0) { __threadfence(); atomicAdd(p.ctr + 1, 1); }\n    return;\n  }\n#endif\n"
                             "  if (tid == 0) {\n    __threadfence();\n    atomicAdd(p.ctr + 1, 1);\n")
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    m0 = build(src, cpp, G, ns, 0, 0, "vks_l0")
    m1 = build(src, cpp, G, ns, 1, er, "vks_l1")
    print(f"[vks] sm_86 builds ok: base attrs {list(m0.attrs())} KS attrs {list(m1.attrs())} (regs/local/smem/G/NS)")
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    sc0, sc1 = scratch(dev), scratch(dev)
    ok_all = True
    cases = [(1, 1, "normal"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"), (9, 21, "hot"), (10, 22, "hot"),
             (11, 23, "skew"), (12, 8, "skew"), (13, 24, "hot"), (16, 9, "normal"), (16, 10, "hot"), (16, 11, "skew"),
             (3, 12, "nolocal"), (16, 13, "nolocal"), (15, 25, "skew")]

    def chk(o, ref, ob, sc):
        tot, row = V.rel_err(o, ref)
        kb, _ = V.rel_err(o, ob)
        c = sc["ctr"]
        err = int(c[2])
        coh = int(c[1]) == int(c[0]) * G
        fz = int(c[8 + V.MAXJ:].abs().sum()) == 0
        good = tot < 6e-3 and row < 3e-2 and kb < 1e-3 and err == 0 and coh and fz and bool(torch.isfinite(o.float()).all())
        return good, f"rel {tot:.3e} row {row:.3e} ks-vs-base {kb:.2e} err {err} bar {'ok' if coh else 'BAD'} fx {'0' if fz else 'NONZERO'}"

    for T, seed, kind in cases:
        x, ids, tw = make_input(T, seed, dev, kind)
        ref = V.ref_fp32(x, ids, tw, w)
        cnt = torch.bincount(ids[ids >= 0].long(), minlength=32)
        ob = V.call(m0, pk, sc0, x, ids, tw)
        for k in range(3):
            o = V.call(m1, pk, sc1, x, ids, tw)
            torch.cuda.synchronize()
            good, msg = chk(o, ref, ob, sc1)
            ok_all &= good
            if k == 0 or not good:
                print(f"[vks] T={T:2d} {kind:7s} experts {int((cnt > 0).sum())} hot(>8) {int((cnt > 8).sum())} max {int(cnt.max())} "
                      f"call {k}: {msg} -> {'ok' if good else 'FAIL'}")
    for T in (8, 12, 16):
        x, ids, tw = make_input(T, 100 + T, dev, "hot")
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        V.call(m1, pk, sc1, x, ids, tw, out=out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            V.call(m1, pk, sc1, x, ids, tw, out=out)
        for r in range(3):
            x2, ids2, tw2 = make_input(T, 200 + 10 * T + r, dev, "normal" if r == 1 else "hot")
            x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
            gr.replay()
            torch.cuda.synchronize()
            ref = V.ref_fp32(x2, ids2, tw2, w)
            ob = V.call(m0, pk, sc0, x2, ids2, tw2)
            torch.cuda.synchronize()
            good, msg = chk(out, ref, ob, sc1)
            ok_all &= good
            print(f"[vks] graph T={T:2d} replay {r}: {msg} -> {'ok' if good else 'FAIL'}")
    print(f"[vks] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

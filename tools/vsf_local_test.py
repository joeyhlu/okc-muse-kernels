"""Local test of VMS_FX (tools/vsf_patches.py) on the layer's VMS source: RTX 3070 (sm_86: FX bits 1 / 2 active, bit 4 is
sm_90-only) bitwise vs the base build, plus sm_90a compile-only builds (regs / spills) of base and FX.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vsf_local_test.py" [fx=7] [layer=kernels/muse_decode_layer/k3_gc_7.py] [sm90=1]
"""
import hashlib
import pathlib
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vms_moe_lab_v1 as V  # noqa: E402
import vsf_patches as F  # noqa: E402
from vks_local_test import layer_src, layer_pos  # noqa: E402
from vms_local_test import make_weights, make_input  # noqa: E402


def build(src, cpp, G, fx, name):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=1", f"-DVMS_FX={fx}", "-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
    tag = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run", "attrs"], extra_cuda_cflags=flags)


def sm90(src, fx):
    with tempfile.TemporaryDirectory() as td:
        k = pathlib.Path(td) / "k.cu"
        body = src.split("#include <torch/extension.h>")[0]
        k.write_text(body, encoding="utf-8")
        r = subprocess.run(["nvcc", "-O3", "-DVMS_G=132", "-DVMS_NS=3", f"-DVMS_FX={fx}", "-Xptxas=-v,-warn-spills",
                            "-gencode=arch=compute_90a,code=sm_90a", "-c", str(k), "-o", str(pathlib.Path(td) / "k.o")],
                           capture_output=True, text=True)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "vms_kernel" in ln or "registers" in ln or "spill" in ln
                 or "error" in ln]
        return r.returncode, lines[-4:]


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    fx = int(opt.get("fx", 7))
    lay = opt.get("layer", "kernels/muse_decode_layer/k3_gc_7.py")
    src0, cpp = layer_src(lay)
    V.POS = layer_pos(lay)
    src = F.apply_fx(src0)
    assert F.check_fx0(src0, src), "FX 0 view != base"
    if int(opt.get("sm90", 1)):
        for f in (0, fx):
            rc, ln = sm90(src, f)
            print(f"[vsf] sm_90a FX={f}: rc {rc} | " + " | ".join(x.strip()[-110:] for x in ln))
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    m0 = build(src, cpp, G, 0, "vsf_l0")
    m1 = build(src, cpp, G, fx, "vsf_l1")
    print(f"[vsf] sm_86 builds ok: base {list(m0.attrs())} fx {list(m1.attrs())}")
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    sc0, sc1 = V.scratch(dev), V.scratch(dev)
    ok_all = True
    cases = [(1, 1, "normal"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"), (9, 21, "hot"), (11, 23, "skew"),
             (12, 8, "skew"), (16, 9, "normal"), (16, 10, "hot"), (3, 12, "nolocal"), (16, 13, "nolocal"), (15, 25, "skew")]

    def chk(o, ref, ob, sc):
        tot, row = V.rel_err(o, ref)
        c = sc["ctr"]
        same = bool(torch.equal(o, ob))
        err, coh = int(c[2]), int(c[1]) == int(c[0]) * G
        good = same and err == 0 and coh and tot < 6e-3 and row < 3e-2
        return good, f"rel {tot:.3e} row {row:.3e} bitwise {'==' if same else '!='} err {err} bar {'ok' if coh else 'BAD'}"

    for T, seed, kind in cases:
        x, ids, tw = make_input(T, seed, dev, kind)
        ref = V.ref_fp32(x, ids, tw, w)
        ob = V.call(m0, pk, sc0, x, ids, tw)
        for k in range(3):
            o = V.call(m1, pk, sc1, x, ids, tw)
            torch.cuda.synchronize()
            good, msg = chk(o, ref, ob, sc1)
            ok_all &= good
            if k == 0 or not good:
                print(f"[vsf] T={T:2d} {kind:7s} call {k}: {msg} -> {'ok' if good else 'FAIL'}")
    for T in (1, 8, 16):
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
            print(f"[vsf] graph T={T:2d} replay {r}: {msg} -> {'ok' if good else 'FAIL'}")
    print(f"[vsf] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

"""Local test of VMS2 builds (tools/vms2_patches.py) on the layer's VMS source: RTX 3070 (sm_86, NS 1; FX bit 4 is sm_90-only)
vs the FP32 reference and the base build (tolerance: VMS is not bitwise run to run), CUDA-graph replays, MK2 marks sanity,
plus sm_90a compile-only builds (regs / spills) of every variant.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vms2_local_test.py" [vars="fx:-DVMS_FX=7;ro:-DVMS_RO=1"] [sm90=1] [sm86=1]
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
import vms2_patches as P  # noqa: E402
from vks_local_test import layer_src, layer_pos  # noqa: E402
from vms_local_test import make_weights, make_input  # noqa: E402


def build(src, cpp, G, defs, name):
    from torch.utils.cpp_extension import load_inline
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=1"] + defs + ["-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
    tag = hashlib.sha1((src + " ".join(flags)).encode()).hexdigest()[:10]
    return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=src, functions=["run", "attrs"], extra_cuda_cflags=flags)


def sm90(src, defs):
    with tempfile.TemporaryDirectory() as td:
        k = pathlib.Path(td) / "k.cu"
        k.write_text(src.split("#include <torch/extension.h>")[0], encoding="utf-8")
        r = subprocess.run(["nvcc", "-O3", "-DVMS_G=132", "-DVMS_NS=3"] + defs + ["-Xptxas=-v,-warn-spills",
                            "-gencode=arch=compute_90a,code=sm_90a", "-c", str(k), "-o", str(pathlib.Path(td) / "k.o")],
                           capture_output=True, text=True)
        lines = [ln for ln in (r.stdout + r.stderr).splitlines() if "registers" in ln or "spill" in ln or "error" in ln]
        return r.returncode, lines[-3:]


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    var = [v.split(":", 1) for v in opt.get("vars", "fx:-DVMS_FX=7;ro:-DVMS_RO=1;n8:-DVMS_NRB=8;mk:-DVMS_MK2=1").split(";")]
    lay = opt.get("layer", "kernels/muse_decode_layer/k3_gc_14.py")
    src0, cpp = layer_src(lay)
    V.POS = layer_pos(lay)
    src = P.apply(src0)
    assert P.check0(src0, src), "default view != base"
    if int(opt.get("sm90", 1)):
        for nm, d in [("base", "")] + var:
            rc, ln = sm90(src, d.split())
            print(f"[vms2] sm_90a {nm}: rc {rc} | " + " | ".join(x.strip()[-90:] for x in ln), flush=True)
    if not int(opt.get("sm86", 1)):
        return 0
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    m0 = build(src, cpp, G, [], "vms2_l0")
    mods = {nm: build(src, cpp, G, d.split(), f"vms2_{nm}") for nm, d in var}
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    sc0 = V.scratch(dev)
    ok_all = True
    cases = [(1, 1, "normal"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"), (9, 21, "hot"), (11, 23, "skew"),
             (16, 9, "normal"), (16, 10, "hot"), (3, 12, "nolocal"), (15, 25, "skew")]

    def chk(o, ref, ob, sc):
        tot, row = V.rel_err(o, ref)
        tb, _ = V.rel_err(o, ob)
        c = sc["ctr"]
        err, coh = int(c[2]), int(c[1]) == int(c[0]) * G
        good = err == 0 and coh and tot < 6e-3 and row < 3e-2 and tb < 2e-3
        return good, f"rel {tot:.3e} row {row:.3e} vs base {tb:.1e} err {err} bar {'ok' if coh else 'BAD'}"

    for nm, m1 in mods.items():
        sc1 = V.scratch(dev)
        mk = "-DVMS_MK2=1" in dict(var)[nm]
        marks = torch.zeros((G * 32,), dtype=torch.int64, device=dev) if mk else None
        for T, seed, kind in cases:
            x, ids, tw = make_input(T, seed, dev, kind)
            ref = V.ref_fp32(x, ids, tw, w)
            ob = V.call(m0, pk, sc0, x, ids, tw)
            for k in range(2):
                o = V.call(m1, pk, sc1, x, ids, tw, marks=marks)
                torch.cuda.synchronize()
                good, msg = chk(o, ref, ob, sc1)
                ok_all &= good
                if k == 0 and (not good or T in (1, 16)):
                    print(f"[vms2] {nm} T={T:2d} {kind:7s}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
        if mk:
            r = marks.view(G, 32).cpu()
            n = r[:, 13]
            print(f"[vms2] {nm} marks row0 N {int(n[0]) & 0xffff} nA {(int(n[0]) >> 16) & 0xffff} jobs {int(n[0]) >> 32} "
                  f"cF {int(r[0, 10])} cE {int(r[0, 11])} cS {int(r[0, 12])} cC {int(r[0, 19])} epiW {int(r[0, 14])} "
                  f"epiB {int(r[0, 23])} cyc {int(r[0, 17] - r[0, 16])} ns {int(r[0, 9] - r[0, 0])}", flush=True)
        T = 8
        x, ids, tw = make_input(T, 108, dev, "hot")
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        V.call(m1, pk, sc1, x, ids, tw, out=out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            V.call(m1, pk, sc1, x, ids, tw, out=out)
        for rr in range(3):
            x2, ids2, tw2 = make_input(T, 300 + rr, dev, "normal" if rr == 1 else "hot")
            x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
            gr.replay()
            torch.cuda.synchronize()
            ref = V.ref_fp32(x2, ids2, tw2, w)
            ob = V.call(m0, pk, sc0, x2, ids2, tw2)
            torch.cuda.synchronize()
            good, msg = chk(out, ref, ob, sc1)
            ok_all &= good
            if not good or rr == 0:
                print(f"[vms2] {nm} graph replay {rr}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
    print(f"[vms2] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

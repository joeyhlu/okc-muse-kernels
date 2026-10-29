"""Local test of VFN builds (tools/vfn_patches.py) on the layer's VMS source (k3_gc_21cg_s1).
  sm90=1: sm_90a compile-only (WSL nvcc 13.4): ptxas registers / spills per arm, MEMBAR / fence lines in the SASS, and the
          identity check (patched source at the defaults -> same SASS as the base source).
  sm86=1: RTX 3070 (sm_86, cp.async ring, NS <= 2) vs the FP32 reference and the base build (VMS is not bitwise run to run:
          ss atomic order), 10 routings x 2 calls + 3 changed-input CUDA-graph replays per arm.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vfn_local_test.py" [sm90=1] [sm86=1]
"""
import hashlib
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vfn_patches as P  # noqa: E402

CUOBJDUMP = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
LAYER = "kernels/muse_decode_layer/k3_gc_21cg_s1.py"
SHIP = ["-DVMS_NS=3", "-DVMS_RO=1"]
# sm_86 variants (NS <= 2 fits 99 KB): k = 0 and k = NS - 1 code paths
SM86 = {"base1": "-DVMS_NS=1", "db0n1": "-DVMS_NS=1 -DVMS_DB=1 -DVMS_DBK=0 -DVMS_FM=1",
        "db0n2": "-DVMS_NS=2 -DVMS_DB=1 -DVMS_DBK=0 -DVMS_FM=1", "db1n2": "-DVMS_NS=2 -DVMS_DB=1 -DVMS_DBK=1 -DVMS_FM=1",
        "fx2db1n2": "-DVMS_NS=2 -DVMS_FX=2 -DVMS_DB=1 -DVMS_DBK=1 -DVMS_FM=1", "fmn2": "-DVMS_NS=2 -DVMS_FM=1"}


def layer_src(path):
    L = (ROOT / path).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    c = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""')
    return L[a:L.index('"""', a)], L[c:L.index('"""', c)]


def merge(base, defs):
    d = list(base)
    for x in defs:
        d = [y for y in d if y.split("=")[0] != x.split("=")[0]] + [x]
    return d


def sm90(src, defs, td, nm):
    k = pathlib.Path(td) / f"{nm}.cu"
    k.write_text(src.split("#include <torch/extension.h>")[0], encoding="utf-8")
    cub = pathlib.Path(td) / f"{nm}.cubin"
    r = subprocess.run(["nvcc", "-O3", "-DVMS_G=132"] + defs + ["-Xptxas=-v,-warn-spills", "-gencode=arch=compute_90a,code=sm_90a",
                        "-cubin", str(k), "-o", str(cub)], capture_output=True, text=True)
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if "registers" in ln or "spill" in ln or "error" in ln]
    if r.returncode:
        return r.returncode, lines[-4:], None
    sass = subprocess.run([CUOBJDUMP, "-sass", str(cub)], capture_output=True, text=True).stdout
    body = "\n".join(ln for ln in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln))
    return 0, lines[-2:], body


def fences(body):
    out = []
    for ln in body.splitlines():
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?);", ln)
        if m and re.search(r"MEMBAR|FENCE|ERRBAR|CCTL|UBLKCP|SYNCS\.ARRIVE|RED\.|ATOM|LDG\.E\.STRONG|\.STRONG\.GPU", m.group(2)):
            out.append(f"{m.group(1)} {m.group(2).strip()[:60]}")
    return out


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    src0, cpp = layer_src(LAYER)
    src = P.apply(src0)
    ok_all = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            rc0, ln0, b0 = sm90(src0, SHIP, td, "orig")
            rc1, ln1, b1 = sm90(src, SHIP, td, "def0")
            same = rc0 == 0 and rc1 == 0 and b0 == b1
            ok_all &= same
            print(f"[vfn] sm_90a base source: {' | '.join(ln0)}", flush=True)
            print(f"[vfn] sm_90a patched at defaults: SASS {'IDENTICAL' if same else 'DIFFERS'} to the base source", flush=True)
            if int(opt.get("fences", 1)) and b0:
                print("[vfn] base SASS fence / atomic / bulk lines:\n   " + "\n   ".join(fences(b0)), flush=True)
            for nm, d in P.ARMS.items():
                rc, ln, b = sm90(src, merge(SHIP, d.split()), td, nm)
                spill = [x for x in ln if "spill" in x]
                ok = rc == 0 and all(" 0 bytes spill stores" in x and " 0 bytes spill loads" in x for x in spill)
                print(f"[vfn] sm_90a {nm:7s} ({d}): rc {rc} {'ok' if ok else 'CHECK'} | {' | '.join(ln)}", flush=True)
                if nm in ("db1", "fx2db1") and b and int(opt.get("fences", 1)):
                    print(f"[vfn] {nm} SASS fence / atomic / bulk lines:\n   " + "\n   ".join(fences(b)), flush=True)
                ok_all &= ok
    if not int(opt.get("sm86", 1)):
        print(f"[vfn] {'ALL PASS' if ok_all else 'FAILURES'}")
        return 0 if ok_all else 1
    import torch
    import vms_moe_lab_v1 as V
    from vks_local_test import layer_pos
    from vms_local_test import make_weights, make_input
    from torch.utils.cpp_extension import load_inline
    V.POS = layer_pos(LAYER)
    G = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"

    def build(s, defs, name):
        fl = ["-O3", f"-DVMS_G={G}"] + merge(["-DVMS_RO=1"], defs) + ["-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
        tag = hashlib.sha1((s + " ".join(fl)).encode()).hexdigest()[:10]
        return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=s, functions=["run", "attrs"], extra_cuda_cflags=fl)
    m0 = build(src0, ["-DVMS_NS=1"], "vfn_l0")
    print(f"[vfn] sm_86 base attrs {list(m0.attrs())}", flush=True)
    mods = {nm: build(src, d.split(), f"vfn_{nm}") for nm, d in SM86.items()}
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    sc0 = V.scratch(dev)
    cases = [(1, 1, "normal"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"), (9, 21, "hot"), (11, 23, "skew"),
             (16, 9, "normal"), (16, 10, "hot"), (3, 12, "nolocal"), (15, 25, "skew")]

    def chk(o, ref, ob, sc):
        tot, row = V.rel_err(o, ref)
        tb, _ = V.rel_err(o, ob)
        c = sc["ctr"]
        err, coh = int(c[2]), int(c[1]) == int(c[0]) * G
        good = err == 0 and coh and tot < 6e-3 and row < 3e-2 and tb < 2e-3 and bool(torch.isfinite(o.float()).all())
        return good, f"rel {tot:.3e} row {row:.3e} vs base {tb:.1e} err {err} bar {'ok' if coh else 'BAD'}"

    marks = torch.zeros((G * 16,), dtype=torch.int64, device=dev)
    for nm, m1 in mods.items():
        print(f"[vfn] sm_86 {nm} attrs {list(m1.attrs())}", flush=True)
        sc1 = V.scratch(dev)
        for T, seed, kind in cases:
            x, ids, tw = make_input(T, seed, dev, kind)
            ref = V.ref_fp32(x, ids, tw, w)
            ob = V.call(m0, pk, sc0, x, ids, tw)
            for k in range(2):
                marks.zero_()
                o = V.call(m1, pk, sc1, x, ids, tw, marks=marks if k else None)
                torch.cuda.synchronize()
                good, msg = chk(o, ref, ob, sc1)
                ok_all &= good
                if k == 0 and (not good or T in (1, 8, 16)):
                    print(f"[vfn] {nm} T={T:2d} {kind:7s}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
            if T == 8 and "FM" in SM86[nm]:
                r = marks.view(G, 16).cpu()
                nz = [int((r[:, i] > 0).sum()) for i in (6, 10, 11, 13)]
                print(f"[vfn] {nm} T=8 marks set (CTAs) mrow6/10/11/13: {nz}", flush=True)
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
                print(f"[vfn] {nm} graph replay {rr}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
    print(f"[vfn] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

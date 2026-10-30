"""Local test of VMS_SH (tools/vsh_patches.py) on the base layer's VMS source (k3_b23_1 == k3_gc_21cg_s1 VMS).
  sm90=1: sm_90a compile-only (WSL nvcc 13.4): registers / spills, patched-at-defaults SASS == base SASS, LL fence lines.
  sm86=1: RTX 3070 (cp.async ring, NS <= 2): LL vs the FP32 reference and the base build on 12 routings x 3 calls, the
          generation counter (ctr[0] +1 per call, ctr[3] back to 0, err 0), and 6 changed-input CUDA-graph replays per arm
          (a stale flagged word from the previous replay would fail the new reference).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vll_local_test.py" [sm90=1] [sm86=1]
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
import vsh_patches as P0  # noqa: E402
import vaq_patches as PQ  # noqa: E402


class P:
    layer_src = staticmethod(P0.layer_src)

    @staticmethod
    def apply(s):
        return PQ.apply(P0.apply(s))

BASEF = pathlib.Path(__file__).resolve().parents[1] / "kernels/lab/k3_b23_1.py"

CUOBJDUMP = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
SHIP = ["-DVMS_NS=3", "-DVMS_RO=1"]
SM86 = {"aq": "-DVMS_NS=2 -DVMS_AQ=1", "shaq": "-DVMS_NS=2 -DVMS_SH=4 -DVMS_AQ=1"}


def sm90(src, defs, td, nm):
    k = pathlib.Path(td) / f"{nm}.cu"
    k.write_text(src.split("#include <torch/extension.h>")[0], encoding="utf-8")
    cub = pathlib.Path(td) / f"{nm}.cubin"
    r = subprocess.run(["nvcc", "-O3", "-DVMS_G=132"] + defs + ["-Xptxas=-v,-warn-spills", "-gencode=arch=compute_90a,code=sm_90a",
                        "-cubin", str(k), "-o", str(cub)], capture_output=True, text=True)
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if "registers" in ln or "spill" in ln or "error" in ln]
    if r.returncode:
        return r.returncode, (r.stdout + r.stderr).splitlines()[-12:], None
    sass = subprocess.run([CUOBJDUMP, "-sass", str(cub)], capture_output=True, text=True).stdout
    body = "\n".join(ln for ln in sass.splitlines() if re.match(r"\s+/\*[0-9a-f]{4,}\*/", ln))
    return 0, lines[-2:], body


def count(body, pat):
    return sum(1 for ln in body.splitlines() if re.search(pat, ln))


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    src0, cpp = P.layer_src(BASEF)
    src = P.apply(src0)
    ok_all = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            rc0, ln0, b0 = sm90(src0, SHIP, td, "orig")
            rc1, ln1, b1 = sm90(src, SHIP, td, "def0")
            same = rc0 == 0 and rc1 == 0 and b0 == b1
            ok_all &= same
            print(f"[vaq] sm_90a base: {' | '.join(ln0)}", flush=True)
            print(f"[vaq] sm_90a patched at defaults: SASS {'IDENTICAL' if same else 'DIFFERS'} to the base", flush=True)
            rc, ln, b = sm90(src, SHIP + ["-DVMS_SH=4"], td, "sh")
            rch, lnh, _ = sm90(src, SHIP + ["-DVMS_SH=4", "-DVMS_FX=1"], td, "shfx")
            print(f"[vaq] sm_90a SH4+FX1: rc {rch} | {' | '.join(lnh)}", flush=True)
            spill = [x for x in ln if "spill" in x]
            ok = rc == 0 and all(" 0 bytes spill stores" in x and " 0 bytes spill loads" in x for x in spill)
            ok_all &= ok
            print(f"[vaq] sm_90a LL: rc {rc} {'ok' if ok else 'CHECK'} | {' | '.join(ln)}", flush=True)
            if b is not None:
                for nm_, bb in (("base", b0), ("LL", b)):
                    print(f"[vaq]   {nm_}: MEMBAR {count(bb, 'MEMBAR')} RED/ATOM {count(bb, r'RED\.|ATOM')} "
                          f"STG.E.64.STRONG {count(bb, r'STG\.E\.64\.STRONG')} LDG.STRONG {count(bb, r'LDG\.E.*STRONG')}", flush=True)
    if not int(opt.get("sm86", 1)):
        print(f"[vaq] {'ALL PASS' if ok_all else 'FAILURES'}")
        return 0 if ok_all else 1
    import torch
    import vms_moe_lab_v1 as V
    from vks_local_test import layer_pos
    from vms_local_test import make_weights, make_input
    from torch.utils.cpp_extension import load_inline
    V.POS = layer_pos("kernels/lab/k3_b23_1.py")
    G = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"

    def build(s, defs, name):
        fl = ["-O3", f"-DVMS_G={G}"] + ["-DVMS_RO=1"] + defs + ["-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
        tag = hashlib.sha1((s + " ".join(fl)).encode()).hexdigest()[:10]
        return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=s, functions=["run", "attrs"], extra_cuda_cflags=fl)
    m0 = build(src0, ["-DVMS_NS=1"], "vll_b0")
    print(f"[vaq] sm_86 base attrs {list(m0.attrs())}", flush=True)
    w = make_weights(dev)
    pk = V.prepack(w, log=lambda s: None)
    sc0 = V.scratch(dev)

    def scratch_ll():
        return V.scratch(dev)
    cases = [(1, 1, "normal"), (1, 2, "normal"), (1, 4, "skew"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"), (8, 17, "normal"), (9, 21, "hot"),
             (11, 23, "skew"), (16, 9, "normal"), (16, 10, "hot"), (3, 12, "nolocal"), (15, 25, "skew"), (1, 31, "nolocal")]

    def chk(o, ref, ob, sc, ncalls, lly=True):
        tot, row = V.rel_err(o, ref)
        tb, _ = V.rel_err(o, ob)
        c = sc["ctr"].cpu()
        err = int(c[2])
        coh = int(c[0]) == ncalls and int(c[1]) == ncalls * G
        good = err == 0 and coh and tot < 6e-3 and row < 3e-2 and tb < 2e-3 and bool(torch.isfinite(o.float()).all())
        return good, f"rel {tot:.3e} row {row:.3e} vs base {tb:.1e} err {err} gen {int(c[0])}/{ncalls} c3 {int(c[3])}"

    for nm, d in SM86.items():
        m1 = build(src, d.split(), f"vll_{nm}")
        print(f"[vaq] sm_86 {nm} attrs {list(m1.attrs())}", flush=True)
        sc1 = scratch_ll()
        ncalls = 0
        for T, seed, kind in cases:
            x, ids, tw = make_input(T, seed, dev, kind)
            ref = V.ref_fp32(x, ids, tw, w)
            ob = V.call(m0, pk, sc0, x, ids, tw)
            for k in range(3):
                o = V.call(m1, pk, sc1, x, ids, tw)
                torch.cuda.synchronize()
                ncalls += 1
                good, msg = chk(o, ref, ob, sc1, ncalls, "LL=3" in d)
                ok_all &= good
                if k == 0 and (not good or T in (1, 8, 16)):
                    print(f"[vaq] {nm} T={T:2d} {kind:7s}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
                elif not good:
                    print(f"[vaq] {nm} T={T:2d} {kind:7s} call {k}: {msg} -> FAIL", flush=True)
        for T in (1, 8, 16):
            x, ids, tw = make_input(T, 108 + T, dev, "hot")
            out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
            V.call(m1, pk, sc1, x, ids, tw, out=out)
            torch.cuda.synchronize()
            ncalls += 1
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                V.call(m1, pk, sc1, x, ids, tw, out=out)
            for rr in range(6):
                x2, ids2, tw2 = make_input(T, 300 + 10 * T + rr, dev, ("normal", "hot", "skew")[rr % 3])
                x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
                out.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                ncalls += 1
                ref = V.ref_fp32(x2, ids2, tw2, w)
                ob = V.call(m0, pk, sc0, x2, ids2, tw2)
                torch.cuda.synchronize()
                good, msg = chk(out, ref, ob, sc1, ncalls, "LL=3" in d)
                ok_all &= good
                if not good or rr == 0:
                    print(f"[vaq] {nm} T={T} graph replay {rr}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
    print(f"[vaq] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

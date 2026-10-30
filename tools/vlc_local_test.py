"""Local test of VMS_LC (tools/vlc_patches.py) on mb_25's layer-VMS source (MB_LV_CUDA, all earlier patches at their defaults).
  sm90=1: sm_90a compile-only (WSL nvcc 13.4): patched-at-defaults SASS == base SASS; LC 1 (and LC 1 + FX 1) regs / spills.
  sm86=1: RTX 3070 (cp.async ring, NS 2): LC vs the base build on 14 routings x 3 calls (FP32-reference and vs-base tolerances as R-vaq: the ship's ss atomics are order-nondeterministic; err 0, generation
          counter ctr[0] == calls and arrivals ctr[1] == calls x G) and 6 changed-input CUDA-graph replays at T 1 / 8 / 16.
    MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
        python "<repo>/tools/vlc_local_test.py" [sm90=1] [sm86=1] [src=mb_25] [patch=vlc] [arms=LC=1,LC=1+FX=1]
  (sm86 runs the first arm; this is a local test tool, never part of a submission)
"""
import hashlib
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vaq_local_test as T  # noqa: E402
import importlib.util  # noqa: E402
import vsh_patches as P0  # noqa: E402

SHIP = ["-DVMS_NS=3", "-DVMS_RO=1"]


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    s = (ROOT / f"kernels/muse_moe_w4a16/{opt.get('src', 'mb_25')}.py").read_text(encoding="utf-8")
    q = "MB_LV_CUDA = r'''"
    a = s.index(q) + len(q)
    src0 = s[a:s.index("'''", a)]
    spec = importlib.util.spec_from_file_location("vpatch", ROOT / f"tools/{opt.get('patch', 'vlc')}_patches.py")   # vlc | vhp | ...
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    src = P.apply(src0)
    arms = [a.split('+') for a in opt.get('arms', 'LC=1,LC=1+FX=1').split(',')]   # each arm: '+'-joined NAME=VAL defines
    _, cpp = P0.layer_src(ROOT / "kernels/lab/k3_b23_1.py")
    ok = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            rc0, ln0, b0 = T.sm90(src0, SHIP, td, "orig")
            rc1, ln1, b1 = T.sm90(src, SHIP, td, "def0")
            same = rc0 == 0 and rc1 == 0 and b0 == b1
            ok &= same
            print(f"[vlc] sm_90a base: {' | '.join(ln0)}", flush=True)
            print(f"[vlc] sm_90a patched at defaults: SASS {'IDENTICAL' if same else 'DIFFERS'}", flush=True)
            for d in ([f"-DVMS_{x}" for x in a] for a in arms):
                nm = "".join(x[6:].replace("=", "") for x in d).lower()
                rc, ln, b = T.sm90(src, SHIP + d, td, nm)
                spill = [x for x in ln if "spill" in x]
                good = rc == 0 and all(" 0 bytes spill stores" in x and " 0 bytes spill loads" in x for x in spill)
                ok &= good
                print(f"[vlc] sm_90a {nm}: rc {rc} {'ok' if good else 'CHECK'} | {' | '.join(ln)}", flush=True)
    if int(opt.get("sm86", 1)):
        import torch
        import vms_moe_lab_v1 as V
        from torch.utils.cpp_extension import load_inline
        from vks_local_test import layer_pos
        from vms_local_test import make_input, make_weights
        V.POS = layer_pos("kernels/lab/k3_b23_1.py")
        G = torch.cuda.get_device_properties(0).multi_processor_count
        dev = "cuda"

        def build(sx, defs, name):
            fl = ["-O3", f"-DVMS_G={G}", "-DVMS_RO=1", "-DVMS_NS=2"] + defs + ["-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
            tag = hashlib.sha1((sx + " ".join(fl)).encode()).hexdigest()[:10]
            return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=sx, functions=["run", "attrs"], extra_cuda_cflags=fl)
        m0 = build(src0, [], "vlc_b0")
        m1 = build(src, [f"-DVMS_{x}" for x in arms[0]], "vlt_" + "".join(arms[0]).replace("=", "").lower())
        print(f"[vlc] sm_86 attrs base {list(m0.attrs())} lc1 {list(m1.attrs())}", flush=True)
        w = make_weights(dev)
        pk = V.prepack(w, log=lambda s_: None)
        sc0, sc1 = V.scratch(dev), V.scratch(dev)
        cases = [(1, 1, "normal"), (1, 2, "normal"), (1, 4, "skew"), (1, 31, "nolocal"), (2, 3, "normal"), (5, 5, "normal"), (8, 7, "hot"),
                 (8, 17, "normal"), (9, 21, "hot"), (11, 23, "skew"), (16, 9, "normal"), (16, 10, "hot"), (3, 12, "nolocal"), (1, 44, "hot")]
        n1 = 0

        def chk(o, ob, sc, ncalls, ref):   # ss is an atomicAdd float sum: the ship itself is not bitwise repeatable
            c = sc["ctr"].cpu()
            err, coh = int(c[2]), int(c[0]) == ncalls and int(c[1]) == ncalls * G
            tot, row = V.rel_err(o, ref)
            tb, _ = V.rel_err(o, ob)
            good = err == 0 and coh and tot < 6e-3 and row < 3e-2 and tb < 2e-3 and bool(torch.isfinite(o.float()).all())
            return good, (f"rel {tot:.3e} row {row:.3e} vs base {tb:.1e} eq {bool(torch.equal(o, ob))} err {err} gen {int(c[0])}/{ncalls} "
                          f"arrivals {int(c[1])}/{ncalls * G}")
        for Tk, seed, kind in cases:
            x, ids, tw = make_input(Tk, seed, dev, kind)
            ref = V.ref_fp32(x, ids, tw, w)
            ob = V.call(m0, pk, sc0, x, ids, tw)
            ob2 = V.call(m0, pk, sc0, x, ids, tw)
            bt, _ = V.rel_err(ob2, ob)
            print(f"[vlc] T={Tk:2d} {kind:7s} base vs base: {bt:.1e} eq {bool(torch.equal(ob2, ob))}", flush=True)
            for k in range(3):
                o = V.call(m1, pk, sc1, x, ids, tw)
                torch.cuda.synchronize()
                n1 += 1
                good, msg = chk(o, ob, sc1, n1, ref)
                ok &= good
                if k == 0 or not good:
                    print(f"[vlc] T={Tk:2d} {kind:7s} call {k}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
        for Tk in (1, 8, 16):
            x, ids, tw = make_input(Tk, 108 + Tk, dev, "hot")
            out = torch.empty((Tk, 4096), dtype=torch.bfloat16, device=dev)
            V.call(m1, pk, sc1, x, ids, tw, out=out)
            torch.cuda.synchronize()
            n1 += 1
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                V.call(m1, pk, sc1, x, ids, tw, out=out)
            for rr in range(6):
                x2, ids2, tw2 = make_input(Tk, 300 + 10 * Tk + rr, dev, ("normal", "hot", "skew")[rr % 3])
                x.copy_(x2)
                ids.copy_(ids2)
                tw.copy_(tw2)
                out.fill_(float("nan"))
                gr.replay()
                torch.cuda.synchronize()
                n1 += 1
                ob = V.call(m0, pk, sc0, x2, ids2, tw2)
                torch.cuda.synchronize()
                good, msg = chk(out, ob, sc1, n1, V.ref_fp32(x2, ids2, tw2, w))
                ok &= good
                if rr == 0 or not good:
                    print(f"[vlc] T={Tk} graph replay {rr}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
    print(f"[vlc] {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

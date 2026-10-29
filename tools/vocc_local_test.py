"""Local test of VOCC builds (tools/vocc_patches.py) on the layer's VMS source.
  sm90=1: sm_90a compile-only (WSL nvcc 13.4): ptxas registers / spills / barriers per variant, SASS opcode counts of the
          consumer loop, and the OCC 0 identity check (patched source at the defaults -> same SASS as the base source).
  sm86=1: RTX 3070 (sm_86, cp.async fallback ring) vs the FP32 reference and the base build (tolerance: VMS is not bitwise
          run to run), 10 routings x 2 calls + 3 changed-input CUDA-graph replays per variant.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vocc_local_test.py" [vars="k16:-DVMS_OCC=1;u2:-DVMS_OCC=2 -DVMS_NS=4"] [sm90=1] [sm86=1]
"""
import collections
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
import vocc_patches as P  # noqa: E402

CUOBJDUMP = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"
SHIP = ["-DVMS_RO=1"]


def layer_src(path):
    L = (ROOT / path).read_text(encoding="utf-8")
    a = L.index('VMS_CUDA_SRC = r"""') + len('VMS_CUDA_SRC = r"""')
    c = L.index('VMS_CPP_SRC = r"""') + len('VMS_CPP_SRC = r"""')
    return L[a:L.index('"""', a)], L[c:L.index('"""', c)]


def flags_for(defs, ns_default):
    d = list(SHIP) + list(defs)
    if not any(x.startswith("-DVMS_NS=") for x in d):
        d = [f"-DVMS_NS={ns_default}"] + d
    return d


def sass_ops(sass):
    """consumer-loop opcode counts: the loop that holds the HMMAs (back branch target <= first HMMA < back branch)"""
    ins = []
    for ln in sass.splitlines():
        m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?\s*([^;]*);", ln)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(5)))
    hm = [a for a, op, _ in ins if op == "HMMA"]
    if not hm:
        return None
    best = None
    for a, op, arg in ins:
        if op == "BRA" and a > hm[-1]:
            t = re.search(r"0x([0-9a-f]+)", arg)
            if t and int(t.group(1), 16) <= hm[0]:
                best = (int(t.group(1), 16), a)
                break
    if not best:
        return None
    c = collections.Counter(op for a, op, _ in ins if best[0] <= a <= best[1])
    return sum(c.values()), dict(c.most_common(8))


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


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    var = [v.split(":", 1) for v in opt.get("vars", "k16:-DVMS_OCC=1;u2:-DVMS_OCC=2 -DVMS_NS=4").split(";")]
    lay = opt.get("layer", "kernels/muse_decode_layer/k3_gc_20.py")
    src0, cpp = layer_src(lay)
    src = P.apply(src0)
    ok_all = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            rc0, ln0, b0 = sm90(src0, flags_for([], 3), td, "orig")
            rc1, ln1, b1 = sm90(src, flags_for([], 3), td, "occ0")
            same = rc0 == 0 and rc1 == 0 and b0 == b1
            ok_all &= same
            print(f"[vocc] sm_90a base source: {' | '.join(ln0)} | loop {sass_ops(b0) if b0 else None}", flush=True)
            print(f"[vocc] sm_90a patched OCC 0: SASS {'IDENTICAL' if same else 'DIFFERS'} to the base source", flush=True)
            for nm, d in var:
                rc, ln, b = sm90(src, flags_for(d.split(), 3), td, nm)
                print(f"[vocc] sm_90a {nm} ({d}): rc {rc} | {' | '.join(ln)} | loop {sass_ops(b) if b else None}", flush=True)
                ok_all &= rc == 0
    if not int(opt.get("sm86", 1)):
        print(f"[vocc] {'ALL PASS' if ok_all else 'FAILURES'}")
        return 0 if ok_all else 1
    import torch
    import vms_moe_lab_v1 as V
    from vks_local_test import layer_pos
    from vms_local_test import make_weights, make_input
    from torch.utils.cpp_extension import load_inline
    V.POS = layer_pos(lay)
    G = torch.cuda.get_device_properties(0).multi_processor_count
    dev = "cuda"

    def build(s, defs, name):
        fl = ["-O3", f"-DVMS_G={G}"] + flags_for(defs, 1) + ["-Xptxas=-warn-spills", "-gencode=arch=compute_86,code=sm_86"]
        tag = hashlib.sha1((s + " ".join(fl)).encode()).hexdigest()[:10]
        return load_inline(name=f"{name}_{tag}", cpp_sources=cpp, cuda_sources=s, functions=["run", "attrs"], extra_cuda_cflags=fl)
    m0 = build(src0, [], "vocc_l0")
    print(f"[vocc] sm_86 base attrs {list(m0.attrs())}", flush=True)
    mods = {nm: build(src, [x.replace("-DVMS_NS=4", "-DVMS_NS=2") for x in d.split()], f"vocc_{nm}") for nm, d in var}
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

    for nm, m1 in mods.items():
        print(f"[vocc] sm_86 {nm} attrs {list(m1.attrs())}", flush=True)
        sc1 = V.scratch(dev)
        for T, seed, kind in cases:
            x, ids, tw = make_input(T, seed, dev, kind)
            ref = V.ref_fp32(x, ids, tw, w)
            ob = V.call(m0, pk, sc0, x, ids, tw)
            for k in range(2):
                o = V.call(m1, pk, sc1, x, ids, tw)
                torch.cuda.synchronize()
                good, msg = chk(o, ref, ob, sc1)
                ok_all &= good
                if k == 0 and (not good or T in (1, 8, 16)):
                    print(f"[vocc] {nm} T={T:2d} {kind:7s}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
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
                print(f"[vocc] {nm} graph replay {rr}: {msg} -> {'ok' if good else 'FAIL'}", flush=True)
    print(f"[vocc] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

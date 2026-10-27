"""vd_ver_ab.py: same-process A/B of VD kernel source versions (R-vd-4): r2 = the exact R-vd-2 source, a = r2 + K rotation
(commit 601a46a), c3 = a + dynamic intra-cluster split (= R-vd-3, commit 33df3a1), cur = c3 with the entry fence only
under the dynamic flag (kernels/lab/vd_lab.py). R-vd-3 measured cur 1.5-2.9 us slower than
R-vd-2 on the same configs (another pod) while W-only and cuBLAS did not move: code or pod? Every arm of a shape is timed in
the same interleaved rounds (cold L2, graph replay), so the answer does not depend on the pod.

    python3 kernels/lab/vd_ver_ab.py [--projs=pooler,qkv,cat,unpooler,o] [--rounds=8] [--reps=21]
"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SUB = [(1, 128, 8, 2, 0, 1), (1, 128, 8, 3, 0, 1), (2, 128, 6, 2, 0, 1), (2, 128, 5, 3, 1, 1), (1, 128, 9, 1, 0, 0)]
CFG = {"pooler": (0, 4096, 8192), "qkv": (1, 2304, 8192), "unpooler": (2, 8192, 4096), "cat": (3, 4608, 8192), "o": (4, 8192, 1024)}


def load(tag, path):
    spec = importlib.util.spec_from_file_location(f"vdv_{tag}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    L = load("cur", ROOT / "kernels/lab/vd_lab.py")
    arg = L.arg
    projs = arg("projs", "pooler,qkv,cat,unpooler,o").split(",")
    rounds, reps = int(arg("rounds", "8")), int(arg("reps", "21"))
    out = Path(arg("out", str(ROOT / "box_out" / "vd4" / "ver.json")))
    vers = {"r2": load("r2", ROOT / "kernels/lab/vd_lab_r2.py"), "a": load("a", ROOT / "kernels/lab/vd_lab_a.py"),
            "c3": load("c3", ROOT / "kernels/lab/vd_lab_c3.py"), "cur": L}
    mods = {k: m.vd_build(SUB, name=f"vdv_{k}") for k, m in vers.items()}
    dev = torch.device("cuda")
    torch.manual_seed(77)
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    err = torch.zeros((4,), dtype=torch.int32, device=dev)
    stamps = torch.zeros((4 * 512,), dtype=torch.int64, device=dev)
    gnull = L.graph_of(lambda: mods["cur"].vd_run(-1, err, err, err, err, err, err, 1, 0, 0, 0, 0, 0))
    res = {}
    for proj in projs:
        kid, n, k = CFG[proj]
        nwg, bn, nst, cl, f32, rbd = SUB[kid]
        bm = 64 * nwg
        w = (torch.randn((n, k), device=dev) * 0.02).to(torch.bfloat16)
        flat, w_rb, w_s = L.vd_pack(w, bm, "front")
        x = torch.randn((128, k), device=dev).to(torch.bfloat16)
        ref = x.double() @ w.double().t()
        odt = torch.float32 if f32 else torch.bfloat16
        arms = {"cublas": (lambda x=x: torch.mm(x, w.t(), out_dtype=torch.float32)) if f32 else (lambda x=x: torch.mm(x, w.t()))}
        ys = {}
        plan = [(v, 8, 0) for v in ("r2", "a", "c3", "cur")] + [("cur", 8, 2), ("r2", 8, 2)]
        if cl > 1:
            plan += [("cur", 8 | 512, 0), ("cur", 8 | 512, 2)]
        for v, fl, pf in plan:
            y = torch.zeros((128, n), dtype=odt, device=dev)
            nm = f"{v}_f{fl}_p{pf}"

            def run(v=v, y=y, fl=fl, pf=pf):
                mods[v].vd_run(kid, x, flat, w, y, err, stamps, n // bm, w_rb, w_s, fl, 0, pf)
            arms[nm] = run
            ys[nm] = y
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        scale = float(ref.abs().max())
        rel = {nm: float((y.double() - ref).abs().max()) / scale for nm, y in ys.items()}
        print(f"[ver] {proj}: err {int(err[0])}; rel " + " ".join(f"{a}={b:.2e}" for a, b in rel.items()), flush=True)
        if int(err[0]):
            err.zero_()
            continue
        names = list(arms)
        graphs = [L.graph_of(arms[nm]) for nm in names] + [gnull]
        meds = {nm: [] for nm in names + ["null"]}
        for _ in range(rounds):
            for nm, t in zip(names + ["null"], L.time_many(graphs, flush, sink, reps)):
                meds[nm].append(t)
        fl0 = statistics.median(meds["null"])
        cub = statistics.median(meds["cublas"]) - fl0
        res[proj] = {nm: statistics.median(meds[nm]) - fl0 for nm in names}
        print(f"[ver] {proj} m128 net us (x vs cuBLAS {cub:.2f}): " + " | ".join(
            f"{nm} {res[proj][nm]:.2f} x{cub / res[proj][nm]:.3f}" for nm in names if nm != "cublas"), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    print("[ver] done", flush=True)


if __name__ == "__main__":
    main()

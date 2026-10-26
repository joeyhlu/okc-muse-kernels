"""K1 v24 lab (box only): bitwise comparison of the shipped K1 v7 build and a candidate build on judge inputs (output
and the new cache rows), one fresh input per call so both start from the same cache.
  python3 kernels/muse_attention_decode/k1_v24_bitwise.py [--cand sectpair] [--shapes 5,3,1,4]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import k1_v24_lab as LAB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cand", default="sectpair")
ap.add_argument("--shapes", default="5,3,1,4")
args = ap.parse_args()
ns = LAB.load_ns()
mods = {nm: LAB.build(ns, nm, LAB.BUILDS[nm], LAB.lab_cuda(ns, LAB.LAB_PATCHES + LAB.EXTRA.get(nm, [])))
        for nm in ("ship", args.cand)}
benches = json.loads((LAB.PROB / "meta.json").read_text())["benchmarks"]
for i in [int(x) for x in args.shapes.split(",")]:
    kw = {k: v for k, v in benches[i].items() if k not in ("load", "weight")}
    outs = {}
    for nm in mods:
        d = LAB.reference.generate_input(**kw)
        ns["_MOD"] = mods[nm]
        ns["_OCC"].clear()
        o = ns["_k1"](d)
        torch.cuda.synchronize()
        rows = d["cache_loc"]
        hkv = d["config"]["num_kv_heads"]
        outs[nm] = (o.clone(), d["k_cache"].view(-1, hkv * 64).view(torch.uint8)[rows].clone(),
                    d["v_cache"].view(-1, hkv * 64).view(torch.uint8)[rows].clone())
        del d
    a, b = outs["ship"], outs[args.cand]
    eq = [torch.equal(x, y) for x, y in zip(a, b)]
    print(f"BITWISE [{i:2d}] {benches[i]['load']} {benches[i]['layer']}: out {eq[0]} k {eq[1]} v {eq[2]}"
          f" (max |d out| {float((a[0].float() - b[0].float()).abs().max()):.3e})", flush=True)

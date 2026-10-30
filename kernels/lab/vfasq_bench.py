"""vfasq_bench.py (R-vfasq-1): standalone VFA (vfa_main, verify q_len 8, full attention) ship vs split-QK (tools/vfasq_patches.py).

Builds the layer file's VFA source alone (no layer import, no prepare, no FA4 prologue: q8 = e4m3(rms(q)) made here) with the
VFASQ lab arms on and VFA_SQ_ON = false, so in one extension:
  d0  ship consumer, real gather            d21 split QK, real gather
  d22 ship consumer, nocopy producer        d23 split QK, nocopy (the producer copies only the first ST tiles, then only
                                                arrives: tiles already resident, consumer-only time at the real tile counts)
Shapes: the attention board's verify 16 / 32 full inputs (generate_input, seeds 610 / 612, page 1) + small correctness cases
(page 64, kv scale 1/64, short contexts, B 33). Correctness: d21 vs d0 rel (and d0 vs a d0 repeat: the merge is FP32 red.add,
not order-deterministic), finite, workspace clean. Timing: one CUDA graph per arm, judge-style (256 MB L2 flush read + 300 us
gate, tools/judge GpuTimer), rotating arm order, median of --reps; ship_b / nc_ship_b are middle controls (same kernel).

  box:   python3 -u kernels/lab/vfasq_bench.py --layer=kernels/lab/k3_vfasq_1.py --out=box_out/vfasq/bench.json
  local: python kernels/lab/vfasq_bench.py --layer=kernels/lab/k3_vfasq_1.py --dry     (inputs + host compile only)
"""
import argparse
import ast
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/judge"))
import judge as J  # noqa: E402

FP8 = torch.float8_e4m3fn
ARMS = (("ship", 0), ("sq", 21), ("ship_b", 0), ("sq_b", 21), ("nc_ship", 22), ("nc_sq", 23), ("nc_ship_b", 22))
SHAPES = ((16, 610), (32, 612))
CASES = ((2, 3000, 1, 1.0, 401), (3, 1500, 64, 1.0, 402), (16, 700, 1, 1.0 / 64, 403), (33, 300, 1, 1.0, 405),
         (2, 20, 1, 1.0, 407), (17, 100, 1, 1.0, 408), (4, 70, 64, 1.0, 409))


def sources(layer):
    vals = {}
    for n in ast.parse(Path(layer).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("VFA_CPP_SRC", "VFA_CUDA_SRC", "VFA_SEGK"):
            vals[n.targets[0].id] = ast.literal_eval(n.value)
    cu = vals["VFA_CUDA_SRC"]
    assert "#if VFA_SQLAB" in cu and "VFA_SQ_ON" in cu, "not a VFASQ lab file (tools/vfasq_patches.py)"
    cu = cu.replace("#if VFA_LAB", "#if 0").replace("#if VFA_SQLAB", "#if 1").replace("VFA_SQ_ON", "false")
    return vals["VFA_CPP_SRC"], cu, int(vals.get("VFA_SEGK", 300))


def build(cpp, cu):
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((cpp + cu).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vfasq_{tag}", cpp_sources=cpp, cuda_sources=cu, functions=["vfa_forward", "vfa_occupancy"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"[vfasq] build {time.time() - t0:.0f}s", flush=True)
    return mod


def inputs(aref, B, ctx, page, kvs, seed, dev, board):
    if board:
        d = aref.generate_input(batch=B, ctx=80000, heads="tp8", layer="full", seed=seed, q_len=8, spread="agentic", dist="coherent",
                                page=1, device=dev)
    else:
        d = aref.generate_input(batch=B, ctx=ctx, heads="tp8", layer="full", seed=seed, q_len=8, spread="fixed", dist="normal",
                                page=page, kv_scale=kvs, device=dev)
    cfg = d["config"]
    hq = cfg["num_q_heads"]
    qkv = d["qkv"].reshape(B * 8, -1).contiguous()
    q = aref._rms(qkv[:, :hq * 64].reshape(B * 8, hq, 64)).to(torch.bfloat16).float().clamp(-448.0, 448.0)
    d["q8"] = q.to(FP8).view(torch.uint8).reshape(B * 8, hq * 64).contiguous()
    d["qkv2"] = qkv
    return d


class Run:
    def __init__(self, mod, sms, segk, dev):
        self.mod, self.G, self.segk, self.dev = mod, 2 * sms, segk, dev

    def ws(self):
        rows = 256 * 128
        return (torch.zeros((rows, 64), dtype=torch.float32, device=self.dev), torch.zeros(rows, dtype=torch.float32, device=self.dev),
                torch.zeros(256, dtype=torch.int32, device=self.dev))

    def __call__(self, d, diag, ws, out):
        cfg = d["config"]
        B, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
        self.mod.vfa_forward(d["q8"], d["qkv2"], d["seq_lens"], d["page_table"], d["k_cache"].view(torch.uint8),
                             d["v_cache"].view(torch.uint8), d["k_scale"], d["v_scale"], ws[0], ws[1], ws[2], out, B, hq, hkv,
                             cfg["page_size"].bit_length() - 1, self.G, self.segk, 0,
                             float(cfg["qk_scale_factor"] * cfg["sm_scale"]), 4, diag)
        return out


def rel(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a - b).norm() / b.norm().clamp(min=1e-30))


def clean(ws):
    return not (int(ws[2].abs().sum()) or float(ws[0].abs().sum()) or float(ws[1].abs().sum()))


def check(run, d, tag):
    cfg = d["config"]
    B, hq = cfg["batch"], cfg["num_q_heads"]
    outs, ok = {}, True
    for name, dg in (("ship", 0), ("ship_rep", 0), ("sq", 21), ("nc_ship", 22), ("nc_sq", 23)):
        ws = run.ws()
        o = torch.empty((B * 8, hq * 64), dtype=torch.bfloat16, device=run.dev)
        run(d, dg, ws, o)
        torch.cuda.synchronize()
        outs[name] = o
        if not clean(ws):
            print(f"[vfasq] {tag} {name}: workspace DIRTY", flush=True)
            ok = False
    r_rep, r_sq = rel(outs["ship_rep"], outs["ship"]), rel(outs["sq"], outs["ship"])
    eq = bool(torch.equal(outs["sq"], outs["ship"]))
    fin = bool(torch.isfinite(outs["sq"].float()).all()) and bool(torch.isfinite(outs["ship"].float()).all())
    good = ok and fin and r_sq <= max(2e-3, 3 * r_rep)
    print(f"[vfasq] {tag}: sq vs ship rel {r_sq:.2e} (bitwise {eq}), ship repeat rel {r_rep:.2e}, finite {fin} -> {'ok' if good else 'FAIL'}",
          flush=True)
    return dict(tag=tag, rel_sq=r_sq, rel_rep=r_rep, bitwise=eq, finite=fin, ok=good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="kernels/lab/k3_vfasq_1.py")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--out", default="box_out/vfasq/bench.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cpp, cu, segk = sources(ROOT / a.layer)
    aref = J.load_problem(ROOT / "problems/muse_attention_decode").ref
    if a.dry:
        for B, seed in SHAPES:
            d = inputs(aref, B, 0, 1, 1.0, seed, dev, True)
            keys = int(d["seq_lens"].long().sum()) + 8 * B
            print(f"[vfasq] dry B={B}: keys {keys}, q8 {tuple(d['q8'].shape)}, qkv {tuple(d['qkv2'].shape)} stride {d['qkv2'].stride()}, "
                  f"page_table {tuple(d['page_table'].shape)}, k_cache {tuple(d['k_cache'].shape)} {d['k_cache'].dtype}", flush=True)
            del d
        build(cpp, cu)
        return 0
    mod = build(cpp, cu)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    occ = int(mod.vfa_occupancy(4))
    print(f"[vfasq] {torch.cuda.get_device_name()} sms {sms} occupancy(ST 4) {occ} segk {segk}", flush=True)
    assert occ >= 2, "VFA needs 2 CTAs / SM"
    run = Run(mod, sms, segk, dev)
    res = {"cases": [], "shapes": {}}
    for (B, ctx, page, kvs, seed) in CASES:
        res["cases"].append(check(run, inputs(aref, B, ctx, page, kvs, seed, dev, False), f"case B={B} ctx={ctx} page={page} kvs={kvs:g}"))
    timer = J.GpuTimer()
    ovh = timer.replay_overhead_us()
    for B, seed in SHAPES:
        d = inputs(aref, B, 0, 1, 1.0, seed, dev, True)
        res["cases"].append(check(run, d, f"board v{B}f"))
        keys = int(d["seq_lens"].long().sum()) + 8 * B
        tiles = keys / run.G / 64.0
        hq = d["config"]["num_q_heads"]
        graphs = {}
        for name, dg in ARMS:
            ws = run.ws()
            o = torch.empty((B * 8, hq * 64), dtype=torch.bfloat16, device=dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(d, dg, ws, o)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=timer.stream):
                run(d, dg, ws, o)
            g.replay()
            torch.cuda.synchronize()
            graphs[name] = (g, ws)
        names = [n for n, _ in ARMS]
        times = {n: [] for n in names}
        for r in range(a.reps):
            k = r % len(names)
            for n in names[k:] + names[:k]:
                times[n].append(timer.time_replay(graphs[n][0]) - ovh)
        med = {n: statistics.median(v) for n, v in times.items()}
        dirty = [n for n, (g, ws) in graphs.items() if not clean(ws)]
        rec = {"keys": keys, "tiles_per_cta": tiles, "median_us": med, "all_us": times, "dirty_ws": dirty,
               "d_sq": (med["sq"] + med["sq_b"] - med["ship"] - med["ship_b"]) / 2, "d_ctrl": med["ship_b"] - med["ship"],
               "d_nc_sq": med["nc_sq"] - (med["nc_ship"] + med["nc_ship_b"]) / 2, "d_nc_ctrl": med["nc_ship_b"] - med["nc_ship"]}
        res["shapes"][f"v{B}f"] = rec
        print(f"[vfasq] board v{B}f: keys {keys}, {tiles:.1f} tiles / CTA, replay overhead {ovh:.1f} us, dirty ws {dirty}", flush=True)
        for n in names:
            print(f"[vfasq]   {n:10s} {med[n]:7.1f} us  (min {min(times[n]):7.1f})", flush=True)
        print(f"[vfasq]   gather  sq - ship {rec['d_sq']:+.2f} us (control ship_b - ship {rec['d_ctrl']:+.2f}) | nocopy sq - ship "
              f"{rec['d_nc_sq']:+.2f} us (control {rec['d_nc_ctrl']:+.2f}); per tile {rec['d_nc_sq'] * 1e3 / tiles:+.1f} ns", flush=True)
        del graphs, d
        torch.cuda.empty_cache()
    ok = all(c["ok"] for c in res["cases"])
    gate = {k: v["d_sq"] <= -3.0 for k, v in res["shapes"].items()}
    res["ok"], res["gate_3us"] = ok, gate
    print(f"[vfasq] checks {'ALL OK' if ok else 'FAILED'}; gate (gather sq - ship <= -3 us): {gate}", flush=True)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(f"[vfasq] wrote {p}", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())

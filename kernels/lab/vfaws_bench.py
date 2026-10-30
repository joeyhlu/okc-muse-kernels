"""vfaws_bench.py (R-vfaws-1): is the VFA merge workspace (ws_o / ws_l / tickets, zero between calls) left dirty by the ship, does
that change outputs, and does the per-thread release fence (tools/vfaws_patches.py, lab DIAG 24) fix it, at what cost?

One VFA extension (no layer import / prepare / prologue; q8 = e4m3(rms(q)) as kernels/lab/vfasq_bench.py) with arms
  ship (DIAG 0, VFA_WSFIX_ON false), fix (DIAG 24), ship_b (DIAG 0, own graph + workspace: middle control).
Per shape (attention-board verify 16 / 32 full; 3 input sets = seeds s, s + 5000, s + 5001, same tensor shapes): goldens per set =
ship eager calls on fresh zero workspaces, accepted when the workspace is clean afterwards and two calls agree (row max rel
< 5e-3). Then --iters rounds; each round, per arm in rotating order: copy input set (round % 3) into the graph's static inputs,
judge-style replay (256 MB L2 flush read + 300 us gate), compare the output row by row ((token, head) rows of 64) to that
set's golden, then inspect the arm's workspace (never re-zeroed by the bench: a dirty workspace is consumed by the next replay,
exactly as in the layer). pdl 0 (graph replays are separate launches; no PDL between them).

  box: python3 -u kernels/lab/vfaws_bench.py --layer=kernels/lab/k3_vfaws_1.py --out=box_out/vfaws/bench.json
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
sys.path[:0] = [str(ROOT / "tools/judge"), str(ROOT / "kernels/lab")]
import judge as J  # noqa: E402
from vfasq_bench import Run, inputs, clean  # noqa: E402

ARMS = (("ship", 0), ("fix", 24), ("ship_b", 0))
SHAPES = ((16, 610), (32, 612))
TENS = ("q8", "qkv2", "seq_lens", "page_table", "k_cache", "v_cache")


def build(layer):
    vals = {}
    for n in ast.parse(Path(layer).read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("VFA_CPP_SRC", "VFA_CUDA_SRC", "VFA_SEGK"):
            vals[n.targets[0].id] = ast.literal_eval(n.value)
    cu = vals["VFA_CUDA_SRC"]
    assert "#if VFA_WSLAB" in cu and "VFA_WSFIX_ON" in cu, "not a VFAWS lab file (tools/vfaws_patches.py)"
    cu = cu.replace("#if VFA_LAB", "#if 0").replace("#if VFA_WSLAB", "#if 1").replace("VFA_WSFIX_ON", "false")
    from torch.utils.cpp_extension import load_inline
    tag = hashlib.sha1((vals["VFA_CPP_SRC"] + cu).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"vfaws_{tag}", cpp_sources=vals["VFA_CPP_SRC"], cuda_sources=cu, functions=["vfa_forward", "vfa_occupancy"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"[vfaws] build {time.time() - t0:.0f}s", flush=True)
    return mod, int(vals.get("VFA_SEGK", 300))


def rowrel(o, g):
    o, g = o.float().view(-1, 64), g.float().view(-1, 64)
    return ((o - g).norm(dim=1) / g.norm(dim=1).clamp(min=1e-20))


def dirt(ws, B):
    """(n dirty ws_o rows, n dirty ws_l rows, n nonzero tickets, max |ws_o|, dirty row ids mod 128, requests)"""
    rows = (ws[0][:B * 128] != 0).any(dim=1)
    l = ws[1][:B * 128] != 0
    t = ws[2][:B] != 0
    idx = torch.nonzero(rows | l).flatten().tolist()
    return dict(o=int(rows.sum()), l=int(l.sum()), t=int(t.sum()), mx=float(ws[0][:B * 128].abs().max()),
                rows=sorted(set(i % 128 for i in idx))[:32], req=sorted(set(i // 128 for i in idx))[:8])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="kernels/lab/k3_vfaws_1.py")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="box_out/vfaws/bench.json")
    ap.add_argument("--smoke", action="store_true", help="local 3070 code-path smoke (sm_86 kernels are empty): goldens unchecked")
    a = ap.parse_args()
    dev = torch.device("cuda")
    mod, segk = build(ROOT / a.layer)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    run = Run(mod, sms, segk, dev)
    if a.smoke:   # local code-path smoke: no sm_90a kernel can run here; out = a copy of the gate columns, ws untouched
        run.__class__ = type("FakeRun", (Run,), {"__call__": lambda self, d, diag, ws, out: out.copy_(d["qkv2"][:, -out.shape[1]:])})
    else:
        print(f"[vfaws] {torch.cuda.get_device_name()} sms {sms} occupancy(ST 4) {int(mod.vfa_occupancy(4))}", flush=True)
    aref = J.load_problem(ROOT / "problems/muse_attention_decode").ref
    timer = J.GpuTimer()
    ovh = timer.replay_overhead_us()
    res = {"shapes": {}}
    for B, seed in SHAPES:
        sets = [inputs(aref, B, 0, 1, 1.0, s, dev, True) for s in (seed, seed + 5000, seed + 5001)]
        for s in sets[1:]:
            assert all(s[k].shape == sets[0][k].shape for k in TENS), "input sets differ in shape"
        hq = sets[0]["config"]["num_q_heads"]
        static = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in sets[0].items()}
        gold, gtries = [], []
        for si, s in enumerate(sets):
            prev, tries = None, 0
            while True:
                tries += 1
                ws = run.ws()
                o = torch.empty((B * 8, hq * 64), dtype=torch.bfloat16, device=dev)
                run(s, 0, ws, o)
                torch.cuda.synchronize()
                if clean(ws):
                    if prev is not None and float(rowrel(o, prev).max()) < 5e-3:
                        break
                    prev = o
                    if a.smoke:
                        break
                assert tries < 12, f"no stable golden for set {si}"
            gold.append(prev)
            gtries.append(tries)
        graphs = {}
        for name, dg in ARMS:
            ws = run.ws()
            o = torch.empty((B * 8, hq * 64), dtype=torch.bfloat16, device=dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run(static, dg, ws, o)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            for t in ws:
                t.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=timer.stream):
                run(static, dg, ws, o)
            torch.cuda.synchronize()
            graphs[name] = (g, ws, o)
        names = [n for n, _ in ARMS]
        rec = {n: {"us": [], "rel": [], "dirty_after": [], "dirty_before": [], "events": []} for n in names}
        state = {n: False for n in names}
        for r in range(a.iters):
            si = r % 3
            k = r % len(names)
            for n in names[k:] + names[:k]:
                g, ws, o = graphs[n]
                for key in TENS:
                    static[key].copy_(sets[si][key])
                torch.cuda.synchronize()
                us = timer.time_replay(g) - ovh
                rr = float(rowrel(o, gold[si]).max())
                dw = not clean(ws)
                x = rec[n]
                x["us"].append(us)
                x["rel"].append(rr)
                x["dirty_before"].append(state[n])
                x["dirty_after"].append(dw)
                if dw and len(x["events"]) < 12:
                    x["events"].append(dict(round=r, set=si, rel=rr, **dirt(ws, B)))
                state[n] = dw
        out = {}
        for n in names:
            x = rec[n]
            rel = x["rel"]
            db, da = x["dirty_before"], x["dirty_after"]
            out[n] = dict(replays=len(rel), dirty_after=sum(da), rel_med=statistics.median(rel), rel_max=max(rel),
                          bad_2e2=sum(v > 2e-2 for v in rel), bad_5e3=sum(v > 5e-3 for v in rel),
                          bad_5e3_dirty_before=sum(v > 5e-3 and b for v, b in zip(rel, db)),
                          bad_5e3_dirty_after_only=sum(v > 5e-3 and d and not b for v, b, d in zip(rel, db, da)),
                          bad_5e3_clean=sum(v > 5e-3 and not d and not b for v, b, d in zip(rel, db, da)),
                          rel_q99=sorted(rel)[int(0.99 * (len(rel) - 1))], us_med=statistics.median(x["us"]), events=x["events"],
                          rel_all=rel, us_all=x["us"], dirty_all=da)
            print(f"[vfaws] v{B}f {n:6s}: replays {len(rel)}, dirty after {sum(da)}, out rows rel > 5e-3: {out[n]['bad_5e3']} "
                  f"(after a dirty ws {out[n]['bad_5e3_dirty_before']}, in the dirtying call {out[n]['bad_5e3_dirty_after_only']}, "
                  f"clean {out[n]['bad_5e3_clean']}), > 2e-2: {out[n]['bad_2e2']}, max {max(rel):.2e}, q99 {out[n]['rel_q99']:.1e}, med {out[n]['rel_med']:.1e}; "
                  f"median {out[n]['us_med']:.1f} us", flush=True)
            for e in x["events"][:4]:
                print(f"[vfaws]     event {e}", flush=True)
        us = {n: rec[n]["us"] for n in names}
        d_fix = statistics.median([f - (s + b) / 2 for f, s, b in zip(us["fix"], us["ship"], us["ship_b"])])
        d_ctl = statistics.median([b - s for s, b in zip(us["ship"], us["ship_b"])])
        out["d_fix_us"], out["d_ctrl_us"], out["golden_tries"] = d_fix, d_ctl, gtries
        print(f"[vfaws] v{B}f paired fix - ship {d_fix:+.2f} us (control ship_b - ship {d_ctl:+.2f}); golden tries {gtries}", flush=True)
        res["shapes"][f"v{B}f"] = out
        del graphs, sets, static, gold
        torch.cuda.empty_cache()
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))
    print(f"[vfaws] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""dpin_lab.py: R-dpin-1 (notes/memory/gpu_requests.md, notes/study/t128_idle_dense.md): standalone A/B of the batched / NN cuBLASLt pins
(tools/dpin_patches.py) against today's pins / default GEMM, on the patched layer's own LT extension (lt_plan_b).

Cases: qkv m128 (today LT_PIN tile 80 c2 cl7 b1 vs new TN b16 tile 92 c2 cl5), qkv m256 (tile 134 c2 cl9 vs tile 124 c0 cl2),
o m128 (torch default vs NN b8 tile 18 c3 cl5). Per case: outputs bitwise new vs today (and both vs torch.mm, rel), then judge-style
timing (GpuTimer flush + 300 us gate, CUDA graph per arm, rotated rounds) in two contexts: clean, and dirty = a 16 MB fill
written right before the GEMM inside the graph (R-cbl-0's in-layer write-back), reported net of a fill-only graph.

    python3 -u kernels/lab/dpin_lab.py [--layer=kernels/lab/k3_dpin_1.py] [--rounds=40] [--out=box_out/dpin/lab.json]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = [  # (name, proj, T, N, K, today cfg or None = torch default, today (nn, batch), new cfg, new (nn, batch))
    ("qkv128", "qkv", 128, 2304, 8192, (66, 80, 35, 1, 0, 0, 2, 0, 7), (0, 1), (66, 92, 35, 1, 0, 0, 2, 0, 5), (0, 16)),
    ("qkv256", "qkv", 256, 2304, 8192, (66, 134, 35, 1, 0, 0, 2, 0, 9), (0, 1), (66, 124, 35, 1, 0, 0, 0, 0, 2), (0, 1)),
    ("o128", "o", 128, 8192, 1024, None, (0, 1), (66, 18, 35, 1, 0, 0, 3, 0, 5), (1, 8)),
]


def log(s):
    print(s, flush=True)


def main():
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    import judge as J
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    layer = Path(a.get("layer", "kernels/lab/k3_dpin_1.py"))
    rounds = int(a.get("rounds", 40))
    outp = Path(a.get("out", "box_out/dpin/lab.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    prob = J.load_problem(ROOT / "problems/muse_decode_layer")
    mod = J.load_kernel(ROOT / layer, "okc_dpin_lab", prob)
    lt = mod._lt_build()
    if lt is None:
        log(f"[dpin] LT build failed: {mod._LT['err']}")
        return 2
    dev = "cuda"
    ws = torch.empty((mod.LT_WS,), dtype=torch.uint8, device=dev)
    gen = torch.Generator(device=dev).manual_seed(2468)
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    dirty = torch.empty((16 << 20) // 4, dtype=torch.float32, device=dev)
    res = {"rounds": rounds, "overhead_us": ov, "cases": {}}

    def plan(T, n, k, cfg, lay):
        code = int(lt.lt_plan_b(T, n, k, 0, lay[0], lay[1], mod.LT_WS, 32))
        pid, cnt = code >> 8, code & 255
        ai = next((i for i in range(cnt) if tuple(lt.lt_cfg(pid, i)) == tuple(cfg)), None)
        if ai is None:
            ai = int(lt.lt_make(pid, list(cfg), mod.LT_WS))
            if ai < 0 or tuple(lt.lt_cfg(pid, ai)) != tuple(cfg):
                return None
        return pid, ai

    for name, proj, T, N, K, tcfg, tlay, ncfg, nlay in CASES:
        w = (torch.randn((N, K), generator=gen, device=dev) / K ** 0.5).to(torch.bfloat16)
        wt = w.t().contiguous()
        x = torch.randn((T, K), generator=gen, device=dev).to(torch.bfloat16)
        arms = {"torch": None}
        if tcfg is not None:
            arms["today"] = (plan(T, N, K, tcfg, tlay), tlay)
        arms["new"] = (plan(T, N, K, ncfg, nlay), nlay)
        if any(v is not None and v[0] is None for v in arms.values()):
            log(f"[dpin] {name}: config rejected by cuBLASLt: " + str({k: v for k, v in arms.items() if v is not None and v[0] is None}))
            res["cases"][name] = {"rejected": True}
            continue
        outs = {}

        def run(arm, o):
            if arms[arm] is None:
                torch.mm(x, w.t(), out=o)
            else:
                (pid, ai), lay = arms[arm]
                lt.lt_run(pid, ai, wt if lay[0] else w, x, o, 0.0, ws)
        for arm in arms:
            o = torch.full((T, N), float("nan"), dtype=torch.bfloat16, device=dev)
            run(arm, o)
            o2 = torch.full_like(o, float("nan"))
            run(arm, o2)
            torch.cuda.synchronize()
            outs[arm] = (o, bool(torch.equal(o, o2)))
        ref = (x.double() @ w.double().t())
        base_arm = "today" if "today" in arms else "torch"
        chk = {arm: {"repeat_bitwise": rep, "bitwise_vs_" + base_arm: bool(torch.equal(o, outs[base_arm][0])),
                     "rel64": float((o.double() - ref).norm() / ref.norm()),
                     "max_abs_vs_" + base_arm: float((o.float() - outs[base_arm][0].float()).abs().max())}
               for arm, (o, rep) in outs.items()}
        log(f"[dpin] {name}: " + " | ".join(f"{arm} " + " ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in c.items())
                                           for arm, c in chk.items()))
        times = {}
        for ctx in ("clean", "dirty"):
            graphs = {}
            names = list(arms) + (["fill"] if ctx == "dirty" else [])
            for arm in names:
                o = torch.empty((T, N), dtype=torch.bfloat16, device=dev)

                def fn(_d, arm=arm, o=o, ctx=ctx):
                    if ctx == "dirty":
                        dirty.fill_(1.0)
                    if arm != "fill":
                        run(arm, o)
                    return o
                graphs[arm] = timer.capture(fn, {})[0]
            tt = {n: [] for n in names}
            for rd in range(rounds):
                for n in names[rd % len(names):] + names[:rd % len(names)]:
                    tt[n].append(timer.time_replay(graphs[n]) - ov)
            med = {n: statistics.median(v) for n, v in tt.items()}
            if ctx == "dirty":
                med = {n: (v - med["fill"] if n != "fill" else v) for n, v in med.items()}
            times[ctx] = med
            b = med[base_arm]
            log(f"[dpin] {name} {ctx}: " + " ".join(f"{n} {v:.2f}" for n, v in med.items()) + f" us | new - {base_arm} {med['new'] - b:+.2f}")
            del graphs
        res["cases"][name] = {"check": chk, "times": times, "base": base_arm}
        outp.write_text(json.dumps(res, indent=1))

    # decision (pre-registered R-dpin-1)
    q = res["cases"].get("qkv128", {})
    verdict = "INVALID (qkv128 rejected)" if not q or q.get("rejected") else None
    if verdict is None:
        c = q["check"]["new"]
        same = c["bitwise_vs_today"]
        tol = c["rel64"] <= 1.5 * q["check"]["today"]["rel64"] + 1e-6
        d = q["times"]["clean"]["new"] - q["times"]["clean"]["today"]
        dd = q["times"]["dirty"]["new"] - q["times"]["dirty"]["today"]
        if not (c["repeat_bitwise"] and (same or tol)):
            verdict = "KILL (numerics)"
        elif d <= -1.5:
            verdict = f"GO ({'bitwise' if same else 'tolerance class'}): qkv128 clean {d:+.2f} us, dirty {dd:+.2f}"
        else:
            verdict = f"KILL: qkv128 clean {d:+.2f} us (bar -1.5), dirty {dd:+.2f}"
    res["verdict"] = verdict
    outp.write_text(json.dumps(res, indent=1))
    log(f"[dpin] verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""bndx_sweep.py: R-bndx-2 (notes/memory/gpu_requests.md, notes/study/bndx.md): launch-config sweep of the STOCK SGLang JIT
fused_norm_residual at T = 128 / 256 on H200 (the layer runs the auto config there; V9_BND_ROUTE only covers buckets <= 32).

Arms: sgl (auto) / sgl2 (position control), t<threads><p|n> = launch_config (threads, prefetch) at cluster 1, cl2 / cl4 = cluster
kernel. Per arm: bitwise flag + max rel error of out / new_stream vs auto; judge GpuTimer (flush + 300 us gate), 30 rotated rounds,
contexts solo [boundary] and chain [cuBLAS producer -> boundary (-> consumer for attn)], both boundaries.

    python3 -u kernels/lab/bndx_sweep.py [--ts=128,256] [--rounds=30] [--out=box_out/bndx/sweep.json]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = 8192
CFG = {"t256p": ((256, True), None), "t256n": ((256, False), None), "t512p": ((512, True), None), "t512n": ((512, False), None),
       "t1024p": ((1024, True), None), "t1024n": ((1024, False), None), "cl2": (None, 2), "cl4": (None, 4)}


def log(s):
    print(s, flush=True)


def main():
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    import judge as J
    import sglang.kernels.ops.muse_spark_v12.norm_residual as NR
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    ts = [int(x) for x in a.get("ts", "128,256").split(",")]
    rounds = int(a.get("rounds", 30))
    outp = Path(a.get("out", "box_out/bndx/sweep.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    res = {"rounds": rounds, "err": {}, "timing": {}}
    gen0 = torch.Generator(device="cuda").manual_seed(77)

    def vec(std):
        return (std * torch.randn((H,), generator=gen0, device="cuda")).to(torch.bfloat16)

    def gates(std):
        g = vec(std).float()
        b = torch.sigmoid(g / 0.3)
        return torch.sqrt(torch.clamp(torch.sigmoid(-g / 0.3) * (1 + b), min=1e-3)).contiguous(), b.contiguous()
    a1, b1 = gates(0.3)
    a2, b2 = gates(0.3)
    ops = {"attn": dict(pw=None, qw=(1.0 + vec(0.1).float()).to(torch.bfloat16), al=a1, bt=b1, isc=False),
           "ffn": dict(pw=(1.0 + vec(0.1).float()).to(torch.bfloat16), qw=(1.0 + vec(0.1).float()).to(torch.bfloat16), al=a2, bt=b2,
                       isc=True)}

    def run(which, arm, branch, stream):
        o = ops[which]
        lc, cs = (None, None) if arm.startswith("sgl") else CFG[arm]
        return NR.fused_norm_residual(branch, stream, 1e-8, 1e-5, post_norm_is_input_scale=o["isc"], post_weight=o["pw"],
                                      pre_weight=o["qw"], alpha=o["al"], beta=o["bt"], want_router=False, launch_config=lc,
                                      cluster_size=cs)[:2]

    gen = torch.Generator(device="cuda").manual_seed(4242)
    arms = ["sgl"] + list(CFG) + ["sgl2"]
    usable = {}
    for T in ts:
        for which in ops:
            stream = torch.randn((T, H), generator=gen, device="cuda") * 2
            branch = (torch.randn((T, H), generator=gen, device="cuda") * 3).to(torch.bfloat16)
            r_out, r_ns = run(which, "sgl", branch, stream)
            for arm in CFG:
                try:
                    o, ns = run(which, arm, branch, stream)
                    torch.cuda.synchronize()
                    eo = float((o.float() - r_out.float()).norm() / r_out.float().norm())
                    en = float((ns - r_ns).norm() / r_ns.norm())
                    bit = bool(torch.equal(o, r_out)) and bool(torch.equal(ns, r_ns))
                    res["err"][f"{which} T{T} {arm}"] = [bit, eo, en]
                    usable[(T, which, arm)] = True
                    log(f"[sweep] T={T} {which} {arm}: bitwise {bit} out rel {eo:.2e} stream rel {en:.2e}")
                except Exception as ex:   # e.g. an unsupported cluster shape
                    usable[(T, which, arm)] = False
                    log(f"[sweep] T={T} {which} {arm}: unavailable ({ex!r:.160})")
    outp.write_text(json.dumps(res, indent=1))

    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    wo = (torch.randn((H, 1024), generator=gen, device="cuda") / 32).to(torch.bfloat16)
    wun = (torch.randn((H, 4096), generator=gen, device="cuda") / 64).to(torch.bfloat16)
    wcon = (torch.randn((4608, H), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    for T in ts:
        for which in ops:
            stream = torch.randn((T, H), generator=gen, device="cuda") * 2
            w_in = wo if which == "attn" else wun
            xin = (torch.randn((T, w_in.shape[1]), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            branch0 = torch.mm(xin, w_in.t())
            al = [x for x in arms if x.startswith("sgl") or usable.get((T, which, x))]
            for ctx in ("solo", "chain"):
                graphs = {}
                for arm in al:
                    def fn(_d, arm=arm, ctx=ctx):
                        br = torch.mm(xin, w_in.t()) if ctx == "chain" else branch0
                        o = run(which, arm, br, stream)[0]
                        return torch.mm(o, wcon.t()) if (ctx == "chain" and which == "attn") else o
                    graphs[arm] = timer.capture(fn, {})[0]
                tt = {n: [] for n in al}
                for rd in range(rounds):
                    for n in al[rd % len(al):] + al[:rd % len(al)]:
                        tt[n].append(timer.time_replay(graphs[n]) - ov)
                med = {n: statistics.median(v) for n, v in tt.items()}
                sg = (med["sgl"] + med["sgl2"]) / 2
                res["timing"][f"{which} T{T} {ctx}"] = {"med": med}
                log(f"[sweep] T={T} {which} {ctx}: sgl {sg:.2f} us | " + " ".join(f"{n} {med[n] - sg:+.2f}" for n in al if not n.startswith("sgl"))
                    + f" | sgl2-sgl {med['sgl2'] - med['sgl']:+.2f}")
                outp.write_text(json.dumps(res, indent=1))
                del graphs

    # decision (pre-registered R-bndx-2)
    bad_ctl = sum(1 for v in res["timing"].values() if abs(v["med"]["sgl2"] - v["med"]["sgl"]) > 0.4)
    best = {}
    for T in ts:
        for which in ops:
            m = res["timing"].get(f"{which} T{T} chain", {}).get("med", {})
            if not m:
                continue
            sg = (m["sgl"] + m["sgl2"]) / 2
            ok = [x for x in m if not x.startswith("sgl") and (res["err"][f"{which} T{T} {x}"][0] or res["err"][f"{which} T{T} {x}"][2] <= 1e-6)]
            if ok:
                b = max(ok, key=lambda x: sg - m[x])
                best[f"{which} T{T}"] = (b, round(sg - m[b], 2), res["err"][f"{which} T{T} {b}"][0])
    log(f"[sweep] best chain gain per cell (arm, us, bitwise): {best}")
    go = bad_ctl <= 2 and all(best.get(f"{w} T256", ("", 0))[1] >= 0.8 for w in ops)
    res["best"], res["verdict"] = best, ("INVALID (controls)" if bad_ctl > 2 else ("GO" if go else "KILL"))
    outp.write_text(json.dumps(res, indent=1))
    log(f"[sweep] verdict: {res['verdict']} (controls off in {bad_ctl} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

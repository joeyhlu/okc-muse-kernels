"""gft_lab.py: R-graft-1 stage 1 (notes/memory/gpu_requests.md): standalone falsifier, SGLang's stock fused_norm_residual in the
ship's configs vs alygg77's Gluon row boundary (kernels/lab/gft_kern.py = tools/graft_patches.py --emit-kern) at T = 128 / 256,
both layer boundaries (attn: no post weight; ffn: post weight as input scale), judge GpuTimer (256 MB L2 flush + 300 us gate),
rotated rounds, contexts solo [boundary] and chain [cuBLAS producer -> boundary (-> consumer for attn)] as kernels/lab/bndx_sweep.py.

Arms: sgl (ship: attn T=256 at (1024 threads, prefetch), everything else SGLang's auto config) / gft / sgl2 (position control).
Numerics: gft vs sgl bitwise flag, out rel, out bf16-differ fraction, stream rel.
Decision (pre-registered): pair gain at T = sum over both boundaries of (mean(sgl, sgl2) - gft), chain context; stage 2 runs iff
some T has pair gain >= 1.5 us and every |sgl2 - sgl| <= 0.4 us.

    python3 -u kernels/lab/gft_lab.py [--ts=128,256] [--rounds=30] [--out=box_out/gft/lab.json]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = 8192
SHIP_CFG = {("attn", 256): (1024, True)}   # V9_BND_ROUTE ('attn', 256): (1, (1024, True)); other cells: auto


def log(s):
    print(s, flush=True)


def run_lab(ts=(128, 256), rounds=30, outp=None):
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    sys.path.insert(0, str(ROOT / "kernels" / "lab"))
    import judge as J
    import sglang.kernels.ops.muse_spark_v12.norm_residual as NR
    import gft_kern as K
    if K._GFT_IMPORT is not None:
        log(f"[gft-lab] gluon unavailable: {K._GFT_IMPORT}")
        return {"verdict": "VOID (gluon)"}
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
        T = branch.shape[0]
        if arm == "gft":
            h = torch.empty_like(branch)
            s = torch.empty_like(stream)
            nw = 16 if T > 192 else 32
            K._graft_row_boundary[(T,)](branch, stream, o["pw"] if o["pw"] is not None else o["qw"], o["qw"], o["al"], o["bt"], h, s,
                                        H, H, o["pw"] is not None, 4, nw, 1e-8, 1e-5, num_warps=nw, launch_pdl=True)
            return h, s
        lc = SHIP_CFG.get((which, T))
        return NR.fused_norm_residual(branch, stream, 1e-8, 1e-5, post_norm_is_input_scale=o["isc"], post_weight=o["pw"],
                                      pre_weight=o["qw"], alpha=o["al"], beta=o["bt"], want_router=False, launch_config=lc,
                                      cluster_size=1 if lc else None)[:2]

    gen = torch.Generator(device="cuda").manual_seed(4242)
    for T in ts:
        for which in ops:
            stream = torch.randn((T, H), generator=gen, device="cuda") * 2
            branch = (torch.randn((T, H), generator=gen, device="cuda") * 3).to(torch.bfloat16)
            r_out, r_ns = run(which, "sgl", branch, stream)
            o, ns = run(which, "gft", branch, stream)
            torch.cuda.synchronize()
            eo = float((o.float() - r_out.float()).norm() / r_out.float().norm())
            en = float((ns - r_ns).norm() / r_ns.norm())
            nd = float((o != r_out).float().mean())
            bit = bool(torch.equal(o, r_out)) and bool(torch.equal(ns, r_ns))
            res["err"][f"{which} T{T}"] = [bit, eo, nd, en]
            log(f"[gft-lab] T={T} {which}: gft vs sgl bitwise {bit} out rel {eo:.2e} (bf16 differ {nd:.1e}) stream rel {en:.2e}")
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    wo = (torch.randn((H, 1024), generator=gen, device="cuda") / 32).to(torch.bfloat16)
    wun = (torch.randn((H, 4096), generator=gen, device="cuda") / 64).to(torch.bfloat16)
    wcon = (torch.randn((4608, H), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    arms = ["sgl", "gft", "sgl2"]
    for T in ts:
        for which in ops:
            stream = torch.randn((T, H), generator=gen, device="cuda") * 2
            w_in = wo if which == "attn" else wun
            xin = (torch.randn((T, w_in.shape[1]), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            branch0 = torch.mm(xin, w_in.t())
            for ctx in ("solo", "chain"):
                graphs = {}
                for arm in arms:
                    def fn(_d, arm=arm, ctx=ctx):
                        br = torch.mm(xin, w_in.t()) if ctx == "chain" else branch0
                        o = run(which, "sgl" if arm == "sgl2" else arm, br, stream)[0]
                        return torch.mm(o, wcon.t()) if (ctx == "chain" and which == "attn") else o
                    graphs[arm] = timer.capture(fn, {})[0]
                tt = {n: [] for n in arms}
                for rd in range(rounds):
                    for n in arms[rd % len(arms):] + arms[:rd % len(arms)]:
                        tt[n].append(timer.time_replay(graphs[n]) - ov)
                med = {n: statistics.median(v) for n, v in tt.items()}
                sg = (med["sgl"] + med["sgl2"]) / 2
                res["timing"][f"{which} T{T} {ctx}"] = {"med": med, "gain": sg - med["gft"], "ctl": med["sgl2"] - med["sgl"]}
                log(f"[gft-lab] T={T} {which} {ctx}: sgl {sg:.2f} us | gft {med['gft'] - sg:+.2f} | sgl2-sgl {med['sgl2'] - med['sgl']:+.2f}")
                if outp:
                    outp.write_text(json.dumps(res, indent=1))
                del graphs
    bad_ctl = [k for k, v in res["timing"].items() if abs(v["ctl"]) > 0.4]
    pair = {T: {ctx: sum(res["timing"][f"{w} T{T} {ctx}"]["gain"] for w in ops) for ctx in ("solo", "chain")} for T in ts}
    numer_ok = all(v[1] < 1e-2 and v[3] < 1e-4 for v in res["err"].values())
    go_t = [T for T in ts if pair[T]["chain"] >= 1.5]
    res["pair_gain"] = pair
    res["verdict"] = ("VOID (controls)" if bad_ctl else "KILL (numerics)" if not numer_ok else
                      ("GO " + ",".join(map(str, go_t))) if go_t else "KILL")
    log(f"[gft-lab] pair gains (sgl - gft, both boundaries) {json.dumps({str(k): {c: round(x, 2) for c, x in v.items()} for k, v in pair.items()})};"
        f" controls off {bad_ctl or 'none'} -> {res['verdict']}")
    if outp:
        outp.write_text(json.dumps(res, indent=1))
    return res


def main():
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    outp = Path(a.get("out", "box_out/gft/lab.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    r = run_lab(tuple(int(x) for x in a.get("ts", "128,256").split(",")), int(a.get("rounds", 30)), outp)
    return 0 if r["verdict"].startswith("GO") else 4


if __name__ == "__main__":
    sys.exit(main())

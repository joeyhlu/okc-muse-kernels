"""bndx_lab.py: R-bndx-1 (notes/memory/gpu_requests.md, notes/study/bndx.md): our own T >= 128 layer boundaries (tools/bndx_kernel.cuh,
SGLang's single-CTA norm_residual arithmetic vendored, persistent rows, channel vectors before the wait, entry trigger) vs the
stock SGLang JIT fused_norm_residual (the kernel the layer's T >= 128 module call runs; in-layer step re-checks the module).

Steps (box only; SGLang + tvm-ffi from the judge image):
  0. ptxas -v through SGLang's JIT build (SGLANG_JIT_LOG_RESOURCE_USAGE=1): regs / spills land in run.log.
  1. bitwise: out and new_stream == the module's at T in (128, 129, 191, 192, 200, 256, 257, 300), both boundaries, every arm,
     plus CUDA-graph replays with the static inputs overwritten (stale-input check).
  2. timing (judge GpuTimer: 256 MB L2 flush + 300 us gate per replay, rotating arm order), T = 128 / 256, both boundaries:
       solo   graph [boundary]
       chain  graph [cuBLAS producer GEMM -> boundary (-> cuBLAS consumer GEMM for attn)] (the layer's neighbours at T >= 128)
     arms: sgl / sgl2 (module, position control), bx (entry trigger, rows looped over G = min(T, 128) CTAs), bxn (+ next-row loads
     before the reductions), bx2 (end trigger = stock placement), bxT (G = T: one row per CTA, the loop's control).

    python3 -u kernels/lab/bndx_lab.py [--ts=128,256] [--rounds=30] [--out=box_out/bndx/lab.json]
"""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUH = ROOT / "tools" / "bndx_kernel.cuh"
H = 8192
ARMS = {"bx": (1, False, False, "G"), "bxn": (1, True, False, "G"), "bx2": (2, False, False, "G"), "bxT": (1, False, False, "T")}
GMAX = 128


def nt_for(T):   # SGLang's _launch_config at H = 8192 on H200 (cluster path off): 256 threads below 192 rows, 512 from 192
    return 256 if T < 192 else 512


def log(s):
    print(s, flush=True)


def main():
    os.environ["SGLANG_JIT_LOG_RESOURCE_USAGE"] = "1"
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    import judge as J
    import sglang.kernels.ops.muse_spark_v12.norm_residual as NR
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    ts = [int(x) for x in a.get("ts", "128,256").split(",")]
    rounds = int(a.get("rounds", 30))
    outp = Path(a.get("out", "box_out/bndx/lab.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    res = {"rounds": rounds, "gate": {}, "bitwise": {}, "timing": {}}

    # Boundary parameters as the layer has them (reference.py: post-norm eps 1e-8, pre-norm eps 1e-5, channel-wise residual gates,
    # effective norm weights 1 + w in bf16; attn: weightless post-norm; ffn: input-scale post-norm with weight). The reference is
    # SGLang's own JIT fused_norm_residual (launch config auto = the kernel the layer's T >= 128 module call runs).
    sys.path.insert(0, str(ROOT / "problems" / "muse_decode_layer"))
    gen0 = torch.Generator(device="cuda").manual_seed(77)

    def vec(std):
        return (std * torch.randn((H,), generator=gen0, device="cuda")).to(torch.bfloat16)

    def gates(std):
        g = vec(std).float()
        b = torch.sigmoid(g / 0.3)
        al = torch.sqrt(torch.clamp(torch.sigmoid(-g / 0.3) * (1 + b), min=1e-3))
        return al.contiguous(), b.contiguous()
    pdl = bool(NR.is_arch_support_pdl())
    ops = {}
    a1, b1 = gates(0.3)
    a2, b2 = gates(0.3)
    ops["attn"] = dict(pw=None, qw=(1.0 + vec(0.1).float()).to(torch.bfloat16), al=a1, bt=b1, isc=False, pe=1e-8, qe=1e-5, cw=True)
    ops["ffn"] = dict(pw=(1.0 + vec(0.1).float()).to(torch.bfloat16), qw=(1.0 + vec(0.1).float()).to(torch.bfloat16), al=a2, bt=b2,
                      isc=True, pe=1e-8, qe=1e-5, cw=True)
    for which, o in ops.items():
        log(f"[bndx] {which}: input-scale post-norm {o['isc']}, post weight {o['pw'] is not None}, pre weight {o['qw'] is not None}, "
            f"channel-wise gate {o['cw']}, eps {o['pe']:g} / {o['qe']:g}, pdl {pdl}")

    def targs(which, nt, arm):
        o = ops[which]
        trig, nxt, spre, _ = ARMS[arm]
        return NR.make_cpp_args(H, nt, pdl, o["isc"], o["pw"] is not None, o["al"] is not None, trig, nxt, spre)

    # ---- step 0: ptxas resource lines come from SGLang's JIT build (SGLANG_JIT_LOG_RESOURCE_USAGE=1 -> -Xptxas=-v), in run.log
    bad = set()

    # one JIT module, one compile: a wrapper per (boundary, threads, arm)
    allc = sorted({(w, nt, arm) for w in ops for nt in (256, 512) for arm in ARMS})
    wname = lambda w, nt, arm: f"run_{w}_{nt}_{arm}"  # noqa: E731
    import time as _t
    t0 = _t.time()
    MOD = NR.load_jit("okc_bndx_lab", *NR.make_cpp_args(H, 1), cuda_files=[str(CUH.resolve())],
                      cuda_wrappers=[(wname(w, nt, arm), f"OkcBndx<{targs(w, nt, arm)}>::run") for w, nt, arm in allc])
    log(f"[bndx] JIT module ({len(allc)} wrappers) built in {_t.time() - t0:.0f} s")

    def run_bx(which, arm, branch, stream, out, ns):
        o = ops[which]
        T = branch.shape[0]
        g = T if ARMS[arm][3] == "T" else min(T, GMAX)
        getattr(MOD, wname(which, nt_for(T), arm))(branch, stream, out, ns, o["pw"], o["qw"], o["al"], o["bt"], o["pe"], o["qe"],
                                                    o["cw"], g)

    def run_sgl(which, branch, stream):
        o = ops[which]
        return NR.fused_norm_residual(branch, stream, o["pe"], o["qe"], post_norm_is_input_scale=o["isc"], post_weight=o["pw"],
                                      pre_weight=o["qw"], alpha=o["al"], beta=o["bt"], want_router=False)[:2]

    # ---- step 1: bitwise, eager + graph replays with changed inputs
    gen = torch.Generator(device="cuda").manual_seed(4242)
    allok = True
    for which in ops:
        for T in (128, 129, 191, 192, 200, 256, 257, 300):
            for arm in ARMS:
                stream = torch.randn((T, H), generator=gen, device="cuda") * 2
                branch = (torch.randn((T, H), generator=gen, device="cuda") * 3).to(torch.bfloat16)
                r_out, r_ns = run_sgl(which, branch, stream)
                out = torch.empty_like(branch)
                ns = torch.empty_like(stream)
                run_bx(which, arm, branch, stream, out, ns)
                torch.cuda.synchronize()
                ok = bool(torch.equal(out, r_out)) and bool(torch.equal(ns, r_ns))
                # graph replay with the static inputs overwritten
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    run_bx(which, arm, branch, stream, out, ns)
                torch.cuda.synchronize()
                gok = True
                for k in range(3):
                    stream.copy_(torch.randn((T, H), generator=gen, device="cuda") * (1 + k))
                    branch.copy_((torch.randn((T, H), generator=gen, device="cuda") * 3).to(torch.bfloat16))
                    g.replay()
                    r_out, r_ns = run_sgl(which, branch, stream)
                    torch.cuda.synchronize()
                    gok = gok and bool(torch.equal(out, r_out)) and bool(torch.equal(ns, r_ns))
                res["bitwise"][f"{which} T{T} {arm}"] = [ok, gok]
                allok = allok and ok and gok
                if not (ok and gok):
                    log(f"[bndx] BITWISE FAIL {which} T={T} {arm}: eager {ok} replays {gok} (out max|d| "
                        f"{float((out.float() - r_out.float()).abs().max()):.3g}, ns max|d| {float((ns - r_ns).abs().max()):.3g})")
                del g
    log(f"[bndx] bitwise (eager + 3 changed-input replays, 2 boundaries x 8 T x {len(ARMS)} arms): {'ALL OK' if allok else 'FAIL'}")
    res["bitwise_all"] = allok
    outp.write_text(json.dumps(res, indent=1))

    # ---- step 2: timing
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    res["overhead_us"] = ov
    wo = (torch.randn((H, 1024), generator=gen, device="cuda") / 32).to(torch.bfloat16)      # o-proj [8192, 1024]
    wun = (torch.randn((H, 4096), generator=gen, device="cuda") / 64).to(torch.bfloat16)     # unpooler [8192, 4096]
    wcon = (torch.randn((4608, H), generator=gen, device="cuda") * 0.02).to(torch.bfloat16)
    for T in ts:
        for which in ops:
            stream = torch.randn((T, H), generator=gen, device="cuda") * 2
            xin = (torch.randn((T, wo.shape[1] if which == "attn" else wun.shape[1]), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
            w_in = wo if which == "attn" else wun
            branch0 = torch.mm(xin, w_in.t())
            out = torch.empty_like(branch0)
            ns = torch.empty_like(stream)
            arms = ["sgl"] + [a for a in ARMS] + ["sgl2"]
            for ctx in ("solo", "chain"):
                graphs = {}
                for arm in arms:
                    def fn(_d, arm=arm, ctx=ctx):
                        br = torch.mm(xin, w_in.t()) if ctx == "chain" else branch0
                        if arm.startswith("sgl"):
                            o = run_sgl(which, br, stream)[0]
                        else:
                            run_bx(which, arm, br, stream, out, ns)
                            o = out
                        if ctx == "chain" and which == "attn":
                            return torch.mm(o, wcon.t())
                        return o
                    graphs[arm] = timer.capture(fn, {})[0]
                tt = {n: [] for n in arms}
                for rd in range(rounds):
                    for n in arms[rd % len(arms):] + arms[:rd % len(arms)]:
                        tt[n].append(timer.time_replay(graphs[n]) - ov)
                med = {n: statistics.median(v) for n, v in tt.items()}
                sg = (med["sgl"] + med["sgl2"]) / 2
                res["timing"][f"{which} T{T} {ctx}"] = {"med": med, "raw": tt}
                log(f"[bndx] T={T} {which} {ctx}: " + " ".join(f"{n} {med[n]:.2f}" for n in arms) + " us | vs sgl: "
                    + " ".join(f"{n} {med[n] - sg:+.2f}" for n in arms if not n.startswith("sgl")) + f" | sgl2-sgl {med['sgl2'] - med['sgl']:+.2f}")
                outp.write_text(json.dumps(res, indent=1))
                del graphs

    # ---- decision (pre-registered, notes/memory/gpu_requests.md R-bndx-1)
    def gain(T, which, ctx, arm):
        m = res["timing"][f"{which} T{T} {ctx}"]["med"]
        return (m["sgl"] + m["sgl2"]) / 2 - m[arm]
    verdict = "INVALID (bitwise)" if not allok else None
    if verdict is None and 256 in ts:
        cand = [arm for arm in ARMS if arm != "bxT" and not any(arm in k for k in bad)]
        best = max(cand, key=lambda arm: gain(256, "attn", "chain", arm) + gain(256, "ffn", "chain", arm))
        g256 = gain(256, "attn", "chain", best) + gain(256, "ffn", "chain", best)
        g128 = (gain(128, "attn", "chain", best) + gain(128, "ffn", "chain", best)) if 128 in ts else 0.0
        ctl = all(abs(res["timing"][k]["med"]["sgl2"] - res["timing"][k]["med"]["sgl"]) <= 0.4 for k in res["timing"])
        if not ctl:
            verdict = "INVALID (sgl2 - sgl > 0.4 us)"
        elif g256 >= 3.0 and g128 >= -0.3:
            verdict = f"GO ({best}): in-layer A/B at v16s / v32s"
        elif g256 < 1.6:
            verdict = f"KILL (best {best}: {g256:+.2f} us at T=256, both boundaries)"
        else:
            verdict = f"BETWEEN ({best}: {g256:+.2f} us)"
        res["best"] = {"arm": best, "gain256": g256, "gain128": g128, "controls_ok": ctl}
        log(f"[bndx] chain gains (both boundaries) T=256 {g256:+.2f} us, T=128 {g128:+.2f} us with {best}; controls {ctl}")
    res["verdict"] = verdict
    outp.write_text(json.dumps(res, indent=1))
    log(f"[bndx] verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

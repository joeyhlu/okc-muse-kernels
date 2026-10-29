"""midr_lab.py: MIDR step 0 (notes/study/fresh_eyes_nonmoe.md s2, R-midr-0): what the attention boundary's FP32 mid-layer
stream store costs at T >= 128 in the graph [o-proj GEMM (cuBLAS) -> attention boundary -> cat GEMM (fp32 out)].

Arms (one CUDA graph each, judge-style replay: 256 MB L2 flush + 300 us gate via tools/judge GpuTimer, rotating order):
  oc      o GEMM + cat GEMM on a static input (no boundary; reference)
  sgl     SGLang's FusedPostNormResidualPreNorm module (the layer's path at T >= 128), twice (sgl, sgl2 = position control)
  vend    SGLang norm_residual.cuh vendored verbatim (own JIT module, same template args / launch config as the module)
  nost    the vendored copy with the new_stream store compiled out (both kernels; MIDR's attention-boundary bytes)
Correctness: every arm's bf16 output vs the module (nost must still match: only the stream write is gone); vend's stream bitwise.
GO if sgl - nost >= 1.5 us at T = 256 (>= 0.5 at T = 128); KILL if < 0.8 us at T = 256.

    python3 -u kernels/lab/midr_lab.py [--ts=128,256] [--rounds=40] [--out=box_out/midr/lab.json]
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = 8192
STORES = ("      new_stream_v[j].store(new_stream, idx * kFParts + j);\n",
          "      new_stream_values[part].store(new_stream, float_index);\n")


def vendored(nr, kind, outdir):
    """(build fn(m) -> module) for the vendored copy `kind` in {'vend', 'nost'} of SGLang's norm_residual.cuh"""
    import sglang.kernels.jit as SJ
    src = (Path(SJ.__file__).parent / "csrc" / "muse_spark_v12" / "norm_residual.cuh").read_text()
    if kind == "nost":
        for s in STORES:
            assert src.count(s) == 1, s
            src = src.replace(s, "      { /* MIDR step 0: new_stream store compiled out (empty loop body) */ }\n")
    f = outdir / f"norm_residual_{kind}.cuh"
    f.write_text(src)
    cache = {}

    def get(m):
        nt, pf = nr._cluster_launch_config(H, m, 1)
        key = (nt, pf)
        if key not in cache:
            args = nr.make_cpp_args(H, nt, nr.is_arch_support_pdl(), pf, 1)
            cache[key] = nr.load_jit(f"muse_spark_v12_norm_residual_midr_{kind}", *args, cuda_files=[str(f.resolve())],
                                     cuda_wrappers=[("run_std", f"MuseSparkNormResidual<{args}, false, false>::run"),
                                                    ("run_scale", f"MuseSparkNormResidual<{args}, true, false>::run")])
        return cache[key]
    return get


def main():
    import torch
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    import judge as J
    import sglang.kernels.ops.muse_spark_v12.norm_residual as NR
    a = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    ts = [int(x) for x in a.get("ts", "128,256").split(",")]
    rounds = int(a.get("rounds", 40))
    outp = Path(a.get("out", "box_out/midr/lab.json"))
    outp.parent.mkdir(parents=True, exist_ok=True)
    prob = J.load_problem(ROOT / "problems/muse_decode_layer")
    ref = prob.ref
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    base = J.load_kernel(prob.dir / "baseline.py", "okc_baseline", prob)
    st = base.prepare(J.clone_data(weights))
    op = st["attn_boundary"]
    op._ensure_cached_weights(torch.bfloat16)
    pw = op._cc_post_weight if op._post_norm.has_weight else None
    qw = op._cc_pre_weight if op._pre_norm.has_weight else None
    al = op._cc_alpha if op._use_residual_gate else None
    bt = op._cc_beta if op._use_residual_gate else None
    isc, pe, qe = bool(op._post_norm_is_input_scale), float(op._post_norm.eps), float(op._pre_norm.eps)
    mods = {k: vendored(NR, k, outp.parent) for k in ("vend", "nost")}
    timer = J.GpuTimer()
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(78)
    wo = weights["o"]
    wcat = (torch.randn((4608, H), generator=gen, device=dev) * 0.02).to(torch.bfloat16)
    res = {"rounds": rounds, "shapes": {}}
    for T in ts:
        xin = (torch.randn((T, wo.shape[1]), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        stream = torch.randn((T, H), generator=gen, device=dev) * 2
        static_in = torch.randn((T, H), generator=gen, device=dev).to(torch.bfloat16)
        br0 = torch.mm(xin, wo.t())
        r_out, r_ns = op(stream, br0, branch_dtype=torch.bfloat16, router_dtype=None)[:2]

        def bnd(arm, branch):
            if arm in ("sgl", "sgl2"):
                return op(stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)[:2]
            out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
            ns = torch.empty((T, H), dtype=torch.float32, device=dev)
            fn = getattr(mods[arm](T), "run_scale" if isc else "run_std")
            fn(branch, stream, out, ns, None, pw, qw, al, bt, pe, qe, False)
            return out, ns

        arms = ["oc", "sgl", "vend", "nost", "sgl2"]
        errs, graphs = {}, {}
        for arm in arms:
            if arm != "oc":
                o, ns = bnd(arm, br0)
                torch.cuda.synchronize()
                e_out = float((o.float() - r_out.float()).abs().max())
                e_ns = float((ns - r_ns).abs().max()) if arm != "nost" else float("nan")
                errs[arm] = (e_out, e_ns)
                print(f"[midr] T={T} {arm}: out max|d| {e_out:.3g} stream max|d| {e_ns:.3g}", flush=True)

            def fn(_d, arm=arm):
                br = torch.mm(xin, wo.t())
                x = static_in if arm == "oc" else bnd(arm, br)[0]
                return torch.mm(x, wcat.t(), out_dtype=torch.float32)
            graphs[arm] = timer.capture(fn, {})[0]
        tt = {n: [] for n in arms}
        for rd in range(rounds):
            for n in arms[rd % len(arms):] + arms[:rd % len(arms)]:
                tt[n].append(timer.time_replay(graphs[n]))
        med = {n: statistics.median(v) for n, v in tt.items()}
        sg = (med["sgl"] + med["sgl2"]) / 2
        print(f"[midr] T={T} graph median us: " + " ".join(f"{n} {med[n]:.2f}" for n in arms), flush=True)
        print(f"[midr] T={T} store cost (sgl - nost) {sg - med['nost']:+.2f} us | vend - nost {med['vend'] - med['nost']:+.2f} | "
              f"sgl2 - sgl {med['sgl2'] - med['sgl']:+.2f} | boundary inc (sgl - oc) {sg - med['oc']:+.2f}", flush=True)
        res["shapes"][T] = {"med": med, "errs": errs, "raw": tt}
        outp.write_text(json.dumps(res, indent=1))
        del graphs
    d256 = None
    if 256 in res["shapes"]:
        m = res["shapes"][256]["med"]
        d256 = (m["sgl"] + m["sgl2"]) / 2 - m["nost"]
        print(f"[midr] verdict T=256 store cost {d256:+.2f} us -> "
              f"{'GO' if d256 >= 1.5 else ('KILL' if d256 < 0.8 else 'PARK')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

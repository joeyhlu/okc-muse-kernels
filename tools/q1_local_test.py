"""Local correctness test of the Q1 one-launch MoE lab kernel (kernels/lab/q1_moe_lab.py) on the RTX 3070 (sm_86 build:
cp.async producer instead of bulk copies; same schedule, consumer, epilogue and finalize).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/q1_local_test.py" [sm_86|sm_90a-compile]

32 local experts (weights from the problem's nvfp4.random_weights), K2A's own prepack / stage repack, FP32 reference
(K2A's _k2a_ref_fp32). Checks: rel error, worst row, error flags, repeat calls, graph replay with changed inputs.
"""
import importlib.util
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import nvfp4  # noqa: E402
import q1_moe_lab as Q  # noqa: E402


def load_board():
    spec = importlib.util.spec_from_file_location("k2a_board", ROOT / "kernels/muse_moe_w4a16/k0_check.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_weights(dev, seed=5252):
    g = torch.Generator(device=dev).manual_seed(seed)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(32,))
    w2, w2_sf, gs2 = nvfp4.random_weights(4096, 4096, generator=g, device=dev, leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
            "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device=dev)).to(torch.bfloat16)}


def repack(K, w, dev):
    ws13 = torch.empty((32, 64, 16, 18432), dtype=torch.uint8, device=dev)
    ws2 = torch.empty((32, 32, 16, 18432), dtype=torch.uint8, device=dev)
    for e in range(32):
        a, b = K._k2a31_prepack_expert(w["w13"][e], w["w13_sf"][e], gate_up=True, bitpos=K.K2A31_BITPOS3)
        ws13[e] = K._k2a33_stage_repack(a, b)
        a, b = K._k2a31_prepack_expert(w["w2"][e], w["w2_sf"][e], gate_up=False, bitpos=K.K2A31_BITPOS3)
        ws2[e] = K._k2a33_stage_repack(a, b)
    return {"ws13": ws13, "ws2": ws2,
            "gsc13": (w["w13_gs"].float() * 256.0).contiguous(), "gsc2": (w["w2_gs"].float() * 256.0).contiguous(),
            "nw": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()}


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
    if arch == "sm_90a-compile":
        mod, dt = Q.q1_build("sm_90a")
        print(f"sm_90a build ok in {dt:.0f}s (cannot run here)")
        return
    dev = torch.device("cuda")
    K = load_board()
    w = make_weights(dev)
    st = repack(K, w, dev)
    torch.cuda.synchronize()
    if arch == "lab":
        # harness dry run: the judge lab end to end, with the Q kernel itself standing in for K2A (no sm_90 here)
        mod, _ = Q.q1_build("sm_86")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        G = sms * max(1, mod.q1_attrs(3)[3])
        kb = Q.q1_bufs(dev, G)
        Q.q1_lab(st, lambda d, s: Q.q1_fn(mod, st, kb, d, 3, G, 0)(), None, rounds=4, inputs=2, sizes=(1, 8), arch="sm_86")
        return
    mod, dt = Q.q1_build("sm_86")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    for cfg in Q.Q1_CFGS:
        print(f"cfg {cfg} attrs (regs, local, smem, occ) {list(mod.q1_attrs(cfg))}")
    cfg = 3
    occ = max(1, mod.q1_attrs(cfg)[3])
    G = sms * occ
    print(f"built in {dt:.0f}s; {sms} SMs, occ {occ}, G {G}")
    bufs = Q.q1_bufs(dev, G)
    worst = 0.0
    cases = [(1, 11, "normal"), (1, 12, "normal"), (1, 13, "skewed"), (2, 14, "normal"), (3, 15, "normal"),
             (4, 16, "normal"), (5, 17, "normal"), (8, 18, "normal"), (8, 19, "skewed"), (8, 20, "normal"),
             (7, 21, "skewed"), (6, 22, "normal")]
    for bs, seed, dist in cases:
        d = Q.q1_case(bs, seed, dev, dist)
        ref = K._k2a_ref_fp32(d, w)
        for k in range(2):
            bufs["err"].zero_()
            out = Q.q1_fn(mod, st, bufs, d, cfg, G, 0)()
            torch.cuda.synchronize()
            tot, row = K._k2a_rel_err(out, ref)
            e = bufs["err"].tolist()[:2]
            fin = bool(torch.isfinite(out.float()).all())
            print(f"bs={bs} seed={seed} {dist} experts {Q.q1_touched(d)} call {k}: rel {tot:.3e} row {row:.3e} err {e} "
                  f"finite {fin}")
            worst = max(worst, tot if fin and e[0] == 0 else 1.0)
    # NST 4 and the null / stream modes must run without error flags
    for c2, md in ((4, 0), (3, 1), (3, 15)):
        bufs2 = Q.q1_bufs(dev, G)
        d = Q.q1_case(8, 31, dev)
        out = Q.q1_fn(mod, st, bufs2, d, c2, G, md)()
        torch.cuda.synchronize()
        e = bufs2["err"].tolist()[:2]
        msg = ""
        if md == 0:
            tot, row = K._k2a_rel_err(out, K._k2a_ref_fp32(d, w))
            msg = f" rel {tot:.3e}"
            worst = max(worst, tot)
        print(f"cfg {c2} mode {md}: err {e}{msg}")
    # graph replay with changed inputs (same shapes)
    d = Q.q1_case(8, 41, dev)
    f = Q.q1_fn(mod, st, bufs, d, cfg, G, 0)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        f()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            out = f()
    torch.cuda.synchronize()
    x_buf = d["hidden_states"]
    for k in range(3):
        d2 = Q.q1_case(8, 50 + k, dev)
        # f captured d's tensors via q1_fn's contiguous() (same storage): copy new values in place
        x_buf.copy_(d2["hidden_states"])
        d["topk_ids"].copy_(d2["topk_ids"])
        d["topk_weights"].copy_(d2["topk_weights"])
        gr.replay()
        torch.cuda.synchronize()
        tot, row = K._k2a_rel_err(out, K._k2a_ref_fp32(d, w))
        print(f"graph replay {k}: experts {Q.q1_touched(d)} rel {tot:.3e} row {row:.3e} err {bufs['err'].tolist()[:2]}")
        worst = max(worst, tot)
    print(f"WORST rel {worst:.3e} -> {'PASS' if worst < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()

"""Local correctness test of the Z1 slice-chained MoE lab kernel (kernels/lab/z1_moe_lab.py) on the RTX 3070
(sm_86 build: cp.async, 1-stage ring, <= 2 tokens, 2-tile sub-groups, no PDL).
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/z1_local_test.py"
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import q1_local_test as L  # noqa: E402  (weights, board import, K2A repack)
import z1_moe_lab as Z  # noqa: E402


def main():
    dev = torch.device("cuda")
    K = L.load_board()
    w = L.make_weights(dev)
    st = L.repack(K, w, dev)
    ws13t, ws2s = Z.z1_repack(st, w["w2"], w["w2_sf"], K._k2a_reencode_scales, K._k2a31_pack_words, K.K2A31_BITPOS3)
    lay = {"ws13t": ws13t, "ws2s": ws2s}
    torch.cuda.synchronize()
    mod, dt = Z.z1_build("sm_86")
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    a = list(mod.z1_attrs())
    print(f"built in {dt:.0f}s; attrs (regs, local, smem, occ) {a}; {sms} SMs")
    G = sms
    bufs = Z.z1_bufs(dev, G)
    worst = 0.0
    for bs, seed, dist in [(1, 11, "normal"), (1, 12, "normal"), (1, 13, "skewed"), (1, 21, "normal"), (2, 14, "normal"),
                           (2, 15, "skewed"), (2, 16, "normal"), (1, 17, "normal")]:
        d = Z.z1_case(bs, seed, dev, dist)
        ref = K._k2a_ref_fp32(d, w)
        for k in range(2):
            bufs["err"].zero_()
            out = Z.z1_fn(mod, st, lay, bufs, d, G, 0)()
            torch.cuda.synchronize()
            tot, row = K._k2a_rel_err(out, ref)
            e = bufs["err"].tolist()[:2]
            fin = bool(torch.isfinite(out.float()).all())
            yz = float(bufs["y"].abs().max())
            print(f"bs={bs} seed={seed} {dist} experts {Z.z1_touched(d)} call {k}: rel {tot:.3e} row {row:.3e} err {e} "
                  f"finite {fin} y-after {yz:.1e}")
            worst = max(worst, tot if (fin and e[0] == 0 and yz == 0.0) else 1.0)
    for md in (1, 9):
        b2 = Z.z1_bufs(dev, G)
        d = Z.z1_case(2, 31, dev)
        Z.z1_fn(mod, st, lay, b2, d, G, md)()
        torch.cuda.synchronize()
        print(f"mode {md}: err {b2['err'].tolist()[:2]}")
    d = Z.z1_case(2, 41, dev)
    f = Z.z1_fn(mod, st, lay, bufs, d, G, 0)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        f()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr, stream=s):
            out = f()
    torch.cuda.synchronize()
    for k in range(3):
        d2 = Z.z1_case(2, 50 + k, dev)
        d["hidden_states"].copy_(d2["hidden_states"])
        d["topk_ids"].copy_(d2["topk_ids"])
        d["topk_weights"].copy_(d2["topk_weights"])
        gr.replay()
        torch.cuda.synchronize()
        tot, row = K._k2a_rel_err(out, K._k2a_ref_fp32(d, w))
        print(f"graph replay {k}: experts {Z.z1_touched(d)} rel {tot:.3e} row {row:.3e} err {bufs['err'].tolist()[:2]}")
        worst = max(worst, tot)
    print(f"WORST rel {worst:.3e} -> {'PASS' if worst < 1e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()

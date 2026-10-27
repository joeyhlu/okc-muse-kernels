"""Local correctness of the CR full call (K1 -> K2 -> K3) on the RTX 3070 (sm_86, no PDL), vs the FP32 reference.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/kernels/lab/cr_full_test.py"
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import cr_k1_lab as L  # noqa: E402
import nvfp4  # noqa: E402


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(5252)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(32,))
    w2, w2_sf, gs2 = nvfp4.random_weights(4096, 4096, generator=g, device=dev, leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
         "w2_gs": gs2.expand(32).contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device=dev)).to(torch.bfloat16)}
    mod, _ = L.cr_build(arch="sm_86")
    ws, gsc = L.cr_prepack(w["w13"], w["w13_sf"], w["w13_gs"])
    ws2, gsc2 = L.cr_prepack_w2(w["w2"], w["w2_sf"], w["w2_gs"])
    nw = (w["post_norm_w"].float() + 1.0).to(torch.bfloat16).float().contiguous()
    G = torch.cuda.get_device_properties(0).multi_processor_count
    st = L.CrState(mod, ws, gsc, G, dev)
    full = L.CrFull(st, ws2, gsc2, nw, pdl=False)
    allok = True
    for bs, seed in ((1, 11), (3, 13), (8, 14), (32, 15), (128, 16), (256, 17), ("L32", 0)):
        if isinstance(bs, str):
            d = L.cr_case_counts(int(bs[1:]), L.CR_LAYER_COUNTS[int(bs[1:])][seed], 5150 + seed, dev)
        else:
            d = L.cr_case(bs, 660000 + seed, dev)
        P1 = L.cr_plan(d, G)
        P2 = L.cr_plan_w2(d, P1, G)
        rel, worst, err, cl, nan, B = L.cr_check_full(full, d, w, P1, P2, L.cr_dequant)
        out1 = B["out"].clone()
        rel2, _, err2, _, _, B2 = L.cr_check_full(full, d, w, P1, P2, L.cr_dequant)
        rep = float((out1.float() - B2["out"].float()).norm() / out1.float().norm().clamp(min=1e-30))
        ok = rel <= 1e-2 and worst <= 5e-2 and err[0] == 0 and cl == 0 and not nan and rep <= 1e-3
        allok &= ok
        print(f"full bs={bs}: experts {len(P1['experts'])} R {P1['R']} | rel {rel:.2e} worst {worst:.2e} err {err} "
              f"cnt-clean {cl == 0} nan {nan} repeat {rep:.1e} -> {'OK' if ok else 'FAIL'}")
    print("ALL OK" if allok else "SOME FAILED")


if __name__ == "__main__":
    main()

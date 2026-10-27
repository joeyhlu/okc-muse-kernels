"""Local correctness test of the VMS one-launch T <= 16 MoE lab kernel (kernels/lab/vms_moe_lab.py) on the RTX 3070
(sm_86 build: cp.async ring, NS = 2, grid = SM count), or a compile-only sm_90a check.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vms_local_test.py" [sm_86|sm_90a]

Checks: rel vs FP32 reference (K2A level ~3.7e-3), err flag 0, repeat calls (state resets), graph replay with changed
inputs, T = 1..16 incl. hot experts (> 8 tokens -> split jobs), non-local (-1) routes, tokens without local routes.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import nvfp4  # noqa: E402
import vms_moe_lab as V  # noqa: E402


def make_weights(dev, seed=5252, E=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    w13, w13_sf, gs13 = nvfp4.random_weights(8192, 4096, generator=g, device=dev, leading=(E,))
    w2, w2_sf, gs2 = nvfp4.random_weights(4096, 4096, generator=g, device=dev, leading=(E,))
    gs13 = gs13.expand(E) * torch.exp(torch.randn((E,), generator=g, device=dev))
    gs2 = gs2.expand(E) * torch.exp(torch.randn((E,), generator=g, device=dev))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.contiguous(), "w2": w2, "w2_sf": w2_sf,
            "w2_gs": gs2.contiguous(), "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device=dev)).to(torch.bfloat16)}


def make_input(T, seed, dev, kind):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn((T, 4096), generator=g, device=dev).to(torch.bfloat16)
    sc = torch.rand((T, 64), generator=g, device=dev)
    if kind == "hot":
        sc[:, :3] += 10.0
    elif kind == "skew":
        sc = sc + torch.rand((64,), generator=g, device=dev).pow(4) * 3
    top = sc.topk(8, dim=-1)
    ids = top.indices.to(torch.int32)
    ids = torch.where(ids < 32, ids, torch.full_like(ids, -1))      # ~half the routes non-local
    if kind == "nolocal" and T > 1:
        ids[0] = -1
    tw = (top.values / top.values.sum(-1, keepdim=True)).float()
    return x, ids.contiguous(), tw.contiguous()


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
    dev = "cuda"
    if arch == "sm_90a":
        mod, dt = V.build(132, 3, "sm_90a", name="vms_lab90")
        print(f"[vms] sm_90a build ok in {dt:.0f} s (compile only; attrs need an H200)")
        return 0
    G = torch.cuda.get_device_properties(0).multi_processor_count
    mod, dt = V.build(G, 2, "sm_86", name="vms_lab86")
    print(f"[vms] sm_86 build ok in {dt:.0f} s: regs/local/smem/G/NS {list(mod.attrs())}")
    w = make_weights(dev)
    pk = V.prepack(w)
    sc = V.scratch(dev)
    ok_all = True
    cases = [(1, 1, "normal"), (1, 2, "normal"), (2, 3, "normal"), (4, 4, "skew"), (5, 5, "normal"), (8, 6, "normal"),
             (8, 7, "hot"), (12, 8, "skew"), (16, 9, "normal"), (16, 10, "hot"), (16, 11, "skew"), (3, 12, "nolocal"),
             (16, 13, "nolocal")]
    for T, seed, kind, fl in [c + (fl,) for fl in (0, 12) for c in cases]:
        x, ids, tw = make_input(T, seed, dev, kind)
        ref = V.ref_fp32(x, ids, tw, w)
        cnt = torch.bincount(ids[ids >= 0].long(), minlength=32)
        for k in range(3):
            o = V.call(mod, pk, sc, x, ids, tw, fl=fl)
            torch.cuda.synchronize()
            tot, row = V.rel_err(o, ref)
            err = int(sc["ctr"][2])
            good = tot < 6e-3 and row < 3e-2 and err == 0 and bool(torch.isfinite(o.float()).all())
            ok_all &= good
            if k == 0 or not good:
                print(f"[vms] fl={fl:2d} T={T:2d} {kind:7s} experts {int((cnt > 0).sum())} max {int(cnt.max())} call {k}: rel {tot:.3e} "
                      f"row {row:.3e} err {err} gen {int(sc['ctr'][0])} bar {int(sc['ctr'][1])} -> {'ok' if good else 'FAIL'}")
    # graph replay with changed inputs
    for T in (1, 8, 16):
        x, ids, tw = make_input(T, 100 + T, dev, "normal")
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        V.call(mod, pk, sc, x, ids, tw, out=out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            V.call(mod, pk, sc, x, ids, tw, out=out)
        for r in range(3):
            x2, ids2, tw2 = make_input(T, 200 + 10 * T + r, dev, "hot" if r == 1 else "normal")
            x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
            gr.replay()
            torch.cuda.synchronize()
            ref = V.ref_fp32(x2, ids2, tw2, w)
            tot, row = V.rel_err(out, ref)
            err = int(sc["ctr"][2])
            good = tot < 6e-3 and row < 3e-2 and err == 0
            ok_all &= good
            print(f"[vms] graph T={T:2d} replay {r}: rel {tot:.3e} row {row:.3e} err {err} -> {'ok' if good else 'FAIL'}")
    print(f"[vms] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

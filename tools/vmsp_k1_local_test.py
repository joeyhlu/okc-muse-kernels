"""Local test of VMSP step 1 (kernels/lab/vmsp_k1.py, VMSP_MATH 2) on the decode_layer reference weights.
  sm90=1: sm_90a compile-only: ptxas registers / spills (MATH 2, OBS 0 / 1) and the ACQBULK SASS gate (no global load above
          griddepcontrol.wait; >= 1 UBLKCP above it).
  sm86=1: RTX 3070 (G = 46, NS 2): per case, two calls + stale-input graph replays (inputs copied on the replay stream):
          router partials vs FP32 slices, routing = host top-8 on the kernel's own partials (tail2's arithmetic), pooler y vs
          bf16(h @ W^T) (fraction of bf16 values that differ, max ulps), MoE output vs FP32 on xn = bf16((y * rstd) * w),
          error flag, barrier, counter reset.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vmsp_k1_local_test.py" [sm90=1] [sm86=1]
"""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in ("kernels/lab", "tools", "tools/judge", "problems/muse_moe_w4a16"):
    sys.path.insert(0, str(ROOT / p))
import vmsp_k1 as K  # noqa: E402
import vmsp_local_test as L0  # noqa: E402

L0.K = K


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ok_all = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            for math, obs in ((2, 0), (2, 1)):
                rc, ln, info = L0.sm90(math, obs, td)
                print(f"[vmsp1] sm_90a MATH {math} OBS {obs}: rc {rc} | {ln} | SASS {info}", flush=True)
                ok_all &= rc == 0 and info is not None and not info["loads_above"] and info["ublkcp_above"] >= 1
    if not int(opt.get("sm86", 1)):
        print(f"[vmsp1] {'ALL PASS' if ok_all else 'FAILURES'}")
        return 0 if ok_all else 1
    import torch
    import judge as J
    import vms_moe_lab_v1 as V
    V.POS = K.VMS_POS
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    mods = {}
    for obs in (0, 1):
        mods[obs], dt = K.build(G, ns=2, math=2, obs=obs, arch="sm_86", name=f"vmsp1_l{obs}")
        print(f"[vmsp1] sm_86 MATH 2 OBS {obs} built {dt:.0f} s attrs {list(mods[obs].attrs())}", flush=True)
    prob = J.load_problem(ROOT / "problems/muse_decode_layer")
    ref = prob.ref
    w = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=8677534258059892491)
    wx = {"w13": w["w13"], "w13_sf": w["w13_sf"], "w13_gs": w["w13_gs"], "w2": w["w2"], "w2_sf": w["w2_sf"],
          "w2_gs": w["w2_gs"], "post_norm_w": w["post_expert_norm"]}
    vpk = V.prepack(wx, log=lambda s: None)
    nwp = (1.0 + w["pre_expert_norm"].float()).to(torch.bfloat16)
    router = w["router"].float().contiguous()
    pool = w["pooler"].to(torch.bfloat16).contiguous()
    pk = K.prepare_real(router, w["router_bias"], pool, nwp, vpk, G)
    print(f"[vmsp1] pack: {pk['nesc']} escapes, tb {pk['tb']:#x}", flush=True)

    def make_case(T, seed, ls, hot):
        g = torch.Generator(device=dev).manual_seed(seed)
        h32 = torch.randn((T, 8192), generator=g, device=dev)
        h32 = h32 * torch.rsqrt(h32.pow(2).mean(-1, keepdim=True)) * (1.0 + w["pre_ffn_norm"].float())
        if hot:
            sel = (ls + torch.tensor([1, 5, 9], device=dev)) % 256
            h32 = h32 + 3.0 * router[sel].sum(0)[None, :] / router[sel].norm(dim=1).mean().pow(2)
        rx = h32.contiguous()
        hb = rx.to(torch.bfloat16).contiguous()
        return hb, rx, torch.tensor([ls], dtype=torch.int32, device=dev)

    def check(tag, res, sc, hb, rx, lst):
        out, tw, tid, lid = (x.clone() for x in res)
        T, ls = hb.shape[0], int(lst[0])
        part = sc["part"][:8 * T * 256].view(8, T, 256).clone()
        fp = K.fake_partials(rx, router)
        prel = float((part - fp).abs().max() / fp.abs().max())
        rtid, rtw, rlid, bsc = K.ref_routing(part, pk["bias"], ls)
        same = bool((tid.sort(-1).values == rtid.sort(-1).values).all())
        tw_ok = bool(torch.allclose(tw.gather(1, torch.argsort(tid, -1)), rtw.gather(1, torch.argsort(rtid, -1)), rtol=1e-6, atol=1e-7))
        lid_ok = bool((lid == torch.where(((tid - ls) % 256) < 32, (tid - ls) % 256, torch.full_like(tid, -1))).all())
        yref = (hb.float() @ pool.float().T).to(torch.bfloat16)
        y = sc["y"][:T]
        nd = int((y != yref).sum())
        dif = (y.float() - yref.float()).abs()
        ulp = int(((dif > yref.float().abs() * 2 ** -7) & (dif > 1e-3 * yref.float().pow(2).mean(-1, keepdim=True).sqrt())).sum())
        xn = K.ref_xn(torch.cat((y, torch.zeros_like(y)), 1), nwp)   # tail2's arithmetic on the kernel's y
        e, row = V.rel_err(out, V.ref_fp32(xn, lid, tw, wx))
        c = sc["ctr"]
        err, coh, rst = int(c[2]), int(c[1]) == int(c[0]) * G, int(c[3]) == 0 and int(c[4]) == 0 and int(c[5]) == 0
        good = (err == 0 and coh and rst and same and tw_ok and lid_ok and prel < 1e-5 and nd <= max(8, T * 4096 // 200)
                and ulp == 0 and e < 1e-2 and row < 5e-2)
        print(f"[vmsp1] {tag}: T={T} routes {int((lid >= 0).sum())} part rel {prel:.1e} ids {same} tw {tw_ok} lid {lid_ok} "
              f"y diff {nd}/{T * 4096} beyond-1ulp {ulp} out rel {e:.2e} row {row:.2e} err {err} bar {coh} reset {rst} -> "
              f"{'ok' if good else 'FAIL'}", flush=True)
        return good

    cases = [(1, 11, 0, False), (1, 12, 64, True), (2, 13, 96, True), (5, 14, 224, False), (8, 15, 32, True), (8, 16, 0, False)]
    for obs, mod in mods.items():
        sc = K.scratch(dev, G)
        for T, seed, ls, hot in cases:
            hb, rx, lst = make_case(T, seed, ls, hot)
            for k in range(2):
                res = K.call(mod, pk, sc, hb, rx, lst, marks=obs == 1, lmarks=obs == 1, pdl=False)
                torch.cuda.synchronize()
                ok_all &= check(f"OBS{obs} seed {seed} call {k}", res, sc, hb, rx, lst)
        for T in (1, 8):
            hb, rx, lst = make_case(T, 500 + T, 32, True)
            outs = tuple(torch.empty_like(x) for x in K.call(mod, pk, sc, hb, rx, lst, pdl=False))
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                K.call(mod, pk, sc, hb, rx, lst, pdl=False, outs=outs)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=st):
                    K.call(mod, pk, sc, hb, rx, lst, pdl=True, outs=outs)
            torch.cuda.synchronize()
            for rr in range(3):
                hb2, rx2, lst2 = make_case(T, 600 + 10 * T + rr, (32 * (rr + 2)) % 256, rr != 1)
                with torch.cuda.stream(st):
                    hb.copy_(hb2); rx.copy_(rx2); lst.copy_(lst2)
                    gr.replay()
                torch.cuda.synchronize()
                ok_all &= check(f"OBS{obs} stale replay T={T} #{rr}", outs, sc, hb2, rx2, lst2)
    print(f"[vmsp1] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

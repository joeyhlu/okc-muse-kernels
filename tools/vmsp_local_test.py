"""Local test of the VMSP skeleton (kernels/lab/vmsp_skel.py).
  sm90=1: sm_90a compile-only (WSL nvcc 13.4): ptxas registers / spills per arm (MATH 0 / 1, OBS 0 / 1) and the PDL SASS gate:
          every instruction above griddepcontrol.wait (ACQBULK) is listed; no global load may sit above it (only the
          pre-wait bulk copies, mbarrier init and constants).
  sm86=1: RTX 3070 (sm_86 cp.async ring, NS 2, G = 46): protocol / counters / routing / x stage / experts per case, two calls
          each, then a CUDA-graph stale-input stress (capture on input A, copy input B into the static buffers ON THE REPLAY
          STREAM, replay once, check against B).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/vmsp_local_test.py" [sm90=1] [sm86=1]
"""
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in ("kernels/lab", "tools", "problems/muse_moe_w4a16"):
    sys.path.insert(0, str(ROOT / p))
import vmsp_skel as K  # noqa: E402

CUOBJDUMP = "/root/cudaenv/lib/python3.14/site-packages/triton/backends/nvidia/bin/cuobjdump"


def sm90(math, obs, td):
    k = pathlib.Path(td) / f"vmsp_{math}{obs}.cu"
    k.write_text(K.CUDA_SRC.split("#include <torch/extension.h>")[0], encoding="utf-8")
    cub = pathlib.Path(td) / f"vmsp_{math}{obs}.cubin"
    fl = [f for f in K.flags(132, 3, math, obs) if not f.startswith("-Xptxas")]
    r = subprocess.run(["nvcc"] + fl + ["-Xptxas=-v,-warn-spills", "-cubin", str(k), "-o", str(cub)], capture_output=True, text=True)
    lines = [ln.strip() for ln in (r.stdout + r.stderr).splitlines() if "registers" in ln or "spill" in ln or "error" in ln]
    if r.returncode:
        return r.returncode, (r.stdout + r.stderr)[-3000:], None
    sass = subprocess.run([CUOBJDUMP, "-sass", str(cub)], capture_output=True, text=True).stdout
    ins = []
    for ln in sass.splitlines():
        mm = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?\s*([^;]*);", ln)
        if mm:
            ins.append((mm.group(3), (mm.group(4) or ""), mm.group(5)))
    acq = [i for i, x in enumerate(ins) if x[0] == "ACQBULK"]
    above = ins[:acq[0]] if acq else ins
    loads = [f"{op}{mod}" for op, mod, _ in above if op.startswith("LD") and op not in ("LDS", "LDSM", "LDC", "LDCU")]
    ublk = sum(1 for op, _, _ in above if op.startswith("UBLKCP"))
    return 0, " | ".join(lines[-2:]), {"acqbulk": len(acq), "above": len(above), "loads_above": loads, "ublkcp_above": ublk}


def main():
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    ok_all = True
    if int(opt.get("sm90", 1)):
        with tempfile.TemporaryDirectory() as td:
            for math, obs in ((0, 0), (1, 0), (0, 1), (1, 1)):
                rc, ln, info = sm90(math, obs, td)
                print(f"[vmsp] sm_90a MATH {math} OBS {obs}: rc {rc} | {ln} | SASS {info}", flush=True)
                ok_all &= rc == 0 and info is not None and not info["loads_above"] and info["ublkcp_above"] >= 1
    if not int(opt.get("sm86", 1)):
        print(f"[vmsp] {'ALL PASS' if ok_all else 'FAILURES'}")
        return 0 if ok_all else 1
    import torch
    import vms_moe_lab_v1 as V
    from vms_local_test import make_weights
    V.POS = K.VMS_POS
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    mods = {}
    for math, obs in ((0, 0), (1, 1)):
        mods[(math, obs)], dt = K.build(G, ns=2, math=math, obs=obs, arch="sm_86", name=f"vmsp_l{math}{obs}")
        print(f"[vmsp] sm_86 MATH {math} OBS {obs} built {dt:.0f} s attrs {list(mods[(math, obs)].attrs())}", flush=True)
    w = make_weights(dev)
    vpk = V.prepack(w, log=lambda s: None)
    gen = torch.Generator(device=dev).manual_seed(99)
    router = (torch.randn((256, 8192), generator=gen, device=dev) * 0.02).contiguous()
    bias = (torch.randn((256,), generator=gen, device=dev) * 0.01)
    prew = (torch.randn((4096,), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
    pk = K.prepare(router, bias, prew, vpk, dev)

    def make_case(T, seed, ls, hot):
        g = torch.Generator(device=dev).manual_seed(seed)
        h32 = torch.randn((T, 8192), generator=g, device=dev)
        if hot:   # push a few local experts up (several jobs, shared experts)
            boost = torch.zeros((256,), device=dev)
            boost[(ls + torch.tensor([1, 5, 9], device=dev)) % 256] = 60.0
            h32 = h32 + (boost @ router)[None, :] / (router.norm(dim=1).pow(2).mean() * 4)
        hb = h32.to(torch.bfloat16).contiguous()
        rx = hb.float().contiguous()
        fp = K.fake_partials(rx, router)
        return hb, rx, fp, torch.tensor([ls], dtype=torch.int32, device=dev)

    def check(nm, math, res, sc, hb, rx, fp, lst, tag):
        out, tw, tid, lid = res
        T = hb.shape[0]
        ls = int(lst[0])
        rtid, rtw, rlid, bsc = K.ref_routing(fp, pk["bias"], ls)
        same = bool((tid.sort(-1).values == rtid.sort(-1).values).all())
        gap = float((bsc.topk(8, -1).values[:, -1:] - bsc.gather(1, tid.long())).max())
        ok_lid = bool((lid == torch.where(((tid - ls) % 256) < 32, (tid - ls) % 256, torch.full_like(tid, -1))).all())
        yok = bool(torch.equal(sc["y"][:T], hb[:, :4096]))
        c = sc["ctr"]
        err, coh, rst = int(c[2]), int(c[1]) == int(c[0]) * G, int(c[3]) == 0 and int(c[4]) == 0 and int(c[5]) == 0
        if math >= 1:
            ref = V.ref_fp32(K.ref_xn(hb, pk["nwp"]), lid, tw, w)
            e, row = V.rel_err(out, ref)
            eok = e < 1e-2 and row < 5e-2 and bool(torch.isfinite(out.float()).all())
        else:
            e, row = 0.0, 0.0
            eok = bool((out == 0).all())
        good = err == 0 and coh and rst and (same or gap < 1e-6) and ok_lid and yok and eok
        print(f"[vmsp] {nm} {tag}: T={T} routes {int((lid >= 0).sum())} ids {'=' if same else f'tie gap {gap:.1e}'} lid {ok_lid} "
              f"y {yok} out rel {e:.2e} row {row:.2e} err {err} bar {'ok' if coh else 'BAD'} ctr-reset {rst} -> "
              f"{'ok' if good else 'FAIL'}", flush=True)
        return good

    cases = [(1, 11, 0, False), (1, 12, 64, True), (2, 13, 96, True), (5, 14, 224, False), (8, 15, 32, True), (8, 16, 0, False),
             (3, 17, 160, True)]
    for (math, obs), mod in mods.items():
        nm = f"M{math}O{obs}"
        sc = K.scratch(dev, G)
        for T, seed, ls, hot in cases:
            hb, rx, fp, lst = make_case(T, seed, ls, hot)
            for k in range(2):
                res = K.call(mod, pk, sc, hb, rx, lst, fpart=fp, marks=obs == 1, lmarks=obs == 1, pdl=False)
                torch.cuda.synchronize()
                ok_all &= check(nm, math, res, sc, hb, rx, fp, lst, f"case seed {seed} call {k}")
            if obs:
                r = K.analyze(sc["marks"].view(G, K.NMK).cpu().tolist(), sc["lmarks"].view(G, K.NLM).cpu().tolist(), G,
                              int(sc["marks"].view(G, K.NMK)[0, 26].item() >> 32), T)
                print(f"   marks: pool_end {r['pool_end']} g1 {r['g1_pool_to_expert_landing']} g2 {r['g2_x_minus_pool_end']} "
                      f"g3 {r['g3_slack']} rates {r['rates']}", flush=True)
        # stale-input stress: capture on A, copy B into A's buffers on the replay stream, replay once, check vs B
        for T in (1, 8):
            hb, rx, fp, lst = make_case(T, 500 + T, 32, True)
            outs = tuple(torch.empty_like(x) for x in K.call(mod, pk, sc, hb, rx, lst, fpart=fp, pdl=False))
            st = torch.cuda.Stream()
            with torch.cuda.stream(st):
                K.call(mod, pk, sc, hb, rx, lst, fpart=fp, pdl=False, outs=outs)
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, stream=st):
                    K.call(mod, pk, sc, hb, rx, lst, fpart=fp, pdl=True, outs=outs)
            torch.cuda.synchronize()
            for rr in range(3):
                hb2, rx2, fp2, lst2 = make_case(T, 600 + 10 * T + rr, (32 * (rr + 2)) % 256, rr != 1)
                with torch.cuda.stream(st):
                    hb.copy_(hb2); rx.copy_(rx2); fp.copy_(fp2); lst.copy_(lst2)
                    gr.replay()
                torch.cuda.synchronize()
                ok_all &= check(nm, math, outs, sc, hb2, rx2, fp2, lst2, f"stale replay T={T} #{rr}")
    print(f"[vmsp] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

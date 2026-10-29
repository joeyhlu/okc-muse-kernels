"""Local correctness test of XS (kernels/lab/xs_moe.py, notes/study/xc_moe.md) on the RTX 3070 (sm_86 build: cp.async ring,
grid = SM count), or a compile-only sm_90a check with ptxas -v.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/xs_local_test.py" [sm_86|sm_90a]

Checks (cases of tools/vms_local_test.py): rel vs the FP32 reference, err flag 0, repeated calls (y re-zeroing, barrier
generations), graph replay with changed inputs, T = 1..16 incl. hot experts (split jobs), non-local routes, tokens without
local routes; plus the XS W2 prepack decoded on the host against the FP32 dequant.
"""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "problems/muse_moe_w4a16"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools"))
import vms_local_test as VT  # noqa: E402
import vms_moe_lab as V  # noqa: E402
import xs_moe as X  # noqa: E402


def check_w2_layout(w, pos=None):
    """decode the XS W2 words / scales of expert 0 on the host (the kernel's r0..r3 decode) vs V.dequant"""
    e = 0
    pos = pos or V.POS
    pk = X.prepack_w2(w["w2"][e], w["w2_sf"][e], pos).cpu()
    ref = V.dequant(w["w2"][e], w["w2_sf"][e], 1.0).cpu()           # [4096 rows, 4096 K], gs = 1
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    bad = 0
    import random
    rng = random.Random(3)
    for _ in range(4000):
        gr, s, lane = rng.randrange(256), rng.randrange(256), rng.randrange(32)
        g, t = lane >> 2, lane & 3
        word = int.from_bytes(bytes(pk[gr, s, 4 * lane:4 * lane + 4].tolist()), "little")
        sc = pk[gr, s, 128 + 2 * g:128 + 2 * g + 2].tolist()
        for r in range(4):
            row = 16 * gr + g + (8 if r & 1 else 0)
            for h in range(2):
                k = 16 * s + 2 * t + h + (8 if r >= 2 else 0)
                code = sum(((word >> pos[r][h][b]) & 1) << b for b in range(4))
                mag = grid[code & 7] * (-1.0 if code & 8 else 1.0)
                scb = sc[1 if r & 1 else 0]
                val = mag * float(torch.tensor([scb], dtype=torch.uint8).view(torch.float8_e4m3fn).float())
                if abs(val - float(ref[row, k])) > 1e-6 * max(1.0, abs(float(ref[row, k]))):
                    bad += 1
    print(f"[xs] W2 layout host decode: {bad} mismatches in 32000 values")
    return bad == 0


def main():
    arch = sys.argv[1] if len(sys.argv) > 1 else "sm_86"
    dev = "cuda"
    if arch == "sm_90a":
        mod, dt = X.build(132, 3, "sm_90a", name="xs_lab90", verbose=True)
        print(f"[xs] sm_90a build ok in {dt:.0f} s (compile only)")
        return 0
    G = torch.cuda.get_device_properties(0).multi_processor_count
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    l2 = "l2" in sys.argv
    mod, dt = X.build(G, ns, "sm_86", name=f"xs_lab86_{ns}_{int(l2)}", l2=l2)
    print(f"[xs] sm_86 build ok in {dt:.0f} s: regs/local/smem/G/NS/sizeof(Smem) {list(mod.attrs())}")
    w = VT.make_weights(dev)
    ok_all = check_w2_layout(w, X.POS_L2 if l2 else None)
    pk = X.prepack(w, l2=l2)
    print(f"[xs] l2 layout {l2}")
    sc = X.scratch(dev)
    cases = [(1, 1, "normal"), (1, 2, "normal"), (2, 3, "normal"), (4, 4, "skew"), (5, 5, "normal"), (8, 6, "normal"),
             (8, 7, "hot"), (12, 8, "skew"), (16, 9, "normal"), (16, 10, "hot"), (16, 11, "skew"), (3, 12, "nolocal"),
             (16, 13, "nolocal")]
    cases += [(T, 40 + T, "skew") for T in (6, 9, 13, 15)] + [(T, 60 + T, "hot") for T in (2, 4, 10)]
    def fits(ids):   # XS keeps <= 1024 own dims per CTA (HSD): njobs * 512 / G A units of 8 dims
        cnt = torch.bincount(ids[ids >= 0].long(), minlength=32)
        nj = int(((cnt + 7) // 8).sum())
        return -(-nj * 512 // G) * 8 + 16 <= 1024, nj
    for T, seed, kind in cases:
        x, ids, tw = VT.make_input(T, seed, dev, kind)
        ref = V.ref_fp32(x, ids, tw, w)
        cnt = torch.bincount(ids[ids >= 0].long(), minlength=32)
        fit, nj = fits(ids)
        for k in range(3):
            sc["ctr"][2] = 0
            o = X.call(mod, pk, sc, x, ids, tw)
            torch.cuda.synchronize()
            tot, row = V.rel_err(o, ref)
            err = int(sc["ctr"][2])
            if not fit:   # over this GPU's per-CTA dim budget (G = SMs): the kernel must flag it (err 3)
                good = err == 3
                if k == 0:
                    print(f"[xs] T={T:2d} {kind:7s} jobs {nj}: over the per-CTA dims at G={G} -> err {err} {'ok' if good else 'FAIL'}")
                ok_all &= good
                continue
            good = tot < 6e-3 and row < 3e-2 and err == 0 and bool(torch.isfinite(o.float()).all())
            ok_all &= good
            if k == 0 or not good:
                print(f"[xs] T={T:2d} {kind:7s} experts {int((cnt > 0).sum())} max {int(cnt.max())} call {k}: rel {tot:.3e} "
                      f"row {row:.3e} err {err} gen {int(sc['ctr'][0])} bar {int(sc['ctr'][1])} -> {'ok' if good else 'FAIL'}")
        ysum = float(sc["ybuf"].abs().sum())
        if ysum != 0.0:
            ok_all = False
            print(f"[xs] T={T} y not re-zeroed after the call: |y| sum {ysum:.3e} -> FAIL")
    for T in (1, 8, 16):
        x, ids, tw = VT.make_input(T, 100 + T, dev, "skew")
        sc["ctr"][2] = 0
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        X.call(mod, pk, sc, x, ids, tw, out=out)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            X.call(mod, pk, sc, x, ids, tw, out=out)
        for r in range(3):
            x2, ids2, tw2 = VT.make_input(T, 200 + 10 * T + r, dev, "hot" if r == 1 else "skew")
            if not fits(ids2)[0]:
                continue
            sc["ctr"][2] = 0
            x.copy_(x2); ids.copy_(ids2); tw.copy_(tw2)
            gr.replay()
            torch.cuda.synchronize()
            ref = V.ref_fp32(x2, ids2, tw2, w)
            tot, row = V.rel_err(out, ref)
            err = int(sc["ctr"][2])
            good = tot < 6e-3 and row < 3e-2 and err == 0
            ok_all &= good
            print(f"[xs] graph T={T:2d} replay {r}: rel {tot:.3e} row {row:.3e} err {err} -> {'ok' if good else 'FAIL'}")
    print(f"[xs] {'ALL PASS' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

"""Local check of NTL (tools/gen_k3_ntl.py) on the RTX 3070 (sm_86): the lab file's own VMS source (base build vs
-DVMS_NTL=1) and tail_kernel2 (PDL instructions removed for sm_86), on the lab file's own VMS weight packing.

Per case: p (pooler output, bf16 [T, 4096]), router partials [8, T, 256] f32, bias, local_start; the [T] sum buffer is filled
the way the pooler epilogue does (sum of p^2 per token, FP32). Reference = tail_kernel2 -> xn / tw / tid / lid -> VMS base.
NTL = one VMS-NTL launch on (p, partials, sums). Checks: tid / lid / tw bitwise, the sum buffer zeroed after the call
(and untouched past T), err flag 0, MoE output vs reference (rel < 1e-3), and vs base VMS on xn built in torch from the
same sums (exact rounding point; rel < 1e-3). Cases: T = 1..16, normal / exact ties / near ties (1-ulp) / hot experts /
no local route, local_start 0 / 96 / 224, two calls each (state reset), random call order.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/ntl_local_test.py" [kernels/lab/k3_ntl_1.py]
"""
import ast
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "kernels/lab"))
import vms_local_test as VT  # noqa: E402

LAB = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_ntl_1.py")
MULT = 0.1767766952966369


def lab_parts():
    s = LAB.read_text(encoding="utf-8")
    src, ns = {}, {"torch": torch}
    want = ("VMS_CUDA_SRC", "VMS_CPP_SRC", "_NTL_VMS_CPP", "V8_TAIL_CPP_SRC", "V8_TAIL_CUDA_SRC", "VMS_POS", "VMS_UB")
    for n in ast.parse(s).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in want:
            src[n.targets[0].id] = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name in ("_vms_pack_words", "_vms_prepack_expert"):
            src[n.name] = ast.get_source_segment(s, n)
    ns["VMS_POS"] = src["VMS_POS"]
    exec(src["_vms_pack_words"], ns)
    exec(src["_vms_prepack_expert"], ns)
    return src, ns


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    src, ns = lab_parts()
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=2", "-DVMS_RO=1", "-gencode=arch=compute_86,code=sm_86"]
    mb = load_inline(name="ntl_vms_base86", cpp_sources=src["VMS_CPP_SRC"], cuda_sources=src["VMS_CUDA_SRC"],
                     functions=["run", "attrs"], extra_cuda_cflags=flags)
    mn = load_inline(name="ntl_vms_ntl86", cpp_sources=src["VMS_CPP_SRC"] + src["_NTL_VMS_CPP"], cuda_sources=src["VMS_CUDA_SRC"],
                     functions=["run", "run_ntl", "attrs"], extra_cuda_cflags=flags + ["-DVMS_NTL=1"])
    cuda = src["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    cuda = cuda.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    tm = load_inline(name="ntl_tail86", cpp_sources=src["V8_TAIL_CPP_SRC"], cuda_sources=cuda, functions=["v8_tail", "v8_router32"],
                     extra_cuda_cflags=["-O3"])
    print(f"[ntl-local] built: base {list(mb.attrs())} ntl {list(mn.attrs())}", flush=True)
    w = VT.make_weights(dev)
    E = w["w13"].shape[0]
    pk = ns["_vms_prepack_expert"]
    wA = torch.stack([pk(w["w13"][e], w["w13_sf"][e], True) for e in range(E)]).contiguous()
    wB = torch.stack([pk(w["w2"][e], w["w2_sf"][e], False) for e in range(E)]).contiguous()
    two62 = float(2 ** 62)
    gsA = (w["w13_gs"].float().reshape(E) * two62).contiguous()
    gsB = (w["w2_gs"].float().reshape(E) * two62).contiguous()
    nwp = (1.0 + w["post_norm_w"].float()).contiguous()
    del w
    sc = {"inter": torch.zeros((128, 4096), dtype=torch.bfloat16, device=dev), "ybuf": torch.zeros((128, 4096), dtype=torch.float32, device=dev),
          "ss": torch.zeros((256,), dtype=torch.float32, device=dev), "ctr": torch.zeros((72,), dtype=torch.int32, device=dev),
          "nom": torch.empty((0,), dtype=torch.int64, device=dev)}
    g0 = torch.Generator(device=dev).manual_seed(9)
    xw = (1.0 + 0.1 * torch.randn((4096,), generator=g0, device=dev)).to(torch.bfloat16)
    bias0 = (0.05 * torch.randn((256,), generator=g0, device=dev)).float()
    ssum = torch.zeros((32,), dtype=torch.float32, device=dev)
    tw_o = torch.zeros((128,), dtype=torch.float32, device=dev)
    lid_o = torch.zeros((128,), dtype=torch.int32, device=dev)
    eps = 1e-5

    def vms(xn, lid, tw, T):
        o = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        mb.run(xn, lid, tw, wA, wB, gsA, gsB, nwp, sc["inter"], sc["ybuf"], sc["ss"], sc["ctr"], o, sc["nom"], T, 1e-8, 0, 0)
        return o

    cases = []
    for T in (1, 2, 3, 5, 8, 9, 12, 16):
        for kind in ("normal", "tie", "near", "hot", "nolocal"):
            cases.append((T, kind))
    order = torch.randperm(len(cases), generator=torch.Generator().manual_seed(77)).tolist()
    ok_all, n = True, 0
    worst, wcase = 0.0, ""
    for ci in order:
        T, kind = cases[ci]
        for ls in (0, 96, 224):
            n += 1
            seed = 1000 + 37 * ci + ls
            gg = torch.Generator(device=dev).manual_seed(seed)
            y = (torch.randn((T, 4096), generator=gg, device=dev) * 0.7).to(torch.bfloat16).contiguous()
            lg = (torch.randn((8, T, 256), generator=gg, device=dev) * 0.5).contiguous()
            b = bias0.clone()
            tix = (torch.arange(-6, 6, device=dev) + ls) % 256   # 12 experts straddling the local boundary at ls
            if kind == "tie":   # exactly equal top biased scores: the lowest index decides local vs not
                lg[:, :, tix] = 0.6
                b[tix] = float(b[ls])
            elif kind == "near":   # equal top logits, biases equal / 1 ulp apart alternately
                lg[:, :, tix] = 0.6
                v = torch.full((12,), float(b[ls]), device=dev)
                v[1::2] = torch.nextafter(v[1::2].clone(), torch.ones(6, device=dev))
                b[tix] = v
            elif kind == "hot":   # every token picks the same 8 local experts
                lg[:, :, ls + 3:ls + 11] += 2.0
            elif kind == "nolocal":   # nothing local: top-8 far away from [ls, ls + 32)
                far = (ls + 128) % 256
                lg[:, :, far:far + 8] += 3.0
            lst = torch.tensor([ls], dtype=torch.int32, device=dev)
            for call in range(2):
                ssum.zero_()
                ssum[:T] = y.float().pow(2).sum(-1)
                ss_copy = ssum.clone()
                xn = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
                tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
                tid = torch.empty((T, 8), dtype=torch.int32, device=dev)
                lid = torch.empty((T, 8), dtype=torch.int32, device=dev)
                tm.v8_tail(y, lg, 0, -1, lg.stride(1), 8, lg.stride(0), xw, b, lst, xn, tw, tid, lid, eps, MULT, 1.0, 1e-15, 2)
                ref = vms(xn, lid, tw, T)
                s_t = torch.rsqrt(ss_copy[:T] / 4096.0 + eps)
                xn2 = ((y.float() * s_t[:, None]) * xw.float()).to(torch.bfloat16)
                ref2 = vms(xn2, lid, tw, T)
                out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
                tid2 = torch.full((T, 8), -7, dtype=torch.int32, device=dev)
                tw_o.fill_(-7.0)
                lid_o.fill_(-7)
                sc["ctr"][2] = 0
                ssum[T:] = 3.0   # past T must stay untouched
                mn.run_ntl(y, lg, b, lst, ssum, xw, tid2, tw_o, lid_o, wA, wB, gsA, gsB, nwp, sc["inter"], sc["ybuf"], sc["ss"],
                           sc["ctr"], out, sc["nom"], T, 1e-8, eps, MULT, 1.0, 1e-15, 0)
                torch.cuda.synchronize()
                ids_ok = torch.equal(tid2, tid) and torch.equal(lid_o[:T * 8].view(T, 8), lid)
                tw_ok = torch.equal(tw_o[:T * 8].view(T, 8), tw)
                reset_ok = bool((ssum[:T] == 0).all()) and bool((ssum[T:] == 3.0).all())
                err = int(sc["ctr"][2])
                rel = float((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
                rel2 = float((out.float() - ref2.float()).norm() / ref2.float().norm().clamp_min(1e-30))
                xd = int((xn2.view(torch.int16) != xn.view(torch.int16)).sum())
                nloc = int((lid >= 0).sum())
                fin = bool(torch.isfinite(out.float()).all()) and bool(torch.isfinite(ref.float()).all())
                ok = ids_ok and tw_ok and reset_ok and err == 0 and fin and rel < 1e-3 and rel2 < 1e-3
                if rel > worst:
                    worst, wcase = rel, f"T={T} {kind} ls={ls} call {call} rel(torch xn) {rel2:.1e} xn!=tail {xd}"
                ok_all &= ok
                if not ok or (call == 0 and ls == 96 and kind in ("tie", "near") and T in (1, 8, 16)):
                    print(f"[ntl-local] T={T:2d} {kind:7s} ls={ls:3d} call {call}: ids {ids_ok} tw {tw_ok} reset {reset_ok} err {err} "
                          f"local {nloc:3d} rel {rel:.1e} rel(torch xn) {rel2:.1e} xn(torch)!=tail {xd} -> {'ok' if ok else 'FAIL'}",
                          flush=True)
                ssum.zero_()
    print(f"[ntl-local] {n} cases x 2 calls: {'ALL PASS' if ok_all else 'FAILURES'} (worst rel vs tail2+VMS {worst:.2e} at {wcase})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

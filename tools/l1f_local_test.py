"""l1f_local_test.py: functional check of the RE GD block on the local RTX 3070 (sm_86; PDL / clusters compiled out).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/l1f_local_test.py" [kernels/lab/k3_l1f_0.py]

The lab file's own GD source, P12 packers and tail_kernel2 (griddepcontrol stripped). Random Gaussian weights as the judge's
generate_weights (randn / sqrt(K)); router FP32 + its hi / lo BF16 split for GC. Per case (VF T = 1, 2, 5, 8; GC T = 9, 16;
local_start 0 / 96 / 224; RE = 1 and RE = 2): pooler y and router partials bitwise vs the unarmed ship launch, tw / tid / lid
bitwise vs tail_kernel2 on the same (y, partials), counters reset + no error flag; two calls in a row; a forced-tie case
(two router rows identical -> equal logits) checks the tie order.
"""
import ast
import math
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "kernels/lab/k3_l1f_0.py")
INST = [(0, 1, 1, 8, 16, 2, 1, 2, 1), (0, 2, 1, 8, 18, 2, 1, 2, 0, 2)]
HIDDEN, D_EXPERT, N_EXPERTS, TOP_K = 8192, 4096, 256, 8
MULT, RSCALE, DEPS, EPS = 0.1767766952966369, 1.0, 1e-15, 1e-6


def lab_parts():
    s = LAB.read_text(encoding="utf-8")
    src = {}
    ns = {"torch": torch, "HIDDEN": HIDDEN, "D_EXPERT": D_EXPERT, "N_EXPERTS": N_EXPERTS, "GD_ESC_CAP": 32, "GD_CAT_UNITS": 2}
    want = ("GD_CUDA_SRC", "GD_CPP_SRC", "V8_TAIL_CPP_SRC", "V8_TAIL_CUDA_SRC")
    fns = ("_gd_i32", "_gd_frag", "_gd_window_base", "_gd_pack", "_gd_cat_virtual", "_gd_cat_pack")
    for n in ast.parse(s).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in want:
            src[n.targets[0].id] = ast.literal_eval(n.value)
        if isinstance(n, ast.FunctionDef) and n.name in fns:
            exec(ast.get_source_segment(s, n), ns)
    return src, ns


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    src, ns = lab_parts()
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(INST))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in k)) for i, k in enumerate(INST))
    cu = src["GD_CUDA_SRC"].replace("@INST@", inst).replace("@INFO@", info)
    gm = load_inline(name="re_gd86_" + str(abs(hash(cu)) % 10 ** 8), cpp_sources=src["GD_CPP_SRC"], cuda_sources=cu,
                     functions=["gd_run", "gd_info", "gdr_run", "gdr_info", "gdc_run", "gd_re_arm", "gd_re_info"],
                     extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86"])
    tcu = src["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    tcu = tcu.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    tm = load_inline(name="re_tail86_" + str(abs(hash(tcu)) % 10 ** 8), cpp_sources=src["V8_TAIL_CPP_SRC"], cuda_sources=tcu,
                     functions=["v8_tail", "v8_router32"], extra_cuda_cflags=["-O3"])
    print(f"[re-local] built; RE info vf {list(gm.gd_re_info(0))} vf_mk {list(gm.gd_re_info(1))} gc {list(gm.gd_re_info(2))} "
          f"gc_mk {list(gm.gd_re_info(3))}", flush=True)
    gen = torch.Generator(device=dev).manual_seed(4242)
    wp = (torch.randn((D_EXPERT, HIDDEN), generator=gen, device=dev) / math.sqrt(HIDDEN)).to(torch.bfloat16)
    w32 = 0.02 * torch.randn((N_EXPERTS, HIDDEN), generator=gen, device=dev)
    w32[77] = w32[13]   # forced tie: experts 13 and 77 always have equal logits
    hi = w32.to(torch.bfloat16)
    lo = (w32 - hi.float()).to(torch.bfloat16)
    bias = (0.01 * torch.randn((N_EXPERTS,), generator=gen, device=dev)).contiguous()
    bias[77] = bias[13]
    nw = (1 + 0.1 * torch.randn((D_EXPERT,), generator=gen, device=dev)).to(torch.bfloat16)
    pk = ns["_gd_pack"](wp, 1, 8, 16, 1, ns["_gd_window_base"](wp))
    wv = ns["_gd_cat_virtual"](hi, lo)
    pc = ns["_gd_cat_pack"](wp, wv, 8, 16, ns["_gd_window_base"](wp), ns["_gd_window_base"](hi), ns["_gd_window_base"](lo))
    print(f"[re-local] packed: pooler {pk['nesc']} escapes, cat {pc['nesc']} escapes", flush=True)
    ctr = torch.zeros((8,), dtype=torch.int32, device=dev)
    mk = torch.zeros((16 + 4 * 256,), dtype=torch.int64, device=dev)
    nomk = torch.empty((0,), dtype=torch.int64, device=dev)
    fails, n = [], 0

    def tail(y, part, ls):
        T = y.shape[0]
        xn = torch.empty((T, D_EXPERT), dtype=torch.bfloat16, device=dev)
        o = (torch.empty((T, TOP_K), dtype=torch.float32, device=dev), torch.empty((T, TOP_K), dtype=torch.int32, device=dev),
             torch.empty((T, TOP_K), dtype=torch.int32, device=dev))
        tm.v8_tail(y, part, 0, -1, part.stride(1), part.shape[0], part.stride(0), nw, bias, ls, xn, o[0], o[1], o[2], EPS, MULT,
                   RSCALE, DEPS, 2)
        return o

    for kind, Ts in (("vf", (1, 2, 5, 8)), ("gc", (9, 16))):
        for T in Ts:
            for lsv in (0, 96, 224):
                for re_mk in (False, True):
                    n += 1
                    g2 = torch.Generator(device=dev).manual_seed(1000 * T + lsv + int(re_mk))
                    h = torch.randn((T, HIDDEN), generator=g2, device=dev)
                    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True))).contiguous()
                    hb = h.to(torch.bfloat16).contiguous()
                    ls = torch.tensor([lsv], dtype=torch.int32, device=dev)
                    res = []
                    for call in range(3):   # call 0: unarmed ship launch; calls 1, 2: armed (state reset between calls)
                        y = torch.full((T, D_EXPERT), float("nan"), dtype=torch.bfloat16, device=dev)
                        p = torch.full((8, T, N_EXPERTS), float("nan"), dtype=torch.float32, device=dev)
                        o = (torch.full((T, TOP_K), float("nan"), device=dev), torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev),
                             torch.full((T, TOP_K), -7, dtype=torch.int32, device=dev))
                        if call:
                            gm.gd_re_arm(ctr, o[0], o[1], o[2], bias, ls, mk if re_mk else nomk, 1, MULT, RSCALE, DEPS)
                        if kind == "vf":
                            gm.gdr_run(2, pk["w"], pk["esc"], hb, y, T, HIDDEN, D_EXPERT, pk["tb"], pk["nbx"], 0, h, w32, p, nomk)
                        else:
                            gm.gdc_run(1, pc["w"], pc["esc"], hb, y, p, T, HIDDEN, D_EXPERT, N_EXPERTS, pc["tb"], pc["tbr"], pc["nbx"], 0)
                        torch.cuda.synchronize()
                        res.append((y, p, o, ctr.tolist()))
                    y0, p0 = res[0][0], res[0][1]
                    ref = tail(y0, p0, ls)
                    torch.cuda.synchronize()
                    for call in (1, 2):
                        y, p, o, c = res[call]
                        msg = []
                        if not torch.equal(y, y0):
                            msg.append("y differs")
                        if not torch.equal(p, p0):
                            msg.append("partials differ")
                        if not (torch.equal(o[1], ref[1]) and torch.equal(o[2], ref[2]) and torch.equal(o[0], ref[0])):
                            msg.append(f"routing differs (tid {int((o[1] != ref[1]).sum())}, lid {int((o[2] != ref[2]).sum())}, "
                                       f"tw equal {bool(torch.equal(o[0], ref[0]))})")
                        if any(c[:4]):
                            msg.append(f"ctr {c[:4]}")
                        if msg:
                            fails.append(f"{kind} T={T} ls={lsv} mk={re_mk} call {call}: {'; '.join(msg)}")
                    if re_mk:
                        m = mk.cpu()
                        r = m[:8].view(2, 4)
                        cta = m[16:].view(256, 4)
                        t0 = int(cta[:, 0].min())
                        print(f"[re-local] {kind} T={T} marks us: routing ready {(int(r[:, 2].max()) - t0) / 1e3:.1f}, "
                              f"pooler end {(int(cta[:, 3].max()) - t0) / 1e3:.1f}, last share {(int(cta[:, 2].max()) - t0) / 1e3:.1f}; routing CTAs "
                              f"(entry / spin exit / done) {[[round((int(x) - t0) / 1e3, 1) for x in rr[:3]] for rr in r if int(rr[0])]}",
                              flush=True)
                        mk.zero_()
                    tied = bool(((ref[1] == 13).any(1) | (ref[1] == 77).any(1)).any())
                    if n % 6 == 1 or tied:
                        print(f"[re-local] {kind} T={T} ls={lsv}: ids[0] {ref[1][0].tolist()} tie-expert picked {tied}", flush=True)
    print(f"[re-local] {n} cases x 2 armed calls: {'ALL PASS' if not fails else str(len(fails)) + ' FAIL'}", flush=True)
    for f in fails[:20]:
        print("   ", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

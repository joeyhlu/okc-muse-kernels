"""Local ($0) per-phase marks of VMS base vs VMS-NTL on the RTX 3070 (sm_86): same routings / weights as
tools/ntl_local_time.py. Uses VMS's p.marks rows (globaltimer, [G * 16]); the NTL build gets one extra mark mrow[10]
right after the routing replay + __syncthreads. Per T and build prints medians over calls of the CTA-median and CTA-max
of each mark relative to the call's earliest CTA entry (us), plus the CUDA-event time per call.
Marks: 0 entry, 10 routed (NTL), 1 job table, 2 first unit landed, 3 first PB job (phase A done on warp 0),
4 doneA wait passed, 5 producer loop end, 7 consumer end, 8 grid barrier passed, 9 end (combine done).
NTL_MK_HOT=1: touch the producer outputs before each call (L2-hot). Extra variants via NTL_MK_VARIANTS="name:flags;..." (flags space separated -D...).

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu -u root -- bash "<repo>/tools/local/cudaenv.sh" \
      python "<repo>/tools/ntl_local_marks.py" kernels/lab/k3_ntl_5.py
"""
import os
import statistics
import sys

import torch

import ntl_local_test as NT

MK = (0, 10, 1, 2, 3, 4, 5, 7, 8, 9)
ROUTE_MARK = ("nb0, nb1);\n  __syncthreads();\n", "nb0, nb1);\n  __syncthreads();\n  if (mk && tid == 0) mrow[10] = gt();\n")

CKV = (r'__device__ __forceinline__ long long nt_ckv(float v) { long long t; asm volatile("{\n.reg .pred q;\nsetp.ne.f32 q, %1, %1;\n'
       r'@q mov.u64 %0, 0;\n@!q mov.u64 %0, %%clock64;\n}" : "=l"(t) : "f"(v)); return t; }' "\n")
SUB_PATCHES = [
    ("#if VMS_NTL\n#if VMS_NTL_NI\n", "#if VMS_NTL\n" + CKV + "#if VMS_NTL_NI\n"),
    ("const int lane, const int tid, const int c, const float4 nb0, const float4 nb1) {\n",
     "const int lane, const int tid, const int c, const float4 nb0, const float4 nb1, long long* mrow, bool mk) {\n"
     "  if (mk && tid == 0) mrow[11] = clock64();\n"),
    ("c, nb0, nb1);\n  __syncthreads();\n", "c, nb0, nb1, mrow, mk);\n  __syncthreads();\n"),
    ("      const float lg[8] = {a0.x,", "      if (mk && tid == 0 && t == 0) mrow[12] = nt_ckv(a0.x + a1.w);\n      const float lg[8] = {a0.x,"),
    ("      int sel_id[8];\n", "      if (mk && tid == 0 && t == 0) mrow[13] = nt_ckv(b[0] + b[7]);\n      int sel_id[8];\n"),
    ("      if (lane == 0) {\n        float sum = 0.f;", "      if (mk && tid == 0 && t == 0) mrow[14] = nt_ckv(sel_s[7] + (float)sel_id[7]);\n"
     "      if (lane == 0) {\n        float sum = 0.f;"),
    ("    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);\n", "    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);\n"
     "    if (mk && tid == 0) mrow[15] = nt_ckv(S.rs[0]);\n"),
]

FR_SUB = [   # NTL_MK_SUB=2: clock64 marks inside nt_route_fr (tid 0): 11 entry, 12 phase-1 stores, 13 after the barrier, 14 top-k done
    ("const int tid, const int c, const float4 nb0, const float4 nb1) {~n  int nls = 0;",
     "const int tid, const int c, const float4 nb0, const float4 nb1, long long* mrow, bool mk) {~n  if (mk && tid == 0) mrow[11] = clock64();~n  int nls = 0;"),
    ("  __syncthreads();~n  if (warp < NW) {~n    if (warp < m) {",
     "  if (mk && tid == 0) mrow[12] = clock64();~n  __syncthreads();~n  if (mk && tid == 0) mrow[13] = clock64();~n  if (warp < NW) {~n    if (warp < m) {"),
    ("    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);~n  }~n#if __CUDA_ARCH__ >= 900~n  asm volatile(~qfence.proxy",
     "    if (tid < m) S.rs[tid] = rsqrtf(ssv / 4096.f + xeps);~n    if (mk && tid == 0) mrow[14] = nt_ckv(S.rtw[7]);~n  }~n#if __CUDA_ARCH__ >= 900~n  asm volatile(~qfence.proxy"),
    ("(ring), warp, lane, tid, c, nb0, nb1);~n  __syncthreads();~n  if (mk && tid == 0) mrow[10]", "(ring), warp, lane, tid, c, nb0, nb1, mrow, mk);~n  __syncthreads();~n  if (mk && tid == 0) mrow[10]"),
]
FR_SUB = [(a.replace("~n", chr(10)).replace("~q", chr(34)), b.replace("~n", chr(10)).replace("~q", chr(34))) for a, b in FR_SUB]


def main():
    from torch.utils.cpp_extension import load_inline
    dev = "cuda"
    G = torch.cuda.get_device_properties(0).multi_processor_count
    src, ns = NT.lab_parts()
    cu = src["VMS_CUDA_SRC"]
    if ROUTE_MARK[0] in cu:
        cu = cu.replace(ROUTE_MARK[0], ROUTE_MARK[1])
    sub = os.environ.get("NTL_MK_SUB") in ("1", "2")
    if os.environ.get("NTL_MK_SUB") == "2":
        cu = cu.replace("#if VMS_NTL && VMS_NTL_FR" + chr(10), "#if VMS_NTL && VMS_NTL_FR" + chr(10) + CKV, 1)
        for a, b in FR_SUB:
            assert cu.count(a) == 1, a
            cu = cu.replace(a, b)
    elif sub:   # clock64 sub-marks inside nt_route, warp 0 lane 0 of every CTA, first token round: mrow[11..15]
        for a, b in SUB_PATCHES:
            assert cu.count(a) == 1, a
            cu = cu.replace(a, b)
    flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=2", "-DVMS_RO=1", "-gencode=arch=compute_86,code=sm_86"]
    if os.environ.get("NTL_MK_ARCH") == "sm_90a":   # H200 box: the layer's VMS flags
        flags = ["-O3", f"-DVMS_G={G}", "-DVMS_NS=3", "-DVMS_RO=1", "-gencode=arch=compute_90a,code=sm_90a"]
    var = os.environ.get("NTL_MK_VARIANTS", "ni1xn0:-DVMS_NTL_NI=1 -DVMS_NTL_XN=0")
    mb = load_inline(name="ntlm_base" + os.environ.get("NTL_MK_ARCH", "86")[-3:], cpp_sources=src["VMS_CPP_SRC"], cuda_sources=cu, functions=["run", "attrs"],
                     extra_cuda_cflags=flags)
    mods, var_fl = {}, {}
    for item in var.split(";"):
        name, fl = item.split(":", 1)
        var_fl[name] = fl
        mods[name] = load_inline(name=f"ntlm_{name}_" + os.environ.get("NTL_MK_ARCH", "86")[-3:], cpp_sources=src["VMS_CPP_SRC"] + src["_NTL_VMS_CPP"],
                                 cuda_sources=cu, functions=["run", "run_ntl", "attrs"],
                                 extra_cuda_cflags=flags + ["-DVMS_NTL=1"] + fl.split())
    print(f"[ntl-mk] G {G} regs base {list(mb.attrs())[:2]} " + " ".join(f"{k} {list(m.attrs())[:2]}" for k, m in mods.items()))
    cuda = src["V8_TAIL_CUDA_SRC"].replace('asm volatile("griddepcontrol.launch_dependents;\\n" ::: "memory");', "")
    cuda = cuda.replace('asm volatile("griddepcontrol.wait;\\n" ::: "memory");', "")
    tm = load_inline(name="ntl_tail86", cpp_sources=src["V8_TAIL_CPP_SRC"], cuda_sources=cuda, functions=["v8_tail", "v8_router32"],
                     extra_cuda_cflags=["-O3"])
    w = NT.VT.make_weights(dev)
    E = w["w13"].shape[0]
    pk = ns["_vms_prepack_expert"]
    wA = torch.stack([pk(w["w13"][e], w["w13_sf"][e], True) for e in range(E)]).contiguous()
    wB = torch.stack([pk(w["w2"][e], w["w2_sf"][e], False) for e in range(E)]).contiguous()
    gsA = (w["w13_gs"].float().reshape(E) * float(2 ** 62)).contiguous()
    gsB = (w["w2_gs"].float().reshape(E) * float(2 ** 62)).contiguous()
    nwp = (1.0 + w["post_norm_w"].float()).contiguous()
    del w
    inter = torch.zeros((128, 4096), dtype=torch.bfloat16, device=dev)
    ybuf = torch.zeros((128, 4096), dtype=torch.float32, device=dev)
    ssv = torch.zeros((256,), dtype=torch.float32, device=dev)
    ctr = torch.zeros((72,), dtype=torch.int32, device=dev)
    marks = torch.zeros((G * 16,), dtype=torch.int64, device=dev)
    g0 = torch.Generator(device=dev).manual_seed(9)
    xw = (1.0 + 0.1 * torch.randn((4096,), generator=g0, device=dev)).to(torch.bfloat16)
    bias = (0.05 * torch.randn((256,), generator=g0, device=dev)).float()
    ssum = torch.zeros((32,), dtype=torch.float32, device=dev)
    tw_o = torch.zeros((128,), dtype=torch.float32, device=dev)
    lid_o = torch.zeros((128,), dtype=torch.int32, device=dev)
    eps = 1e-5
    hot = os.environ.get("NTL_MK_HOT") == "1"
    for T in [int(t) for t in os.environ.get("NTL_MK_T", "1,8,16").split(",")]:
        gg = torch.Generator(device=dev).manual_seed(500 + T)
        y = (torch.randn((T, 4096), generator=gg, device=dev) * 0.7).to(torch.bfloat16).contiguous()
        lg = (torch.randn((8, T, 256), generator=gg, device=dev) * 0.5).contiguous()
        lst = torch.tensor([96], dtype=torch.int32, device=dev)
        ss0 = y.float().pow(2).sum(-1)
        xn = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        tw = torch.empty((T, 8), dtype=torch.float32, device=dev)
        tid = torch.empty((T, 8), dtype=torch.int32, device=dev)
        lid = torch.empty((T, 8), dtype=torch.int32, device=dev)
        tm.v8_tail(y, lg, 0, -1, lg.stride(1), 8, lg.stride(0), xw, bias, lst, xn, tw, tid, lid, eps, NT.MULT, 1.0, 1e-15, 2)
        out = torch.empty((T, 4096), dtype=torch.bfloat16, device=dev)
        tid2 = torch.empty((T, 8), dtype=torch.int32, device=dev)
        lgr = lg[0].clone()   # NTL_NS=1 variants: producer-reduced logits, slice order 0..7 (bitwise = tail2's sum)
        for q in range(1, 8):
            lgr = lgr + lg[q]
        lgr = torch.cat([lgr.unsqueeze(0), lg[1:]]).contiguous()   # host wants [8, T, 256]; NS=1 reads slice 0 only
        runs = {"base": lambda mk: mb.run(xn, lid, tw, wA, wB, gsA, gsB, nwp, inter, ybuf, ssv, ctr, out, mk, T, 1e-8, 0, 0)}
        for name, m in mods.items():
            runs[name] = (lambda m, L: lambda mk: m.run_ntl(y, L, bias, lst, ssum, xw, tid2, tw_o, lid_o, wA, wB, gsA, gsB, nwp, inter,
                                                         ybuf, ssv, ctr, out, mk, T, 1e-8, eps, NT.MULT, 1.0, 1e-15, 0))(m, lgr if "VMS_NTL_NS=1" in var_fl[name] else lg)
        for name, fn in runs.items():
            ev, rows, subr = [], [], [[], [], [], []]
            for i in range(160):
                ssum[:T].copy_(ss0)
                if hot:   # producer outputs L2-resident as in the layer (tail2 / router just wrote them)
                    for h in ((xn, lid, tw) if name == "base" else (lg, y)):
                        h.add_(0)
                marks.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                fn(marks)
                b.record()
                b.synchronize()
                if i >= 30:
                    ev.append(a.elapsed_time(b) * 1000.0)
                    mm = marks.view(G, 16).cpu()
                    t0 = int(mm[:, 0].min())
                    r = []
                    for k in MK:
                        col = [float(v - t0) / 1000.0 for v in mm[:, k].tolist() if v > 0]
                        r.append((statistics.median(col), max(col)) if col else (float("nan"), float("nan")))
                    rows.append(r)
                    if sub and name != "base":
                        for c in range(G):
                            v = mm[c].tolist()
                            for j, (x0, x1) in enumerate(((11, 12), (12, 13), (13, 14), (14, 15))):
                                subr[j].append(v[x1] - v[x0])
            med = [(statistics.median(x[i][0] for x in rows), statistics.median(x[i][1] for x in rows)) for i in range(len(MK))]
            if name != "base":
                print(f"[ntl-mk] T={T:2d} {name:7s} ids bitwise vs tail2 {bool(torch.equal(tid2, tid))}")
            if sub and name != "base":
                print(f"[ntl-mk] T={T:2d} {name:7s} route cycles "
                      + " / ".join(f"{statistics.median(x):.0f}" for x in subr) + " (SUB=1: loads, sigmoid, top-k, writes+rest; SUB=2: phase 1, barrier, top-k+rs, -)")
            print(f"[ntl-mk] T={T:2d} {name:7s} ev {statistics.median(ev):6.1f} | " +
                  " ".join(f"m{k} {a:5.1f}/{b:5.1f}" for k, (a, b) in zip(MK, med)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

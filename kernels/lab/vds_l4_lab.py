"""vds_l4_lab.py: lever P9a step 0, GD side (notes/study/gap_oct28.md s9): q / k / v rows first in GD qkv at T = 1 / 8 (VDS builder,
2026-10-28). The layer's own gd_kernel (from k3_vf_10) gets a release counter: every q / k / v cluster leader bumps it after its
stores; the gate-row clusters hold (bounded spin, then proceed) until all q / k / v tiles are released. KS1 would start its
post-wait chain at the release instead of at GD qkv's end. Measured here, standalone (weights cold, x L2-hot):
  ship   GD qkv as shipped (rel off)
  held   GD qkv with the hold + release; release time = when the last q / k / v leader released, from kernel entry
The window = ship time - release time is how much earlier KS1 could start; held - ship is the gate-row restart cost.

    python3 kernels/lab/vds_l4_lab.py [--ts=1,8] [--reps=21] [--rounds=6] [--layer=kernels/lab/k3_vf_10_ab.py]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vds_lab as VL  # noqa: E402

QKV_ROUTE = (1, 1, 4, 8, 2, 4, 1)   # k3_vf_10 GD_ROUTE[("qkv", 1 | 8)]
QKV_ROWS, REL_RT = 2304, 80         # q / k / v = rows 0 .. 1279 = row tiles 0 .. 79; gate = tiles 80 .. 143


def l4_patch(cuda, cpp):
    """the layer's GD source + the L4 release counter (q / k / v leaders) and gate-row hold; gd_run_rel entry point."""
    def rep(s, old, new, what):
        n = s.count(old)
        if n != 1:
            raise ValueError(f"l4 anchor {what!r} found {n} times")
        return s.replace(old, new)
    cuda = rep(cuda, "float* part; long long* marks;",
               "float* part; long long* marks; int* rel; int rel_rt; int rel_n; unsigned long long* rel_t;", "Args")
    cuda = rep(cuda, "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0>\n__global__",
               "__device__ __forceinline__ int l4_ld_acq(const int* p) {\n  int v;\n"
               "  asm volatile(\"ld.acquire.gpu.global.b32 %0, [%1];\\n\" : \"=r\"(v) : \"l\"(p) : \"memory\");\n  return v;\n}\n"
               "template <int OUT, int MT, int WR, int WK, int KU, int ST, int S, int MB, int XP, int VF = 0>\n__global__", "helper")
    cuda = rep(cuda, "  pdl_wait();\n",
               "  pdl_wait();\n"
               "  if (A.rel != nullptr && (int)blockIdx.x >= A.rel_rt) {   // L4: gate rows wait for the q / k / v rows (bounded)\n"
               "    if (threadIdx.x == 0) {\n"
               "      const int v0 = l4_ld_acq(A.rel);\n"
               "      const int tgt = (v0 / A.rel_n) * A.rel_n + A.rel_n;\n"
               "      const long long t0 = clock64();\n"
               "      while (l4_ld_acq(A.rel) < tgt && clock64() - t0 < 40000000LL) __nanosleep(64);\n"
               "    }\n"
               "    __syncthreads();\n"
               "  }\n"
               "  if (A.rel_t != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) A.rel_t[1] = (unsigned long long)vf_gt();\n",
               "gate hold")
    cuda = rep(cuda, "      for (int i = (int)threadIdx.x; i < NO; i += NT) st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);\n      return;\n",
               "      for (int i = (int)threadIdx.x; i < NO; i += NT) st_async(rb + 4u * (uint32_t)i, cta_value(i), rm);\n"
               "      if (A.rel_t != nullptr && threadIdx.x == 0) atomicMax(&A.rel_t[2], (unsigned long long)vf_gt());\n"
               "      return;\n", "non-leader exit stamp")
    cuda = rep(cuda, "        store(i, ok ? v : __int_as_float(0x7fc00000));\n      }\n    }\n#endif\n",
               "        store(i, ok ? v : __int_as_float(0x7fc00000));\n      }\n    }\n"
               "    if (A.rel != nullptr && (int)blockIdx.x < A.rel_rt) {   // L4: this q / k / v row tile is stored\n"
               "      __threadfence();\n"
               "      __syncthreads();\n"
               "      if (threadIdx.x == 0) {\n"
               "        const int old = atomicAdd(A.rel, 1);\n"
               "        if (A.rel_t != nullptr && (old % A.rel_n) == A.rel_n - 1) A.rel_t[0] = (unsigned long long)vf_gt();\n"
               "      }\n"
               "    }\n"
               "    if (A.rel_t != nullptr && threadIdx.x == 0) atomicMax(&A.rel_t[2], (unsigned long long)vf_gt());\n#endif\n", "leader release")
    n0 = cuda.count("A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;")
    if n0 < 1:
        raise ValueError("l4 anchor 'Args init' not found")
    cuda = cuda.replace("A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;",
                        "A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr; A.rel = nullptr; A.rel_t = nullptr; "
                        "A.rel_rt = 0; A.rel_n = 1;")
    cuda = rep(cuda, "  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks;",
               "  A.x32 = x32; A.w32 = w32; A.part = part; A.marks = marks; A.rel = nullptr; A.rel_t = nullptr; A.rel_rt = 0; A.rel_n = 1;",
               "gdr Args init")
    cuda = rep(cuda, "int gd_info_raw(int64_t inst, int64_t* res) {",
               "int gd_run_rel_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo,\n"
               "                   uint32_t tb, int nbx, int nz, int pdl, int* rel, int rel_rt, int rel_n, unsigned long long* rel_t,\n"
               "                   cudaStream_t st) {\n"
               "  Args A;\n"
               "  A.w = reinterpret_cast<const uint4*>(w);\n"
               "  A.esc = reinterpret_cast<const uint32_t*>(esc);\n"
               "  A.x = reinterpret_cast<const uint16_t*>(x);\n"
               "  A.out = out;\n"
               "  A.M = M; A.K = K; A.ldo = ldo; A.tb = tb;\n"
               "  A.trig = (pdl >> 1) & 1;\n"
               "  A.x32 = nullptr; A.w32 = nullptr; A.part = nullptr; A.marks = nullptr;\n"
               "  A.rel = rel; A.rel_rt = rel_rt; A.rel_n = rel_n; A.rel_t = rel_t;\n"
               "  switch (inst) {\n@INST@\n    default: return -1;\n  }\n}\n\n"
               "int gd_info_raw(int64_t inst, int64_t* res) {", "host entry")
    cpp = rep(cpp, "int gd_info_raw(int64_t inst, int64_t* res);",
              "int gd_info_raw(int64_t inst, int64_t* res);\n"
              "int gd_run_rel_raw(int64_t inst, const void* w, const void* esc, const void* x, void* out, int M, int K, int ldo,\n"
              "                   uint32_t tb, int nbx, int nz, int pdl, int* rel, int rel_rt, int rel_n, unsigned long long* rel_t,\n"
              "                   cudaStream_t st);\n"
              "void gd_run_rel(int64_t inst, torch::Tensor w, torch::Tensor esc, torch::Tensor x, torch::Tensor out, int64_t M,\n"
              "                int64_t K, int64_t ldo, int64_t tb, int64_t nbx, int64_t nz, int64_t pdl, torch::Tensor rel,\n"
              "                int64_t rel_rt, int64_t rel_n, torch::Tensor rel_t) {\n"
              "  const int e = gd_run_rel_raw(inst, w.data_ptr(), esc.data_ptr(), x.data_ptr(), out.data_ptr(), (int)M, (int)K,\n"
              "      (int)ldo, (uint32_t)tb, (int)nbx, (int)nz, (int)pdl, rel.numel() ? rel.data_ptr<int>() : nullptr, (int)rel_rt, (int)rel_n,\n"
              "      rel_t.numel() ? reinterpret_cast<unsigned long long*>(rel_t.data_ptr()) : nullptr,\n"
              "      at::cuda::getCurrentCUDAStream());\n"
              "  TORCH_CHECK(e == 0, \"gd_run_rel: inst \", inst, \" CUDA error \", e);\n"
              "}", "cpp decl")
    return cuda, cpp


def main():
    ts_ = [int(v) for v in VL.arg("ts", "1,8").split(",")]
    reps = int(VL.arg("reps", "21"))
    rounds = int(VL.arg("rounds", "6"))
    layer = VL.arg("layer", str(ROOT / "kernels" / "lab" / "k3_vf_10_ab.py"))
    out_path = Path(VL.arg("out", str(ROOT / "box_out" / "vdsl4" / "lab.json")))
    from torch.utils.cpp_extension import load_inline
    gns = VL.layer_extract(layer, ("GD_", "_GD", "_gd"), ("K2A_BUCKETS", "_k2a_bucket_of", "SU1_K3_TRIG_AT"))
    cuda, cpp = l4_patch(gns["GD_CUDA_SRC"], gns["GD_CPP_SRC"])
    key = gns["_gd_key"](QKV_ROWS, 8192, QKV_ROUTE)
    cuda = cuda.replace("@INST@", "    GD_INST(0, %s)" % ", ".join(str(v) for v in key)).replace(
        "@INFO@", "    GD_INFO(0, %s)" % ", ".join(str(v) for v in key))
    t0 = time.time()
    tag = hashlib.sha1((cpp + cuda).encode()).hexdigest()[:10]
    mod = load_inline(name=f"vdsl4_gd_{tag}", cpp_sources=cpp, cuda_sources=cuda,
                      functions=["gd_run", "gd_info", "gd_unpack", "gd_run_rel"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"], verbose=False)
    print(f"[l4] built GD qkv {key} in {time.time() - t0:.1f}s; info {list(mod.gd_info(0))}", flush=True)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(21)
    W = (torch.randn((QKV_ROWS, 8192), generator=gen, device=dev) / math.sqrt(8192)).to(torch.bfloat16)
    mt, wr, wk, ku, st_, s, xp = QKV_ROUTE
    base = gns["_gd_window_base"](W)
    pk = gns["_gd_pack"](W, wr, wk, ku, s, base)
    rel = torch.zeros((4,), dtype=torch.int32, device=dev)
    rel_t = torch.zeros((4,), dtype=torch.int64, device=dev)
    e64 = torch.empty(0, dtype=torch.int64, device=dev)
    flush = torch.empty((64 << 20,), dtype=torch.float32, device=dev).uniform_().view(4096, -1)
    sink = torch.empty((flush.shape[1],), dtype=torch.float32, device=dev)
    results = {"rows": []}
    for T in ts_:
        x = torch.randn((T, 8192), generator=gen, device=dev).to(torch.bfloat16)
        y0 = torch.empty((T, QKV_ROWS), dtype=torch.bfloat16, device=dev)
        y1 = torch.empty((T, QKV_ROWS), dtype=torch.bfloat16, device=dev)

        def ship(x=x, y=y0):
            mod.gd_run(0, pk["w"], pk["esc"], x, y, T, 8192, QKV_ROWS, pk["tb"], pk["nbx"], 1, 2)

        def held(x=x, y=y1, rt=e64):
            mod.gd_run_rel(0, pk["w"], pk["esc"], x, y, T, 8192, QKV_ROWS, pk["tb"], pk["nbx"], 1, 2, rel, REL_RT, REL_RT, rt)
        ship()
        held()
        torch.cuda.synchronize()
        same = bool(torch.equal(y0, y1))
        ref = (x.double() @ W.double().t()).to(torch.bfloat16).double()
        rel_err = float((y1.double() - ref).norm() / ref.norm())
        cnt = int(rel[0].item())
        print(f"[l4] T{T}: outputs bitwise ship == held {same}; rel vs fp64->bf16 {rel_err:.1e}; counter {cnt} (multiple of {REL_RT}: "
              f"{cnt % REL_RT == 0})", flush=True)
        graphs = [VL.graph_of(ship), VL.graph_of(held)]
        warm = lambda: torch.sum(x, dim=0)  # noqa: E731
        meds = {"ship": [], "held": []}
        for _ in range(rounds):
            tt = VL.time_many(graphs, flush, sink, reps, warm)
            meds["ship"].append(tt[0])
            meds["held"].append(tt[1])
        rels, hend, send = [], [], []
        erel = torch.empty(0, dtype=torch.int32, device=dev)
        for _ in range(15):
            for mode in ("held", "ship"):
                torch.sum(flush, dim=0, out=sink)
                warm()
                rel_t.zero_()
                if mode == "held":
                    held(rt=rel_t)
                else:
                    mod.gd_run_rel(0, pk["w"], pk["esc"], x, y0, T, 8192, QKV_ROWS, pk["tb"], pk["nbx"], 1, 2, erel, REL_RT, REL_RT, rel_t)
                torch.cuda.synchronize()
                a, b, c = int(rel_t[0].item()), int(rel_t[1].item()), int(rel_t[2].item())
                if mode == "held" and a and b:
                    rels.append((a - b) / 1e3)
                    hend.append((c - b) / 1e3)
                if mode == "ship" and b and c:
                    send.append((c - b) / 1e3)
        u_s, u_h = statistics.median(meds["ship"]), statistics.median(meds["held"])
        med = lambda v: statistics.median(v) if v else float("nan")  # noqa: E731
        r_med, h_end, s_end = med(rels), med(hend), med(send)
        row = {"T": T, "same": same, "rel": rel_err, "ship_us": u_s, "held_us": u_h, "release_us": r_med, "held_end_us": h_end,
               "ship_end_us": s_end, "release_all": rels, "counter_after": int(rel[0].item())}
        results["rows"].append(row)
        print(f"[l4] T{T}: graph us ship {u_s:.2f} held {u_h:.2f} (held - ship {u_h - u_s:+.2f}); in-kernel from entry: ship end "
              f"{s_end:.2f}, held: q/k/v released {r_med:.2f} (min {min(rels) if rels else float('nan'):.2f}, max "
              f"{max(rels) if rels else float('nan'):.2f}), held end {h_end:.2f}; window = ship end - release = {s_end - r_med:.2f} us",
              flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1))
    print("[l4] done", flush=True)


if __name__ == "__main__":
    main()

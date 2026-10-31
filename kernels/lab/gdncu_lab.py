"""gdncu_lab.py (R-gdncu-0): why does the GD dense GEMV grow with T? Standalone cold-L2 timing of the shipped routes at T = 1 / 16 / 32 and of the
MT = 4 routes fed fewer tokens (M = 16 / 8 / 1: the instruction stream is identical, only the number of distinct x rows, i.e. the x footprint
in L1 / L2, changes), plus eager launches inside NVTX range 'gdncu' for Nsight Compute (`--mode=ncu`, one launch per config, in CFGS order).
    python3 -u kernels/lab/gdncu_lab.py --layer=kernels/lab/k3_hh_3tekv.py --out=box_out/gdncu/time.json [--mode=time|ncu] [--reps=25]
    smoke test on the RTX 3070: --arch=sm_86 --synthetic=1 (cluster routes, S > 1, are skipped there)
Builds one module with every instance (GD block of the layer file by AST, kernels/lab/gdst_lab.py extract), packs each matrix on the judge weight
recipe (or randn / sqrt(k) bf16 with --synthetic=1), checks every output vs FP64, times one gd_run launch as a CUDA graph replay after a 256 MB
L2 flush (median of reps, net of an empty graph). `ncu` mode launches each config eagerly once inside NVTX range gdncu (the log line 'ncu launch <tag>' gives the order).
"""
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "kernels/lab"))
sys.path.insert(0, str(ROOT / "tools/judge"))
sys.path.insert(0, str(ROOT / "tools/box"))
import gdst_lab as G  # noqa: E402  (extract(); its module-level ARGS parse is shared)

ARGS = G.ARGS
MODE = ARGS.get("mode", "time")
REPS = int(ARGS.get("reps", "25"))
OUT = ROOT / ARGS.get("out", "box_out/gdncu/time.json")

# (tag, proj, route (mt, wr, wk, ku, st, s, xp), M)
CFGS = [
    ("qkv_T1", "qkv", (1, 1, 4, 8, 2, 4, 1), 1), ("qkv_T16", "qkv", (2, 1, 4, 8, 1, 4, 0), 16), ("qkv_T32", "qkv", (4, 1, 4, 8, 1, 4, 0), 32),
    ("qkv_T32_M16", "qkv", (4, 1, 4, 8, 1, 4, 0), 16), ("qkv_T32_M8", "qkv", (4, 1, 4, 8, 1, 4, 0), 8), ("qkv_T32_M1", "qkv", (4, 1, 4, 8, 1, 4, 0), 1),
    ("pool_T1", "pooler", (1, 1, 8, 16, 2, 1, 7), 1), ("pool_T16", "pooler", (2, 1, 8, 16, 2, 1, 0), 16), ("pool_T32", "pooler", (4, 1, 8, 16, 1, 1, 0), 32),
    ("pool_T32_M16", "pooler", (4, 1, 8, 16, 1, 1, 0), 16), ("pool_T32_M1", "pooler", (4, 1, 8, 16, 1, 1, 0), 1),
    ("unp_T16", "unpooler", (2, 2, 4, 16, 2, 1, 6), 16), ("unp_T32", "unpooler", (4, 2, 4, 16, 1, 1, 0), 32),
    ("unp_T32_M16", "unpooler", (4, 2, 4, 16, 1, 1, 0), 16), ("unp_T32_M1", "unpooler", (4, 2, 4, 16, 1, 1, 0), 1),
    ("o_T1", "o", (1, 1, 8, 2, 2, 1, 6), 1), ("o_T16", "o", (2, 2, 4, 4, 2, 1, 6), 16), ("o_T32", "o", (4, 1, 4, 4, 1, 1, 0), 32),
]
# --xh=1 (tools/gdxh_patches.py): XP bit 8 = token-half X; the same routes with the bit set, plus the M = 1 probe of the MT 4 ones
XH_CFGS = [
    ("qkv_T32x", "qkv", (4, 1, 4, 8, 1, 4, 8), 32), ("qkv_T32x_M1", "qkv", (4, 1, 4, 8, 1, 4, 8), 1), ("qkv_T16x", "qkv", (2, 1, 4, 8, 1, 4, 8), 16),
    ("pool_T32x", "pooler", (4, 1, 8, 16, 1, 1, 8), 32), ("pool_T32x_M1", "pooler", (4, 1, 8, 16, 1, 1, 8), 1),
    ("unp_T32x", "unpooler", (4, 2, 4, 16, 1, 1, 8), 32), ("unp_T32x_M1", "unpooler", (4, 2, 4, 16, 1, 1, 8), 1), ("unp_T16x", "unpooler", (2, 2, 4, 16, 2, 1, 14), 16),
    ("o_T32x", "o", (4, 1, 4, 4, 1, 1, 8), 32),
]


def main():
    ns = G.extract(G.LAYER)
    if ARGS.get("xh", "0") == "1":
        sys.path.insert(0, str(ROOT / "tools"))
        import gdxh_patches  # noqa: E402
        ns["GD_CUDA_SRC"] = gdxh_patches.patch(ns["GD_CUDA_SRC"])
        CFGS.extend(XH_CFGS)
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if ARGS.get("synthetic", "0") == "1":
        g = torch.Generator(device=dev).manual_seed(3)
        W = {p: (torch.randn(ns["GD_DIMS"][p], generator=g, device=dev) / math.sqrt(ns["GD_DIMS"][p][1])).to(torch.bfloat16)
             for p in ("qkv", "o", "pooler", "unpooler")}
    else:
        import judge as J  # noqa: E402
        prob = J.load_problem(ROOT / "problems/muse_decode_layer")
        ref = prob.ref
        weights = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=G.RSEED)
        W = {p: weights[ns["GD_WEIGHT"][p]].contiguous() for p in ("qkv", "o", "pooler", "unpooler")}
        del weights
    arch = ARGS.get("arch", "sm_90a")
    gencode = [f"-gencode=arch=compute_{arch[3:]},code={arch}"]
    keys, kmap = [], {}
    for tag, proj, rt, M in CFGS:
        n, k = W[proj].shape
        key = ns["_gd_key"](n, k, rt)
        if key is None:
            print(f"[gdncu] {tag}: invalid geometry {rt}", flush=True)
            continue
        kmap[(proj, rt)] = key
        if key not in keys:
            keys.append(key)
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(keys))
    cuda = ns["GD_CUDA_SRC"].replace("@INST@", inst).replace("@INFO@", info)
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="gdncu_" + hashlib.sha1((ns["GD_CPP_SRC"] + cuda + arch).encode()).hexdigest()[:10],
                      cpp_sources=ns["GD_CPP_SRC"], cuda_sources=cuda, functions=["gd_run", "gd_info", "gd_unpack"],
                      extra_cuda_cflags=["-O3"] + gencode, verbose=False)
    print(f"[gdncu] built {len(keys)} instances in {time.time() - t0:.0f}s", flush=True)
    flush_buf = torch.empty((256 << 20) // 4, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(5)
    xs = {k: torch.randn((32, k), generator=gen, device=dev).to(torch.bfloat16) for k in (8192, 4096, 1024)}

    def timed_graph(fn):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            flush_buf.add_(1.0)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            gr.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts), min(ts)

    empty_med, _ = timed_graph(lambda: None)
    print(f"[gdncu] empty graph {empty_med:.2f} us, {sms} SMs, mode {MODE}", flush=True)
    packs, rows, ys = {}, [], {}
    for tag, proj, rt, M in CFGS:
        if (proj, rt) not in kmap:
            continue
        key = kmap[(proj, rt)]
        inst_id = keys.index(key)
        inf = mod.gd_info(inst_id)
        n, k = W[proj].shape
        mt, wr, wk, ku, st_, s, xp = rt
        ctas = (n // 16 // wr) * s
        if s > 1 and not arch.startswith("sm_90"):
            print(f"[gdncu] {tag}: cluster route skipped on {arch}", flush=True)
            continue
        pk_key = (proj, wr, wk, ku, s)
        if pk_key not in packs:
            packs[pk_key] = ns["_gd_pack"](W[proj], wr, wk, ku, s, ns["_gd_window_base"](W[proj]))
        pk = packs[pk_key]
        x = xs[k][:M].contiguous()
        y = torch.empty((M, n), dtype=torch.bfloat16, device=dev)
        run = lambda: mod.gd_run(inst_id, pk["w"], pk["esc"], x, y, M, k, n, pk["tb"], pk["nbx"], 1, 0)  # noqa: E731
        try:
            run()
            torch.cuda.synchronize()
        except Exception as ex:   # noqa: BLE001
            print(f"[gdncu] {tag}: LAUNCH FAILED {ex!r}"[:300], flush=True)
            continue
        refo = x.double() @ W[proj].double().T
        rel = float((y.double() - refo).norm() / refo.norm())
        yk, bit = (proj, M, tuple(rt[:6])), ""
        if rt[6] & 8 and yk in ys:   # token-half X (XP bit 8): same geometry, same accumulation order -> the output must be bitwise the shipped one
            bit = f" bitwise={bool(torch.equal(y, ys[yk]))}"
        elif not rt[6] & 8:
            ys[yk] = y.clone()
        mb = pk["w"].numel() * 4 / 1e6
        if MODE == "ncu":
            torch.cuda.synchronize()
            flush_buf.add_(1.0)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_push("gdncu")
            run()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            print(f"[gdncu] ncu launch {tag}: {proj} {rt} M={M} regs {inf[0]} lmem {inf[1]} occ {inf[2]} ctas {ctas} rel {rel:.1e}", flush=True)
            continue
        med, mn = timed_graph(run)
        net = med - empty_med
        row = {"tag": tag, "proj": proj, "rt": rt, "M": M, "regs": inf[0], "lmem": inf[1], "occ": inf[2], "ctas": ctas, "net_us": round(net, 2),
               "min_us": round(mn - empty_med, 2), "TBs_p12": round(mb / net, 2) if net > 0 else 0, "MB": round(mb, 1), "rel": rel, "bit": bit.strip()}
        rows.append(row)
        print(f"[gdncu] {tag:13s} {proj:8s} {str(rt):22s} M={M:2d}: net {net:6.2f} us (min {mn - empty_med:6.2f}) {mb / net:.2f} TB/s P12 ({mb:.1f} MB) "
              f"regs {inf[0]} lmem {inf[1]} occ {inf[2]} ctas {ctas} rel {rel:.1e}{bit}", flush=True)
    if MODE != "ncu":
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"empty_us": empty_med, "rows": rows}, indent=1))
        # growth with T: net time of the shipped route per bucket, and what the MT = 4 instruction stream costs with fewer distinct x rows
        by = {r["tag"]: r["net_us"] for r in rows}
        for pj in ("qkv", "pool", "unp", "o"):
            ts = [(t, by[f"{pj}_{t}"]) for t in ("T1", "T16", "T32") if f"{pj}_{t}" in by]
            pr = [(m, by[f"{pj}_T32_{m}"]) for m in ("M16", "M8", "M1") if f"{pj}_T32_{m}" in by]
            print(f"[gdncu] SUMMARY {pj:5s}: shipped " + ", ".join(f"{t} {v:.2f}" for t, v in ts) + " | MT4 route fed " + ", ".join(f"{m} {v:.2f}" for m, v in pr), flush=True)
            xv = [(t, by[f"{pj}_{t}"]) for t in ("T16x", "T32x", "T32x_M1") if f"{pj}_{t}" in by]
            if xv:
                print(f"[gdncu] SUMMARY {pj:5s}: XH " + ", ".join(f"{t} {v:.2f}" for t, v in xv), flush=True)


if __name__ == "__main__":
    main()

"""gdst_lab.py: standalone cold-L2 timing of GD geometries for the T = 17..32 dense routes (R-gdst-1).
    python3 -u kernels/lab/gdst_lab.py --layer=kernels/lab/gdst_base.py --out=box_out/gdst1/lab.json [--reps=15] [--rseed=...]
Takes the GD block (GD_* / _gd* names) of a layer file by AST (no layer import), builds one module with every candidate instance,
packs each matrix on the judge weight recipe, checks the output vs FP64, then times one gd_run launch per candidate as a CUDA graph
replay after a 256 MB L2 flush (median over reps, net of an empty graph). Candidates: the shipped T = 32 routes (ST 1), ST 2 / 3 at
4 warps per CTA (3 CTAs / SM), and MT 2 / ST 2 with two token slices (nz = 2; each weight unit read twice, the twin from L2 if close).
"""
import ast
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
LAYER = ROOT / ARGS.get("layer", "kernels/lab/gdst_base.py")
OUT = ROOT / ARGS.get("out", "box_out/gdst1/lab.json")
REPS = int(ARGS.get("reps", "15"))
RSEED = int(ARGS.get("rseed", "8677534258059892491"))
sys.path.insert(0, str(ROOT / "tools/judge"))
sys.path.insert(0, str(ROOT / "tools/box"))


def extract(path):
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name and (name.startswith(("GD_", "_GD", "_gd")) or name in ("K2A_BUCKETS", "_k2a_bucket_of")):
            nodes.append(node)
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "math": math, "HIDDEN": 8192, "D_EXPERT": 4096, "UNPOOL_SLICE": 1024,
          "SLICED_UNPOOL_MAX_TOKENS": 8, "V12_ZBUF": True, "R1_UNPOOL_INTO_ZEROS": True, "K2A_M_MAX": 4096, "N_EXPERTS": 256,
          "TOP_K": 8, "RMS_EPS": 1e-5, "_pt1_wait": lambda name: None, "DISABLED": [], "_k3_disable": lambda *a: None,
          "_v8_tail": None, "SU1_K3_TRIG_AT": set()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)   # lab tool only
    return ns


# (proj, (mt, wr, wk, ku, st, s, xp), nz); the first of each proj is the shipped T = 32 route
CANDS = [
    ("pooler", (4, 1, 8, 16, 1, 1, 0), 1), ("pooler", (4, 1, 4, 32, 2, 1, 0), 1), ("pooler", (4, 1, 4, 32, 3, 1, 0), 1),
    ("pooler", (2, 1, 8, 16, 2, 1, 0), 2), ("pooler", (2, 1, 4, 32, 2, 1, 0), 2), ("pooler", (2, 1, 4, 32, 3, 1, 0), 2),
    ("unpooler", (4, 2, 4, 16, 1, 1, 0), 1), ("unpooler", (4, 2, 2, 32, 2, 1, 0), 1), ("unpooler", (4, 2, 2, 32, 3, 1, 0), 1),
    ("unpooler", (2, 2, 4, 16, 2, 1, 0), 2), ("unpooler", (2, 2, 2, 32, 2, 1, 0), 2), ("unpooler", (2, 2, 2, 32, 3, 1, 0), 2),
    ("qkv", (4, 1, 4, 8, 1, 4, 0), 1), ("qkv", (4, 1, 4, 16, 2, 2, 0), 1), ("qkv", (4, 1, 4, 16, 3, 2, 0), 1),
    ("qkv", (4, 1, 4, 32, 2, 1, 0), 1), ("qkv", (2, 1, 4, 8, 1, 4, 0), 2), ("qkv", (2, 1, 4, 16, 2, 2, 0), 2),
    ("o", (4, 2, 2, 8, 2, 1, 0), 1), ("o", (4, 1, 4, 4, 1, 1, 0), 1), ("o", (2, 2, 2, 8, 2, 1, 0), 2),
]
TS = [32, 17]


def main():
    ns = extract(LAYER)
    dev = "cuda"
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if ARGS.get("synthetic", "0") == "1":   # local smoke test: randn / sqrt(k) bf16 like the judge recipe, no judge import
        g = torch.Generator(device=dev).manual_seed(3)
        W = {p: (torch.randn(ns["GD_DIMS"][p], generator=g, device=dev) / math.sqrt(ns["GD_DIMS"][p][1])).to(torch.bfloat16)
             for p in ("qkv", "o", "pooler", "unpooler")}
    else:
        import judge as J  # noqa: E402  (tools/judge on the box)
        prob = J.load_problem(ROOT / "problems/muse_decode_layer")
        ref = prob.ref
        weights = ref.generate_weights(ref.WEIGHTS_SEED, device=dev, router_seed=RSEED)
        W = {p: weights[ns["GD_WEIGHT"][p]].contiguous() for p in ("qkv", "o", "pooler", "unpooler")}
    arch = ARGS.get("arch", "sm_90a")
    gencode = [f"-gencode=arch=compute_{arch[3:]},code={arch}"]
    print(f"[gdst] weights: {{p: tuple(w.shape) for p, w in W.items()}}", flush=True)
    keys, kmap = [], {}
    for proj, rt, nz in CANDS:
        n, k = W[proj].shape
        key = ns["_gd_key"](n, k, rt)
        if key is None:
            print(f"[gdst] {proj} {rt}: invalid geometry", flush=True)
            continue
        kmap[(proj, rt)] = key
        if key not in keys:
            keys.append(key)
    inst = "\n".join("    GD_INST(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(keys))
    info = "\n".join("    GD_INFO(%d, %s)" % (i, ", ".join(str(v) for v in kk)) for i, kk in enumerate(keys))
    cuda = ns["GD_CUDA_SRC"].replace("@INST@", inst).replace("@INFO@", info)
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    mod = load_inline(name="gdst_" + hashlib.sha1((ns["GD_CPP_SRC"] + cuda + arch).encode()).hexdigest()[:10],
                      cpp_sources=ns["GD_CPP_SRC"], cuda_sources=cuda, functions=["gd_run", "gd_info", "gd_unpack"],
                      extra_cuda_cflags=["-O3"] + gencode, verbose=False)
    print(f"[gdst] built {len(keys)} instances in {time.time() - t0:.0f}s", flush=True)
    flush_buf = torch.empty((256 << 20) // 4, dtype=torch.float32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(5)
    xs = {T: {k: torch.randn((T, k), generator=gen, device=dev).to(torch.bfloat16) for k in (8192, 4096, 1024)} for T in TS}

    def flush():
        flush_buf.add_(1.0)

    def timed_graph(fn):
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            flush()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts), min(ts)

    empty_med, _ = timed_graph(lambda: None)
    print(f"[gdst] empty graph {empty_med:.2f} us", flush=True)
    packs, rows = {}, []
    for proj, rt, nz in CANDS:
        if (proj, rt) not in kmap:
            continue
        key = kmap[(proj, rt)]
        inst_id = keys.index(key)
        inf = mod.gd_info(inst_id)
        n, k = W[proj].shape
        mt, wr, wk, ku, st_, s, xp = rt
        ctas = (n // 16 // wr) * s
        note = ""
        if inf[1] > 64:
            note += f" SPILL{inf[1]}"
        if inf[2] * sms < ctas * nz:
            note += f" WAVES{ctas * nz / (inf[2] * sms):.2f}"
        pk_key = (proj, wr, wk, ku, s)
        if pk_key not in packs:
            base = ns["_gd_window_base"](W[proj])
            packs[pk_key] = ns["_gd_pack"](W[proj], wr, wk, ku, s, base)
        pk = packs[pk_key]
        if s > 1 and not arch.startswith("sm_90"):
            print(f"[gdst] {proj:8s} {str(rt):26s}: cluster route skipped on {arch}", flush=True)
            continue
        for T in TS:
            if T > 8 * mt * nz:
                continue
            x = xs[T][k]
            y = torch.empty((T, n), dtype=torch.bfloat16, device=dev)
            try:
                mod.gd_run(inst_id, pk["w"], pk["esc"], x, y, T, k, n, pk["tb"], pk["nbx"], nz, 0)
                torch.cuda.synchronize()
            except Exception as ex:   # a launch error on one candidate must not end the sweep
                print(f"[gdst] {proj:8s} {str(rt):26s} nz{nz} T={T:2d}: LAUNCH FAILED {ex!r}"[:300], flush=True)
                rows.append({"proj": proj, "rt": rt, "nz": nz, "T": T, "error": repr(ex)[:200]})
                continue
            refo = x.double() @ W[proj].double().T
            rel = float((y.double() - refo).norm() / refo.norm())
            med, mn = timed_graph(lambda: mod.gd_run(inst_id, pk["w"], pk["esc"], x, y, T, k, n, pk["tb"], pk["nbx"], nz, 0))
            net = med - empty_med
            mb = pk["w"].numel() * 4 / 1e6
            row = {"proj": proj, "rt": rt, "nz": nz, "T": T, "regs": inf[0], "lmem": inf[1], "occ": inf[2], "ctas": ctas * nz,
                   "net_us": round(net, 2), "min_us": round(mn - empty_med, 2), "TBs_p12": round(mb / net, 2) if net > 0 else 0,
                   "rel": rel, "note": note.strip()}
            rows.append(row)
            print(f"[gdst] {proj:8s} {str(rt):26s} nz{nz} T={T:2d}: net {net:6.2f} us (min {mn - empty_med:6.2f}) {mb / net:.2f} TB/s P12 "
                  f"regs {inf[0]} lmem {inf[1]} occ {inf[2]} ctas {ctas * nz} rel {rel:.1e}{note}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"empty_us": empty_med, "rows": rows, "sms": sms}, indent=1))
    print(f"[gdst] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""K1 v24 lab (box only): anatomy of K1 v7's L=1 kernel on FULL layers (decode 16 / 32, ctx 80000, page 1, FP8 KV;
the layer route k1_main<1, 4>, cps 2, fold) and candidate fixes. The CUDA / host code is k1_v7.py's (== the K1 section of
kernels/muse_decode_layer/k3_v23.py), executed from source with FALLBACK forced (no import-time build); every variant is
the same source + lab patches selected by -D flags, built as its own extension, and called through k1_v7's own _k1()
host function (module swapped per variant). Judge inputs (problem generate_input), graph replay, cold L2 (1 GB read
sweep), variants replayed interleaved; check_implementation on every non-mask variant.

Masks (K1L bits, timing only, results invalid): 1 NOCOMP (main-loop tile_compute1 skipped: loads + waits + syncs only),
2 NOPV (no V frags / convert / PV mma), 4 NOQK (no K loads / convert / QK mma), 8 NOCVT (e4m3x2->f16x2 = one LOP),
16 NOMERGE (the last CTA skips merge_finalize). Builds: K1L_MINB (CTAs/SM in __launch_bounds__ of the L=1 kernel).
  python3 kernels/muse_attention_decode/k1_v24_lab.py [--shapes 5,3,1,4] [--rows ...]
"""
import argparse
import ast
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROB = HERE.parents[1] / "problems" / "muse_attention_decode"
sys.path.insert(0, str(PROB))
import reference  # noqa: E402
sys.path.insert(0, str(HERE))
import k1_v24_sk as S  # noqa: E402

SRC_FILE = HERE / "k1_v7.py"

LAB_PATCHES = [
    ("macros", "namespace k1 {\n",
     "namespace k1 {\n#ifndef K1L\n#define K1L 0\n#endif\n#ifndef K1L_MINB\n#define K1L_MINB 2\n#endif\n"),
    ("cvt", "__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {\n",
     "__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t v) {\n  if (K1L & 8) return v & 0x3C3C3C3Cu;\n"),
    ("noqk", "  float sc[NT][4];\n#pragma unroll\n  for (int jn = 0; jn < NT; ++jn) {\n    const int key = kw + 8 * jn + g;\n",
     "  float sc[NT][4];\n#pragma unroll\n  for (int jn = 0; jn < NT; ++jn) {\n"
     "    if (K1L & 4) { sc[jn][0] = 0.25f * jn; sc[jn][1] = 0.5f; sc[jn][2] = 0.f; sc[jn][3] = 0.f; continue; }\n"
     "    const int key = kw + 8 * jn + g;\n"),
    ("nopv", "#pragma unroll\n  for (int kk = 0; kk < NT / 2; ++kk) {\n    uint32_t w01[4], w23[4];\n",
     "#pragma unroll\n  for (int kk = 0; kk < NT / 2; ++kk) {\n    if (K1L & 2) { o[kk][0] += __uint_as_float(pa[kk][0] & 0x3fffffffu); continue; }\n"
     "    uint32_t w01[4], w23[4];\n"),
    ("nocomp", "      tile_compute1<2>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, need, ts, seq, seq, W);\n",
     "      if (!(K1L & 1)) tile_compute1<2>(Kb, Kb + TILE * ROWB, kw, h, g, t, qf, o, mA, lA, cl2, need, ts, seq, seq, W);\n"),
    ("nomerge", "      merge_finalize<L>(p, p.part_o + base * ROWS * 64, p.part_lse + base * ROWS, S_b, b, hp, true);\n",
     "      if (!(K1L & 16)) merge_finalize<L>(p, p.part_o + base * ROWS * 64, p.part_lse + base * ROWS, S_b, b, hp, true);\n"),
    ("minb", "__launch_bounds__(NTHR, (L == 1) ? 2 : K1_L8_MINB) k1_main(",
     "__launch_bounds__(NTHR, (L == 1) ? K1L_MINB : K1_L8_MINB) k1_main("),
]

# Candidate fix (bitwise: same math, same loads): the L=1 loader thread (key lk = tid >> 2, part = tid & 3) of warp w
# loads keys 8 (w >> 1)*2 .. of the tile, i.e. warps 2 q4, 2 q4 + 1 load exactly the 16 keys (both heads) that warps
# (h, q4) = (0, q4), (1, q4) consume. So the per-tile CTA barrier can be a 64-thread named barrier per warp pair (ids
# 1..4): cp.async completion (each thread's own wait_group) + bar.sync orders the pair's smem rows, and a slot's rows
# are rewritten only by the pair that reads them.
EXTRA = {"pair": [S.PAIR_PATCH], "pair3": [S.PAIR_PATCH], "pairnc": [S.PAIR_PATCH], "sk": S.SK_PATCHES,
         "skpair": S.SK_PATCHES + [S.PAIR_PATCH], "sect": [S.SECT_PATCH], "sectpair": [S.SECT_PATCH, S.PAIR_PATCH],
         "sectskp": S.SK_PATCHES + [S.SECT_PATCH, S.PAIR_PATCH]}
# build name -> -D flags
BUILDS = {
    "ship": [], "nocomp": ["-DK1L=1"], "nopv": ["-DK1L=2"], "noqk": ["-DK1L=4"], "nocvt": ["-DK1L=8"],
    "nomerge": ["-DK1L=16"], "minb3": ["-DK1L_MINB=3"],
    "pair": [], "pair3": ["-DK1L_MINB=3"], "pairnc": ["-DK1L=1"], "sk": [], "skpair": [],
    "sect": [], "sectpair": [], "sectskp": [],
}
MASKED = {"nocomp", "nopv", "noqk", "nocvt", "nomerge", "pairnc"}


def load_ns():
    """k1_v7.py executed with FALLBACK = True (no build at import); returns its namespace."""
    txt = SRC_FILE.read_text(encoding="utf-8")
    assert txt.count("FALLBACK = False ") == 1
    txt = txt.replace("FALLBACK = False ", "FALLBACK = True ", 1)
    for what, old, new in S.HOST_PATCHES:   # v24 host: Gx + B partial slots, fold == 2 on K1_STREAMK routes
        assert txt.count(old) == 1, what
        txt = txt.replace(old, new)
    ns = {"__name__": "k1_v7_lab"}
    exec(compile(txt, str(SRC_FILE), "exec"), ns)
    return ns


def lab_cuda(ns, patches=LAB_PATCHES):
    src = ns["CUDA_SRC"]
    for what, old, new in patches:
        n = src.count(old)
        if n != 1:
            raise RuntimeError(f"k1_v24 lab patch {what!r}: anchor found {n} times")
        src = src.replace(old, new)
    return src


def build(ns, name, flags, cuda=None):
    from torch.utils.cpp_extension import load_inline
    cuda = cuda if cuda is not None else lab_cuda(ns)
    tag = hashlib.sha1((ns["CPP_SRC"] + cuda + " ".join(flags)).encode()).hexdigest()[:10]
    t0 = time.time()
    mod = load_inline(name=f"k1v24_{name}_{tag}", cpp_sources=ns["CPP_SRC"], cuda_sources=cuda,
                      functions=["k1_forward", "k1_occupancy", "k1_prof_read"],
                      extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v", "-gencode=arch=compute_90a,code=sm_90a"] + flags,
                      verbose=False)
    print(f"[v24] built {name} {flags} in {time.time() - t0:.0f}s; occupancy L1 st4 {mod.k1_occupancy(1, 4)} "
          f"st6 {mod.k1_occupancy(1, 6)}", flush=True)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="5,3,1,4")
    ap.add_argument("--builds", default=",".join(BUILDS))
    ap.add_argument("--rows", default="")   # extra rows: build:stages:cps, e.g. ship:6:2
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default="box_out/v24/summary.json")
    args = ap.parse_args()
    ns = load_ns()
    dev = torch.device("cuda")
    benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
    mods = {nm: build(ns, nm, BUILDS[nm], lab_cuda(ns, LAB_PATCHES + EXTRA.get(nm, [])))
            for nm in args.builds.split(",")}
    if args.shapes == "none":
        print("[v24] build only", flush=True)
        return
    flush = torch.ones(256 * 1024 * 1024, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)

    def call(nm, data, stages=None, cps=None):
        ns["_MOD"] = mods[nm]
        ns["_OCC"].clear()
        return ns["_k1"](data, stages=stages, cps=cps)

    def timeit(graphs, reps):
        ts = [[] for _ in graphs]
        for _ in range(reps):
            for j, g in enumerate(graphs):
                torch.sum(flush, dim=0, out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                ts[j].append(a.elapsed_time(b) * 1000.0)
        return [statistics.median(t) for t in ts]

    eg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(eg):
        pass
    floor = timeit([eg], 50)[0]
    rows = [(nm, None, None) for nm in mods]
    for r in [x for x in args.rows.split(",") if x]:
        nm, st, cps = r.split(":")
        rows.append((nm, int(st), int(cps)))
    res = {}
    custom = {100: {"batch": 8, "ctx": 80000, "heads": "tp8", "layer": "full", "page": 1, "spread": "agentic",
                    "dist": "coherent", "load": "decode 8", "weight": 0, "seed": 777},
              101: {"batch": 24, "ctx": 20000, "heads": "tp8", "layer": "full", "page": 1, "spread": "agentic",
                    "dist": "coherent", "load": "decode 24", "weight": 0, "seed": 778},
              102: {"batch": 3, "ctx": 3000, "heads": "tp8", "layer": "full", "page": 1, "spread": "agentic",
                    "dist": "coherent", "load": "decode 3", "weight": 0, "seed": 779}}
    for i in [int(x) for x in args.shapes.split(",")]:
        bm = custom[i] if i in custom else benches[i]
        kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
        data = reference.generate_input(**kw)
        kvb = int(data["cache_seqlens"].sum()) * 256
        if not bm.get("q_len") and bm["layer"] == "full":   # split plan (k1 plan_C / balanced boundaries), as on device
            seqs = [int(x) for x in data["seq_lens"].tolist()]
            nts = [(s + 63) // 64 for s in seqs]
            T, Gx, B = sum(nts), 132 * 2, len(nts)
            lo, hi = max(2, -(-T // Gx)), max(2, -(-T // max(1, Gx - B)))
            while lo < hi:
                mid = (lo + hi) // 2
                if sum(max(1, -(-n // mid)) for n in nts) <= Gx:
                    hi = mid
                else:
                    lo = mid + 1
            S = [max(1, -(-n // lo)) for n in nts]
            mx = max(-(-n // s) for n, s in zip(nts, S))
            print(f"PLAN shape {i}: B {B} tiles {T} Gx {Gx} C {lo} CTAs used {sum(S)} | max split {mx} tiles vs mean "
                  f"{T / Gx:.1f} (x{mx / (T / Gx):.3f}); seq min/max {min(seqs)}/{max(seqs)}", flush=True)
        lab = f"[{i:2d}] {bm['load']} {bm['layer']} B{bm['batch']} ({kvb / 1e6:.0f} MB KV)"
        chk = []
        for nm, st, cps in rows:
            if nm in MASKED:
                continue
            d2 = reference.generate_input(**kw)
            out = call(nm, d2, st, cps)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(d2, reference.collect(d2, out))
            chk.append(f"{nm}{'' if st is None else f':{st}:{cps}'} {'PASS' if ok else 'FAIL ' + msg[:60]}")
            del d2
        print(f"CHK {lab}: " + " | ".join(chk), flush=True)
        graphs, names = [], []
        for nm, st, cps in rows:
            for _ in range(2):
                call(nm, data, st, cps)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call(nm, data, st, cps)
            graphs.append(g)
            names.append(nm if st is None else f"{nm}:{st}:{cps}")
        t = [x - floor for x in timeit(graphs, args.reps)]
        base = t[0]
        res[lab] = dict(zip(names, t))
        print(f"K1 {lab} | " + " | ".join(f"{n} {x:7.2f} ({x - base:+.2f}, {kvb / x / 1e6:.2f} TB/s)" for n, x in zip(names, t)),
              flush=True)
        del graphs, data
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("[v24] done", flush=True)


if __name__ == "__main__":
    main()

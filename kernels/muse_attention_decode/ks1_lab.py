"""KS1 lab (box only): one-launch B = 1 sliding attention (tools/ks1_patches.py) vs today's FA4 v8 chain, plus clock-mark anatomy.

  python3 kernels/muse_attention_decode/ks1_lab.py --mode build          # compile only (KS1 + K1 v7 with K1_PROF): `box.sh run`
  python3 kernels/muse_attention_decode/ks1_lab.py --mode alone          # self-test, judge-shape checks, standalone timing, anatomy
  python3 kernels/muse_attention_decode/ks1_lab.py --mode layer --layer kernels/muse_decode_layer/k3_v29_ks1.py
                                                                         # in-layer %globaltimer anatomy of KS1 (shapes 0, 6)
Standalone = the attention problem's judge inputs (shape 0 decode 1 sliding, shape 6 verify 1 sliding), graph replay, cold L2
(256 MB read), empty-graph floor subtracted, variants interleaved. KS1's Python glue is KS1_PY exec()'d with k1_fa4v8's helpers
(the same text the layer carries).
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROB = ROOT / "problems" / "muse_attention_decode"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROB))
sys.path.insert(0, str(ROOT / "tools"))
import ks1_patches as KP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="alone")
ap.add_argument("--layer", default="kernels/muse_decode_layer/k3_v29_ks1.py")
ap.add_argument("--shapes", default="0,6")
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--k1", type=int, default=1)
ap.add_argument("--preset", default="")   # layer mode: "NAME=json;NAME=json" module globals set before prepare
ap.add_argument("--sets", default="")     # layer mode: JSON list of {global: value} arms (anatomy per arm, after the default)
args = ap.parse_args()


def say(*a):
    print(*a, flush=True)


def ks1_namespace(helpers):
    ns = {"torch": torch, "time": time, "hashlib": hashlib, "FP8": torch.float8_e4m3fn, "ROPE_THETA": 500000.0,
          "KS1_CPP_SRC": KP.KS1_CPP_SRC, "KS1_CUDA_SRC": KP.KS1_CUDA_SRC, "__name__": "ks1_lab_ns"}
    ns.update(helpers)
    exec(compile(KP.KS1_PY, "KS1_PY", "exec"), ns)
    return ns


def k1_prof_ns():
    """k1_v7.py (== the layer's K1) executed with FALLBACK = True, rebuilt with -DK1_PROF=1 (per-CTA %globaltimer stamps)."""
    from torch.utils.cpp_extension import load_inline
    txt = (HERE / "k1_v7.py").read_text(encoding="utf-8")
    assert txt.count("FALLBACK = False ") == 1
    ns = {"__name__": "k1_v7_prof"}
    exec(compile(txt.replace("FALLBACK = False ", "FALLBACK = True ", 1), "k1_v7.py", "exec"), ns)
    tag = hashlib.sha1((ns["CPP_SRC"] + ns["CUDA_SRC"] + "prof").encode()).hexdigest()[:10]
    t0 = time.time()
    ns["_MOD"] = load_inline(name=f"k1v7prof_{tag}", cpp_sources=ns["CPP_SRC"], cuda_sources=ns["CUDA_SRC"],
                             functions=["k1_forward", "k1_occupancy", "k1_prof_read"],
                             extra_cuda_cflags=["-O3", "-lineinfo", "-DK1_PROF=1", "-gencode=arch=compute_90a,code=sm_90a"],
                             verbose=False)
    say(f"[lab] K1 v7 (K1_PROF) built in {time.time() - t0:.0f}s")
    return ns


def phases(stamps, names, ref_slot, keep=None):
    """stamps [CTAs, slots] int64 ns (0 = not reached): per slot min / median / max relative to min(stamps[:, ref_slot])."""
    st = stamps.clone()
    alive = st[:, ref_slot] > 0
    if keep is not None:
        alive &= keep
    st = st[alive]
    t0 = int(st[:, ref_slot].min())
    out = []
    for k, nm in names.items():
        col = st[:, k]
        col = col[col > 0]
        if col.numel() == 0:
            continue
        rel = (col - t0).double() / 1000.0
        out.append(f"{nm} {rel.min():.2f}/{rel.median():.2f}/{rel.max():.2f}")
    return f"{int(alive.sum())} CTAs | " + "  ".join(out) + " us (min/med/max)"


KS4_NAMES = {0: "entry", 1: "prewait", 2: "wait", 3: "qprep", 4: "tile0", 5: "attn", 6: "partial", 7: "ticket1", 8: "merge1",
             9: "ticket2", 10: "merged"}
KS1_NAMES = {0: "entry", 1: "prewait", 2: "wait", 3: "prep", 4: "sync", 5: "attn", 6: "cta-merge", 7: "partial", 8: "ticket",
             9: "merged"}


def main_alone():
    import reference
    import k1_fa4v8 as M
    dev = torch.device("cuda")
    ns = ks1_namespace({"_k1fa4_case": M._k1fa4_case, "_k1fa4_reference": M._k1fa4_reference,
                        "_k1fa4_r1_path": M._k1fa4_r1_path})
    mod = ns["_ks1_build"]()
    say(f"[lab] KS1 built in {ns['_KS1']['build_s']:.0f}s")
    for l, n, g in ns["KS1_INSTS"]:
        say(f"[lab] KS1 L{l} NS{n} NTG{g}: regs/local/smem/occ {list(mod.ks1_info(l, n, g))}")
    k1ns = k1_prof_ns() if args.k1 else None
    if args.mode == "build":
        return
    cos_sin = M.k1fa4_cos_sin(500000.0, 64, dev)
    ok_geo = ns["_ks1_self_test"](cos_sin, dev, ns["KS1_INSTS"], verbose=True)
    say(f"[lab] self-test ok geometries {sorted(ok_geo)}")
    benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
    flush = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)

    def timeit(graphs, reps):
        ts = [[] for _ in graphs]
        for _ in range(reps):
            for j, gr in enumerate(graphs):
                torch.sum(flush, dim=0, out=sink)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                gr.replay()
                b.record()
                b.synchronize()
                ts[j].append(a.elapsed_time(b) * 1000.0)
        return [statistics.median(t) for t in ts]

    def capture(fn):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        return gr

    eg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(eg):
        pass
    floor = timeit([eg], 50)[0]
    say(f"[lab] empty-graph floor {floor:.2f} us")
    for i in [int(x) for x in args.shapes.split(",")]:
        bm = benches[i]
        kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
        L = bm.get("q_len", 1)
        geos = [(n, g) for l, n, g in sorted(ok_geo) if l == L]
        # correctness on the judge's inputs (2 seeds): check_implementation of KS1 per geometry and of FA4 v8
        for sd in (0, 1):
            msgs = []
            for nm, fn in [("fa4", None)] + [(f"ks1_{n}x{g}", (n, g)) for n, g in geos]:
                d = reference.generate_input(**{**kw, "seed": kw["seed"] + 1000 * sd})
                if fn is None:
                    out = M.k1fa4_run(d)
                else:
                    out = ns["_ks1_call"](d["qkv"].view(L, -1), d, cos_sin, fn, 0, True).view(1, L, -1)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(d, reference.collect(d, out))
                msgs.append(f"{nm} {'PASS' if ok else 'FAIL'} ({msg[:70]})")
            say(f"[lab] CHK [{i}] {bm['load']} seed+{1000 * sd}: " + " | ".join(msgs))
        data = reference.generate_input(**kw)
        qkv = data["qkv"].view(L, -1)
        rows = [("fa4", capture(lambda: M.k1fa4_run(data)))]
        for n, g in geos:
            for pdl in (True,):
                rows.append((f"ks1_{n}x{g}", capture(lambda n=n, g=g, pdl=pdl: ns["_ks1_call"](qkv, data, cos_sin, (n, g), 0, pdl))))
        if k1ns is not None and L == 1:
            k1ns["_OCC"].clear()
            rows.append(("k1v7", capture(lambda: k1ns["_k1"](data, fold=True))))
        t = [x - floor for x in timeit([r[1] for r in rows], args.reps)]
        say(f"[lab] TIME [{i}] {bm['load']} {bm['layer']} L{L}: " + " | ".join(f"{r[0]} {x:6.2f}" for r, x in zip(rows, t)))
        # KS1 standalone anatomy (stamps per CTA, relative to the first CTA's entry)
        for n, g in geos:
            prof = torch.zeros(16 * 2 * g * (n + 1), dtype=torch.int64, device=dev)
            ns["KS1_PROF_BUF"] = prof
            for mode in ("cold", "warm"):
                acc = []
                for _ in range(5):
                    if mode == "cold":
                        torch.sum(flush, dim=0, out=sink)
                    else:   # code + data in L2 / icache: the same call twice back to back, stamps of the second
                        ns["_ks1_call"](qkv, data, cos_sin, (n, g), 0, True)
                    prof.zero_()
                    ns["_ks1_call"](qkv, data, cos_sin, (n, g), 0, True)
                    torch.cuda.synchronize()
                    acc.append(prof.view(-1, 16).cpu().clone())
                say(f"[lab] ANAT ks1_{n}x{g} [{i}] {mode} (rel. first entry; last run) {phases(acc[-1], KS1_NAMES, 0)}")
            ns["KS1_PROF_BUF"] = None
        if k1ns is not None and L == 1:
            k1ns["_MOD"].k1_prof_read()
            for _ in range(3):
                torch.sum(flush, dim=0, out=sink)
                k1ns["_k1"](data, fold=True)
                torch.cuda.synchronize()
                st = k1ns["_MOD"].k1_prof_read().view(4096, 16)[:, :8]
            names = {0: "entry", 1: "plan", 2: "q+newkv", 3: "loop-done", 4: "partial-smem", 5: "ticket", 7: "merge-start",
                     6: "merged"}
            act = st[:, 2] > 0
            say(f"[lab] ANAT k1v7 [{i}] active CTAs: {phases(st, names, 0, keep=act)}")
            say(f"[lab] ANAT k1v7 [{i}] all CTAs:    {phases(st, {0: 'entry', 1: 'plan'}, 0)}")
        del rows, data
        torch.cuda.empty_cache()


def main_layer():
    sys.path.insert(0, str(ROOT / "tools" / "judge"))
    sys.path.insert(0, str(ROOT / "tools" / "box"))
    import judge as J
    from layer_budget import FIELDS
    prob = J.load_problem(ROOT / "problems" / "muse_decode_layer")
    ref = prob.ref
    for b in prob.benches:
        b.update(FIELDS[b["load"]])
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=8677534258059892491)
    mod = J.load_kernel(ROOT / args.layer, "okc_submission", prob)
    for kv in [x for x in args.preset.split(";") if x]:
        k, v = kv.split("=", 1)
        setattr(mod, k, json.loads(v))
    state = mod.prepare(J.clone_data(weights))
    timer = J.GpuTimer()
    say(f"[lab] layer ks1 state {sorted(state.get('ks1') or [])}")

    def keyfix(v):
        if isinstance(v, dict):
            return {(int(k) if isinstance(k, str) and k.isdigit() else k): keyfix(x) for k, x in v.items()}
        return v
    arms = [("default", {})] + [(f"set{i}", keyfix(x)) for i, x in enumerate(json.loads(args.sets or "[]"))]
    for idx, (arm, sets) in [(i, a) for i in [int(x) for x in args.shapes.split(",")] for a in arms]:
        case = prob.benches[idx]
        d0 = J.make_input(ref, case, case["seed"], weights, "cuda")
        L = d0["config"]["q_len"]
        saved = {k: getattr(mod, k) for k in sets}
        for k, x in sets.items():
            setattr(mod, k, x)
        full = not d0["config"]["window"]
        n, g = (mod.KS4_GEOM[L][1], 0) if full else mod.KS1_GEOM[L]
        prof = torch.zeros(16 * 1024, dtype=torch.int64, device="cuda")
        mod.KS1_PROF_BUF = prof
        st = J.clone_data(d0)
        try:
            gr, _ = timer.capture(lambda dd: mod.kernel_fn(dd, state), st)
        finally:
            mod.KS1_PROF_BUF = None
            for k, x in saved.items():
                setattr(mod, k, x)
        say(f"[lab] arm {arm}: {sets}")
        runs = []
        for r in range(12):
            prof.zero_()
            timer.time_replay(gr)
            torch.cuda.synchronize()
            runs.append(prof.view(-1, 16).cpu().clone())
        # summary over runs: per-run spans relative to the earliest wait return
        spans, pre, post = [], [], []
        last = 10 if full else 9
        for x in runs[2:]:
            x = x[x[:, 2] > 0]
            w = x[:, 2]
            t0 = int(w.min())
            spans.append((int(x[:, last].max()) - t0) / 1000.0)
            pre.append((t0 - int(x[:, 0].min())) / 1000.0)
            post.append((int(w.max()) - t0) / 1000.0)
        say(f"[lab] LAYER [{idx}] {case['load']} {case['layer']} L{L} NS{n} NTG{g}: wait-return -> merged "
            f"{statistics.median(spans):.2f} us (median of {len(spans)}); first entry -> first wait-return {statistics.median(pre):.2f} us; "
            f"wait-return spread {statistics.median(post):.2f} us")
        say(f"[lab] LAYER [{idx}] last run (rel. first wait-return): {phases(runs[-1], KS4_NAMES if full else KS1_NAMES, 2)}")
        del gr, st


if args.mode == "layer":
    main_layer()
else:
    main_alone()
say("[lab] done")

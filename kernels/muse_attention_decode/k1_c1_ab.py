"""C1 same-process A/B on the box: k1_v4 (incumbent, its own build) vs k1_v9c (cluster merge, and its v4 merge in the
same build), judge-style: one CUDA graph per (shape, variant), 256 MB L2 flush before every timed replay, variants
interleaved rep by rep (drift cancels), empty-graph replay subtracted, median per variant + median paired ratio vs v4.
Every variant is checked with the problem's reference.check_implementation on the same inputs, and outputs are compared
bit for bit with k1_v4's (the cluster merge is expected to be bit-identical on the scored sliding shapes).

  python3 kernels/muse_attention_decode/k1_c1_ab.py                       # dec1/16/32 sliding + full, ver1/16/32 sliding (lab)
  python3 kernels/muse_attention_decode/k1_c1_ab.py --shapes 0,2,4 --reps 60
  python3 kernels/muse_attention_decode/k1_c1_ab.py --prof                 # + K1_PROF phase stamps (patched copies, one cold call)
  python3 kernels/muse_attention_decode/k1_c1_ab.py --json box_out/c1_ab.json
  python3 kernels/muse_attention_decode/k1_c1_ab.py --selftest             # correctness only (box.sh run, no lock)

Variants (per shape kind; see VARIANTS):
  v4      k1_v4._k1 (L=1) / k1_v4._k1 L=8 kernel (unrouted, lab)     c0    k1_v9c, v4 merge (cluster=False)
  c1      k1_v9c cluster merge, table defaults (pad1 at sliding L=1)   c1np  cluster, no smem padding
  c1r1    cluster, relaxed arrive at barrier 1                         c1r3  cluster, relaxed arrive at both barriers
  c1m2    cluster, min_chunk 2 (L=8: 16 two-tile splits at ver1)       r1    rung-1 FA3 path (the routed L=8 incumbent)
Verdict (the brief): GATE = every sliding L=1 shape saves >= 2 us (v4 - c1, same process) and no full-layer shape gets
slower in the v9c build (c0 vs v4 > +1%); KILL = mean saving < 1.5 us, or a full-layer regression. Else MARGINAL.
"""
import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROB = ROOT / "problems" / "muse_attention_decode"
sys.path.insert(0, str(PROB))
import reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="0,2,4,1,3,5,6,8,10", help="meta.json benchmark indices")
ap.add_argument("--reps", type=int, default=40)
ap.add_argument("--variants", default="", help="comma list overriding VARIANTS for every shape")
ap.add_argument("--prof", action="store_true")
ap.add_argument("--json", default="")
ap.add_argument("--selftest", action="store_true", help="only the two self-tests (correctness, no timing, no lock)")
args = ap.parse_args()


def load(path, name, prof=False):
    if prof:   # K1_PROF build: a patched copy (different source hash -> separate extension)
        text = Path(path).read_text().replace("#define K1_PROF 0 ", "#define K1_PROF 1 ")
        assert "#define K1_PROF 1 " in text, "K1_PROF define not found"
        tmp = Path(tempfile.mkdtemp(prefix="c1prof_")) / Path(path).name
        tmp.write_text(text)
        path = tmp
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dev = torch.device("cuda")
V4 = load(HERE / "k1_v4.py", "k1v4")
C9 = load(HERE / "k1_v9c.py", "k1v9c")
assert V4._self_test(dev), "k1_v4 self-test failed"
assert C9._self_test(dev), "k1_v9c self-test failed (v4 cases)"
print(f"AB k1_v9c cluster_ok={C9._STATE['cluster_ok']}", flush=True)
if args.selftest:
    assert C9._STATE["cluster_ok"][1], "cluster self-test failed for L=1"
    sys.exit(0)
PV4 = load(HERE / "k1_v4.py", "k1v4p", prof=True) if args.prof else None
PC9 = load(HERE / "k1_v9c.py", "k1v9cp", prof=True) if args.prof else None
if args.prof:
    assert PV4._self_test(dev) and PC9._self_test(dev)


def variants_for(bm):
    L = bm.get("q_len", 1)
    if args.variants:
        names = args.variants.split(",")
    elif L == 1 and bm["layer"] == "sliding":
        names = ["v4", "c0", "c1", "c1np", "c1r1", "c1r3"]
    elif L == 1:
        names = ["v4", "c0", "c1"]          # full: v9c routes the v4 merge (c0); c1 is the lab cluster row
    else:   # L=8 sliding: v4's unrouted L=8 kernel, v4 vs cluster merge; c1m2 = 2-tile splits (cs up to 16)
        names = ["r1", "v4", "c0", "c1", "c1r3", "c1m2"]
    return names


def fn_of(name, mods):
    v4, c9 = mods
    return {
        "v4": lambda d: v4._k1(d),
        "r1": lambda d: v4._fa3_path(d),
        "c0": lambda d: c9._k1(d, cluster=False),
        "c1": lambda d: c9._k1(d, cluster=True),
        "c1np": lambda d: c9._k1(d, cluster=True, pad1=False),
        "c1r1": lambda d: c9._k1(d, cluster=True, relax=1),
        "c1r3": lambda d: c9._k1(d, cluster=True, relax=3),
        "c1m2": lambda d: c9._k1(d, cluster=True, min_chunk=2),
    }[name]


flush_buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
sink = torch.empty((), dtype=torch.float32, device=dev)


def time_one(g):
    torch.sum(flush_buf, dim=0, out=sink)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000.0


def capture(fn, data):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(data)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(data)
    return g


eg = torch.cuda.CUDAGraph()
with torch.cuda.graph(eg):
    pass
floor = statistics.median(time_one(eg) for _ in range(50))
print(f"AB empty-graph replay {floor:.2f} us (subtracted)", flush=True)

benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
report = {"floor_us": floor, "shapes": []}
for i in [int(x) for x in args.shapes.split(",") if x]:
    bm = benches[i]
    kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
    L = bm.get("q_len", 1)
    label = f"[{i:2d}] {bm['load']:9s} {bm['layer']:7s} B{bm['batch']} L{L}"
    data = reference.generate_input(**kw)
    names = variants_for(bm)
    outs, checks, graphs = {}, {}, {}
    for nm in names:
        fn = fn_of(nm, (V4, C9))
        try:
            out = fn(data)
            torch.cuda.synchronize()
            pick = C9._STATE.get("last_pick") if nm.startswith("c1") else None
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
            outs[nm] = out.clone()
            checks[nm] = (ok, msg[:70], pick)
            graphs[nm] = capture(fn, data)
        except Exception as ex:  # noqa: BLE001
            checks[nm] = (False, f"ERR {str(ex)[:100]}", None)
    if args.prof:
        for nm in [n for n in names if n in graphs and n != "r1"]:
            mod = PV4 if nm == "v4" else PC9
            fn = fn_of(nm, (PV4, PC9))
            for _ in range(3):
                fn(data)
            mod._MOD.k1_prof_read()
            torch.sum(flush_buf, dim=0, out=sink)
            fn(data)
            pr = mod._MOD.k1_prof_read().view(4096, 16).double()
            v = pr[pr[:, 0] > 0]
            t0 = v[:, 0].min()
            parts = []
            for sl in range(8):
                col = v[:, sl]
                col = col[col > 0]
                if len(col):
                    d = (col - t0) / 1000.0
                    parts.append(f"s{sl} med {d.median():.2f} max {d.max():.2f} (n{len(col)})")
            print(f"PROF {label} {nm} ctas={len(v)}: " + "; ".join(parts), flush=True)
    ts = {nm: [] for nm in graphs}
    for _ in range(args.reps):                       # interleaved: one replay of each variant per round
        for nm in graphs:
            ts[nm].append(time_one(graphs[nm]) - floor)
    base = "v4" if "v4" in ts else next(iter(ts))
    row = {"idx": i, "label": label, "variants": {}}
    cells = []
    for nm in names:
        if nm not in ts:
            cells.append(f"{nm} {checks[nm][1]}")
            row["variants"][nm] = {"error": checks[nm][1]}
            continue
        med = statistics.median(ts[nm])
        ratio = statistics.median(b / max(x, 1e-3) for b, x in zip(ts[base], ts[nm])) if nm != base else 1.0
        same = bool(torch.equal(outs[nm], outs[base])) if base in outs else None
        dmax = float((outs[nm].float() - outs[base].float()).abs().max()) if base in outs else float("nan")
        ok, msg, pick = checks[nm]
        flag = "" if ok else f" CHECK-FAIL({msg})"
        cs = f" cs={pick[0]}{'p' if pick[1] else ''}" if pick else ""
        cells.append(f"{nm} {med:7.2f}{cs} x{ratio:.3f}{' =' if same else f' d{dmax:.1e}'}{flag}")
        row["variants"][nm] = {"us": med, "ratio_vs_" + base: ratio, "bit_equal_" + base: same, "maxdiff": dmax,
                               "check": ok, "pick": pick, "samples": ts[nm]}
    print(f"AB {label} | " + " | ".join(cells), flush=True)
    report["shapes"].append(row)
    del data, graphs
    torch.cuda.empty_cache()

# verdict per the brief
sv = [r for r in report["shapes"] if "sliding" in r["label"] and r["label"].endswith("L1")]
fu = [r for r in report["shapes"] if "full" in r["label"] and r["label"].endswith("L1")]
saves = [(r["label"], r["variants"]["v4"]["us"] - r["variants"]["c1"]["us"]) for r in sv
         if "us" in r["variants"].get("v4", {}) and "us" in r["variants"].get("c1", {})]
regress = [(r["label"], r["variants"]["c0"]["us"] / r["variants"]["v4"]["us"] - 1) for r in fu
           if "us" in r["variants"].get("v4", {}) and "us" in r["variants"].get("c0", {})]
if saves:
    mean = sum(s for _, s in saves) / len(saves)
    worst_full = max((x for _, x in regress), default=0.0)
    verdict = ("GATE" if all(s >= 2.0 for _, s in saves) and worst_full <= 0.01 else
               "KILL" if mean < 1.5 or worst_full > 0.01 else "MARGINAL")
    print("AB saved per sliding L=1 call (v4 - c1): " + ", ".join(f"{' '.join(l.split()[2:6])} {s:+.2f} us" for l, s in saves)
          + f"; mean {mean:+.2f} us; full-layer v9c-build drift (c0/v4 - 1): "
          + ", ".join(f"{x:+.2%}" for _, x in regress) + f" -> {verdict}", flush=True)
    report["verdict"] = {"saves_us": saves, "mean_us": mean, "full_drift": regress, "verdict": verdict}
if args.json:
    Path(args.json).write_text(json.dumps(report, indent=1))
    print(f"AB wrote {args.json}", flush=True)

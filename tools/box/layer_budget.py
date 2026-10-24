"""Per-op time budget of the decode layer: stock (baseline.py) vs ours (k3_v07), all 12 judge shapes.

Box (one process, nsys only records the profiled section):
  nsys profile -o box_out/lb --force-overwrite true -t cuda,nvtx --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      python3 tools/box/layer_budget.py run kernels/muse_decode_layer/k3_v07.py [--repeats 20] [--replays 5]
  nsys export --type sqlite -o box_out/lb.sqlite box_out/lb.nsys-rep
Laptop (no torch needed):
  python3 tools/box/layer_budget.py analyze box_out/lb.sqlite box_out/lb_meta.json > notes/...

What `run` does, per scored shape, exactly as tools/judge/judge.py run_benchmarks does it:
  * weights: generate_weights(WEIGHTS_SEED), then weights_fields with the task.yml routing targets back-solved in
    notes/study/01-challenge.md (router_load = simulated mean over random routers, router_distinct = back-solved from
    the published floors), so the router is re-drawn to the hidden load targets like the real judge does (the
    plain clone without --fields keeps the lighter WEIGHTS_SEED router);
  * inputs: generate_input + case_fields (local_start = busiest 32-expert range), seed + 42 (rep + 1);
  * one CUDA graph per side on its own static buffers; per repetition fresh inputs copied into both, 256 MB L2
    flush read before every replay, CUDA-event timing, order swapped every repetition, replay cost subtracted,
    median of paired (sum) ratios  -> the judge-style paired speedup and µs per shape;
  * then (nsys capture range on) for each shape and side: R graph replays, each after an L2 flush, inside NVTX
    ranges "G|side|idx|r", and one EAGER run with marker kernels (torch.cuda._sleep -> spin_kernel) before and
    after every op, inside NVTX range "M|side|idx". The eager run labels every kernel with its op (per stream);
    the graph kernels are then matched to those labels by (name, grid, block) in start order.
`analyze` builds per-op exclusive time (main-stream kernel if one runs, else side-stream kernel, else idle gap),
which sums exactly to the graph's GPU span, and per-op floors.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------------------------
# Constants (notes/lessons.md §5-6, problems/muse_decode_layer/reference.py)
# ----------------------------------------------------------------------------------------------
BW = 4.3e12          # streaming floor rate (lessons §5: 4.29-4.48 TB/s with >=32 KiB/SM in flight)
BW_GATHER = 3.6e12   # random 128 B row gathers (paged KV, page 1) (lessons §6: 3.61 TB/s)
PEAK = 989e12        # BF16 dense
RAMP = 3.0           # µs, one streaming launch's fixed cost (lessons §6: t = 3.05 µs + bytes / 4.74 TB/s)
LAUNCH = 1.0         # µs, each extra small launch in a graph (lessons §6: +0.97 µs)
HIDDEN, QKV, QS, KVS, DEXP = 8192, 2304, 1024, 128, 4096
EXPERT_BYTES = 3 * DEXP * DEXP * (0.5 + 1 / 16)   # W13 + W2 + E4M3 group scales, one expert
FIELDS = {"decode 1": {"router_load": 2.57, "router_distinct": 2.50},
          "decode 16": {"router_load": 30.9, "router_distinct": 7.53},
          "decode 32": {"router_load": 60.9, "router_distinct": 9.00},
          "verify 1": {"router_load": 14.14, "router_distinct": 7.50},
          "verify 16": {"router_load": 205.5, "router_distinct": 18.45},
          "verify 32": {"router_load": 409.5, "router_distinct": 21.35}}

OPS = ["qkv", "qknorm_kv", "attention", "combine", "o_norm_gate", "o_proj", "attn_boundary", "router", "pooler",
       "pre_expert_norm", "v8_tail", "moe_align", "w13", "swiglu", "w2", "finalize", "experts_fused", "unpooler",
       "ffn_boundary", "gaps"]
OP_DESC = {
    "qkv": "qkv_gate GEMM", "qknorm_kv": "QK-norm (+RoPE), q->FP8, K/V FP8 store", "attention": "attention main",
    "combine": "split-KV combine", "o_norm_gate": "per-head RMSNorm x sigmoid(gate)", "o_proj": "o GEMM",
    "attn_boundary": "post-attn norm + residual + pre-FFN norm", "router": "router GEMM(s) + top-8 (side stream, exposed part)",
    "pooler": "pooler GEMM", "pre_expert_norm": "pre-expert RMSNorm", "v8_tail": "V8 tail: routing + pre-expert norm + local ids", "moe_align": "local ids + moe_align/sort",
    "w13": "Marlin W13", "swiglu": "silu_and_mul", "w2": "Marlin W2", "finalize": "post-expert norm weighted sum",
    "experts_fused": "K2A v2.2 (route+W13+SwiGLU+W2+finalize)", "unpooler": "unpooler GEMM (+pad/zeros)",
    "ffn_boundary": "post-FFN norm + residual + next pre-attn norm", "gaps": "idle GPU inside the graph"}


# ----------------------------------------------------------------------------------------------
# run (box)
# ----------------------------------------------------------------------------------------------
class Marker:
    """Emits a spin_kernel on the current stream before/after each wrapped op when on (eager only)."""

    def __init__(self):
        self.on = False
        self.log = []   # (cuda_stream handle, "B"/"E", label) in emission order

    def mark(self, kind, label):
        if self.on:
            import torch
            s = torch.cuda.current_stream().cuda_stream
            torch.cuda._sleep(1)
            self.log.append((int(s), kind, label))


M = Marker()


def wrap(label, fn):
    def w(*a, **k):
        lab = label(*a, **k) if callable(label) else label
        M.mark("B", lab)
        r = fn(*a, **k)
        M.mark("E", lab)
        return r
    w.__wrapped__ = fn
    return w


class Proxy:
    def __init__(self, label, obj):
        self._label, self._obj = label, obj

    def __call__(self, *a, **k):
        return wrap(self._label, self._obj)(*a, **k)

    def __getattr__(self, n):
        v = getattr(self._obj, n)
        return wrap(self._label, v) if callable(v) else v


def gemm_label(x, w, *a, **k):
    n, kk = w.shape[0], w.shape[1]
    return {(2304, 8192): "qkv", (8192, 1024): "o_proj", (4096, 8192): "pooler"}.get((n, kk), "unpooler")


def marlin_label(*a, **k):
    return "w13" if k.get("size_n") == 2 * DEXP else "w2"


def instrument(mod, state):
    g = mod.__dict__
    for name, lab in (("_gemm", gemm_label), ("store_cache", "store_cache"), ("flash_attn_with_kvcache", "fa3"),
                      ("moe_align_block_size", "moe_align"), ("moe_wna16_marlin_gemm", marlin_label),
                      ("silu_and_mul", "swiglu"), ("fused_post_expert_norm_weighted_sum", "finalize"),
                      ("_post_expert_norm_weighted_sum", "finalize"), ("_k1_run", "k1"), ("_k2a_experts", "k2a"),
                      ("_v8_gemm", lambda x, layer, *a, **k: {"qkv_gate": "qkv", "pooler": "pooler"}.get(layer, "unpooler")),
                      ("_v8_logits", "router_logits"), ("_v8_tail", "v8_tail")):
        if name in g and callable(g[name]):
            g[name] = wrap(lab, g[name])
    for key, lab in (("qk_norm", "qk_norm"), ("o_norm_gate", "o_norm_gate"), ("attn_boundary", "attn_boundary"),
                     ("routing", "routing"), ("pre_expert_norm", "pre_expert_norm"), ("ffn_boundary", "ffn_boundary")):
        if key in state:
            state[key] = Proxy(lab, state[key])


def arg(argv, name, default):
    for a in argv:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
    return default


def run(argv):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judge"))
    import judge as J
    sub_path = Path(argv[0])
    repeats = arg(argv, "repeats", 20)
    replays = arg(argv, "replays", 5)
    shapes_sel = arg(argv, "shapes", "")
    router_seed = arg(argv, "router-seed", -1)
    out_meta = Path(arg(argv, "meta", "box_out/lb_meta.json"))
    t00 = time.time()
    prob = J.load_problem(Path("problems/muse_decode_layer"))
    ref = prob.ref
    for b in prob.benches:
        b.update(FIELDS[b["load"]])
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    if router_seed < 0:
        t0 = time.time()
        router_seed = ref.weights_fields(weights, prob.benches, 0)["router_seed"]
        print(f"[lb] weights_fields -> router_seed {router_seed} ({time.time() - t0:.0f}s)", flush=True)
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=router_seed)
    base = J.load_kernel(prob.dir / "baseline.py", "okc_baseline", prob)
    sub = J.load_kernel(sub_path, "okc_submission", prob)
    states = {"base": base.prepare(J.clone_data(weights)), "sub": sub.prepare(J.clone_data(weights))}
    instrument(base, states["base"])
    instrument(sub, states["sub"])
    calls = {"base": lambda d: base.kernel_fn(d, states["base"]), "sub": lambda d: sub.kernel_fn(d, states["sub"])}
    timer = J.GpuTimer()
    ov = timer.replay_overhead_us()
    print(f"[lb] ready in {time.time() - t00:.0f}s; replay overhead {ov:.2f} us; router_seed {router_seed}", flush=True)
    idxs = [int(s) for s in shapes_sel.split(",")] if shapes_sel else list(range(len(prob.benches)))
    meta = {"router_seed": router_seed, "overhead_us": ov, "sub": str(sub_path), "shapes": {}}
    keep = {}
    for idx in idxs:
        case = prob.benches[idx]
        t0 = time.time()
        seed0 = case["seed"]
        d0 = J.make_input(ref, case, seed0, weights, "cuda")
        static = {s: J.clone_data(d0) for s in ("base", "sub")}
        graphs = {s: timer.capture(calls[s], static[s]) for s in ("base", "sub")}
        b_raw, s_raw, routes, distinct = [], [], [], []
        for rep in range(repeats):
            data = J.make_input(ref, case, seed0 + J.SEED_STEP * (rep + 1), weights, "cuda")
            for s in ("base", "sub"):
                J.copy_into(static[s], data)
            torch.cuda.synchronize()
            t = {}
            for s in J.order_for(rep):
                t[s] = timer.time_replay(graphs[s][0])
            b_raw.append(t["base"])
            s_raw.append(t["sub"])
            ids = graphs["base"][1][2]
            loc = (ids.long() - int(data["local_start"])) % 256
            loc = loc[loc < 32]
            routes.append(int(loc.numel()))
            distinct.append(int(torch.unique(loc).numel()))
        sr = J.ShapeResult(case["load"], case["weight"], J.subtract_overhead(b_raw, ov), J.subtract_overhead(s_raw, ov))
        cfg = static["base"]["config"]
        sl = static["base"]["seq_lens"].tolist()
        info = {"load": case["load"], "weight": case["weight"], "layer": case["layer"], "batch": cfg["batch"],
                "q_len": cfg["q_len"], "tokens": cfg["tokens"], "seq_lens": sl,
                "base_us": sr.base_us, "sub_us": sr.sub_us, "speedup": sr.speedup, "base_med": sr.base_median,
                "sub_time": sr.time_us, "sub_med": statistics.median(sr.sub_us),
                "routes_mean": statistics.mean(routes), "distinct_mean": statistics.mean(distinct),
                "routes_last": routes[-1], "distinct_last": distinct[-1]}
        meta["shapes"][str(idx)] = info
        keep[idx] = (static, graphs)
        print(f"[lb] [{idx:2d}] {case['load']:>10s} {case['layer']:>7s} T={cfg['tokens']:3d}: sglang {sr.base_median:7.1f}"
              f"  ours {sr.time_us:7.1f} us  x{sr.speedup:.3f}  routes {info['routes_mean']:.1f} distinct"
              f" {info['distinct_mean']:.2f}  ({repeats} reps, {time.time() - t0:.0f}s)", flush=True)
        out_meta.write_text(json.dumps(meta, indent=1))
    # ---- nsys section ----
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for idx in idxs:
        static, graphs = keep[idx]
        for s in ("base", "sub"):
            for r in range(replays):
                with torch.cuda.stream(timer.stream):
                    timer.flush()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push(f"G|{s}|{idx}|{r}")
                with torch.cuda.stream(timer.stream):
                    graphs[s][0].replay()
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_pop()
            # eager, marked (inputs = a clone of the static buffers: same lengths, routing)
            d = J.clone_data(static[s])
            with torch.cuda.stream(timer.stream):
                timer.flush()
            torch.cuda.synchronize()
            M.log = []
            M.on = True
            torch.cuda.nvtx.range_push(f"M|{s}|{idx}")
            with torch.cuda.stream(timer.stream):
                calls[s](d)
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            M.on = False
            meta["shapes"][str(idx)][f"marks_{s}"] = M.log
            meta["shapes"][str(idx)][f"main_stream_{s}"] = int(timer.stream.cuda_stream)
            del d
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    out_meta.write_text(json.dumps(meta, indent=1))
    print(f"[lb] done in {time.time() - t00:.0f}s -> {out_meta}", flush=True)


# ----------------------------------------------------------------------------------------------
# analyze (laptop)
# ----------------------------------------------------------------------------------------------
def load_sqlite(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    strings = dict(cur.execute("select id, value from StringIds")) if "StringIds" in tables else {}
    cols = [r[1] for r in cur.execute("pragma table_info(NVTX_EVENTS)")]
    tcol = "text" if "text" in cols else None
    q = f"select start, end, {tcol or 'NULL'}, {'textId' if 'textId' in cols else 'NULL'} from NVTX_EVENTS where end is not null"
    ranges = []
    for st, en, txt, tid in cur.execute(q):
        t = txt if txt is not None else strings.get(tid)
        if t and (t.startswith("G|") or t.startswith("M|")):
            ranges.append((st, en, t))
    ranges.sort()
    rt = {}
    for tbl in ("CUPTI_ACTIVITY_KIND_RUNTIME",):
        for cid, st in cur.execute(f"select correlationId, start from {tbl}"):
            rt[cid] = st
    kcols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    namecol = "demangledName" if "demangledName" in kcols else "shortName"
    ks = []
    for st, en, sid, cid, nm, gx, gy, gz, bx, by, bz in cur.execute(
            f"select start, end, streamId, correlationId, {namecol}, gridX, gridY, gridZ, blockX, blockY, blockZ"
            f" from CUPTI_ACTIVITY_KIND_KERNEL"):
        ks.append({"start": st, "end": en, "stream": sid, "cid": cid, "name": strings.get(nm, str(nm)),
                   "key": (strings.get(nm, str(nm)), gx, gy, gz, bx, by, bz), "launch": rt.get(cid, st)})
    # memsets / memcpys inside the graph (torch.zeros, copies) are GPU work too, not idle time
    for tbl, tag in (("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]")):
        if tbl not in tables:
            continue
        for st, en, sid, cid, nb in cur.execute(f"select start, end, streamId, correlationId, bytes from {tbl}"):
            ks.append({"start": st, "end": en, "stream": sid, "cid": cid, "name": f"{tag} {nb} B",
                       "key": (tag, nb, 0, 0, 0, 0, 0), "launch": rt.get(cid, st)})
    # assign kernels to NVTX ranges by the launching API call's timestamp
    import bisect
    starts = [r[0] for r in ranges]
    by_range = defaultdict(list)
    for k in ks:
        i = bisect.bisect_right(starts, k["launch"]) - 1
        if i >= 0 and ranges[i][0] <= k["launch"] <= ranges[i][1]:
            by_range[ranges[i][2]].append(k)
    for v in by_range.values():
        v.sort(key=lambda k: k["start"])
    return by_range


def short(name, n=70):
    return name if len(name) <= n else name[: n - 3] + "..."


def label_eager(kernels, marks, main_handle):
    """Per nsys stream, walk kernels in order; spin_kernels are the markers. Returns [(kernel, label, role)]."""
    host = defaultdict(list)
    for h, kind, lab in marks:
        host[h].append((kind, lab))
    per_stream = defaultdict(list)
    for k in kernels:
        per_stream[k["stream"]].append(k)
    # map nsys streams to host streams by marker count
    counts = {s: sum("spin_kernel" in k["name"] for k in v) for s, v in per_stream.items()}
    mapping = {}
    for s, c in counts.items():
        cands = [h for h, v in host.items() if len(v) == c and h not in mapping.values()]
        if c and cands:
            mapping[s] = cands[0]
    out = []
    for s, v in per_stream.items():
        h = mapping.get(s)
        role = "main" if h == main_handle else "side"
        seq = host.get(h, [])
        mi, cur, last_end = 0, None, None
        for k in v:
            if "spin_kernel" in k["name"]:
                kind, lab = seq[mi] if mi < len(seq) else ("?", "?")
                mi += 1
                if kind == "B":
                    cur = lab
                else:
                    cur, last_end = None, lab
                continue
            if cur is not None:
                lab = cur
            else:
                nxt = next((seq[j][1] for j in range(mi, len(seq)) if seq[j][0] == "B"), "$")
                lab = f"{last_end or '^'}>{nxt}"
            out.append((k, lab, role))
    return out


def bucket(label, name, role):
    n = name.lower()
    if label == "k1":
        return "combine" if "combine" in n else "attention"
    if label == "fa3":
        return "combine" if "combine" in n else "attention"
    if label == "k2a":
        return "experts_fused"
    if label in ("qk_norm", "store_cache"):
        return "qknorm_kv"
    if label == "v8_tail":
        return "v8_tail"
    if label in ("routing", "router_logits") or role == "side":
        return "router"
    direct = {"qkv": "qkv", "o_proj": "o_proj", "o_norm_gate": "o_norm_gate", "attn_boundary": "attn_boundary",
              "pooler": "pooler", "pre_expert_norm": "pre_expert_norm", "moe_align": "moe_align", "w13": "w13",
              "w2": "w2", "swiglu": "swiglu", "finalize": "finalize", "unpooler": "unpooler",
              "ffn_boundary": "ffn_boundary"}
    if label in direct:
        return direct[label]
    if ">" in label:
        a, b = label.split(">", 1)
        if b == "ffn_boundary":
            return "unpooler"                       # F.pad / zeros(+mm) of the sliced unpooler
        if a in ("qk_norm", "store_cache") or b == "store_cache":
            return "qknorm_kv"                      # k/v/q casts to FP8
        if a == "fa3":
            return "attention"                      # FA3 output cast to BF16
        if b == "v8_tail" and a == "attn_boundary":
            return "pooler"                         # V8 "cat": [pooler; router hi; lo] GEMM
        if b in ("pre_expert_norm", "moe_align") or a == "pooler":
            return "moe_align"                      # local ids (expert_map / table)
        return bucket(b, name, role) if b in direct or b in ("k1", "fa3", "k2a", "routing", "qk_norm") else "unknown:" + label
    return "unknown:" + label


def exclusive(kernels):
    """kernels: [(start, end, op, role)]. Returns ({op: µs exclusive}, span µs, {op: busy µs}, gaps list)."""
    ev = sorted({k[0] for k in kernels} | {k[1] for k in kernels})
    t0, t1 = ev[0], ev[-1]
    excl = defaultdict(float)
    busy = defaultdict(float)
    for s, e, op, role in kernels:
        busy[op] += (e - s) / 1e3
    srt = sorted(kernels, key=lambda k: k[0])
    for a, b in zip(ev[:-1], ev[1:]):
        act = [k for k in srt if k[0] <= a and k[1] >= b]
        if not act:
            excl["gaps"] += (b - a) / 1e3
            continue
        mains = [k for k in act if k[3] == "main"]
        pick = (mains or act)[0]
        excl[pick[2]] += (b - a) / 1e3
    return excl, (t1 - t0) / 1e3, busy


def floors(info):
    """Per-op standalone floor in µs (bytes / BW + ramp, or FLOP / peak if larger) and the per-op bytes."""
    T, B, L, layer = info["tokens"], info["batch"], info["q_len"], info["layer"]
    att = [min(s + L, 2048) if layer == "sliding" else s + L for s in info["seq_lens"]]
    kv_bytes = sum(att) * KVS * 2
    att_flops = 4.0 * sum(att) * L * QS
    d = info["distinct_mean"]
    unp_n = 1024 if T <= 8 else HIDDEN

    def gemm(n, k):
        by = n * k * 2 + T * k * 2 + T * n * 2
        return by, 2.0 * T * n * k

    rows = {}
    rows["qkv"] = gemm(QKV, HIDDEN)
    rows["qknorm_kv"] = (T * QKV * 2 + T * (QS + 2 * KVS), 0)
    rows["attention"] = (kv_bytes, att_flops)
    rows["combine"] = (0, 0)
    rows["o_norm_gate"] = (T * (QS * 2 + QS * 2 + QS * 2), 0)
    rows["o_proj"] = gemm(HIDDEN, QS)
    rows["attn_boundary"] = (T * HIDDEN * (4 + 2 + 4 + 2 + (4 if T <= 8 else 0)), 0)
    rows["router"] = (256 * HIDDEN * 4 + T * HIDDEN * 2, 2.0 * T * 256 * HIDDEN * (1 if T <= 8 else 2))
    rows["pooler"] = gemm(DEXP, HIDDEN)
    rows["pre_expert_norm"] = (T * DEXP * 4, 0)
    rows["v8_tail"] = (0, 0)
    rows["moe_align"] = (T * 8 * 8, 0)
    rows["w13"] = (d * EXPERT_BYTES * 2 / 3, 2.0 * info["routes_mean"] * 2 * DEXP * DEXP)
    rows["swiglu"] = (info["routes_mean"] * DEXP * 2 * 3, 0)
    rows["w2"] = (d * EXPERT_BYTES / 3, 2.0 * info["routes_mean"] * DEXP * DEXP)
    rows["finalize"] = (info["routes_mean"] * DEXP * 2 + T * DEXP * 2, 0)
    rows["unpooler"] = gemm(unp_n, DEXP)
    rows["ffn_boundary"] = (T * HIDDEN * (4 + 2 + 4 + 2), 0)
    big = {"qkv", "attention", "o_proj", "pooler", "w13", "w2", "unpooler"}
    out = {}
    for op, (by, fl) in rows.items():
        bw = BW_GATHER if op == "attention" else BW
        t_mem = by / bw * 1e6
        t_fl = fl / PEAK * 1e6
        if op == "combine":
            out[op] = (0.0, 0.0, by)
            continue
        ramp = RAMP if op in big else LAUNCH
        out[op] = (max(t_mem, t_fl) + ramp, max(t_mem, t_fl), by)   # (standalone floor, marginal floor, bytes)
    out["experts_fused"] = (out["w13"][1] + out["w2"][1] + RAMP, out["w13"][1] + out["w2"][1],
                            out["w13"][2] + out["w2"][2])
    out["gaps"] = (0.0, 0.0, 0)
    return out


def analyze(argv):
    db, meta_p = argv[0], argv[1]
    meta = json.loads(Path(meta_p).read_text())
    by_range = load_sqlite(db)
    res = {}
    detail = []
    for idx, info in sorted(meta["shapes"].items(), key=lambda kv: int(kv[0])):
        res[idx] = {"info": info}
        for side in ("base", "sub"):
            if f"marks_{side}" not in info:
                continue
            eager = by_range.get(f"M|{side}|{idx}", [])
            lab = label_eager(eager, info[f"marks_{side}"], info[f"main_stream_{side}"])
            pool = defaultdict(list)
            for k, l, role in lab:
                pool[k["key"]].append((k, l, role))
            per_rep_excl, per_rep_busy, spans, kern_rows = [], [], [], None
            r = 0
            while f"G|{side}|{idx}|{r}" in by_range:
                gk = by_range[f"G|{side}|{idx}|{r}"]
                used = defaultdict(int)
                rows, unmatched = [], 0
                for k in gk:
                    cands = pool.get(k["key"], [])
                    j = used[k["key"]]
                    if j < len(cands):
                        _, l, role = cands[j]
                        used[k["key"]] += 1
                    else:
                        l, role = "unmatched", "main"
                        unmatched += 1
                    rows.append((k["start"], k["end"], bucket(l, k["name"], role), role, k["name"], l))
                excl, span, busy = exclusive([(a, b, op, role) for a, b, op, role, _, _ in rows])
                per_rep_excl.append(excl)
                per_rep_busy.append(busy)
                spans.append(span)
                if r == 0:
                    kern_rows = rows
                    if unmatched or len(gk) != len(lab):
                        print(f"WARN shape {idx} {side}: graph kernels {len(gk)} eager {len(lab)} unmatched {unmatched}",
                              file=sys.stderr)
                r += 1
            if not spans:
                continue
            ops = set().union(*per_rep_excl)
            med = {op: statistics.median([e.get(op, 0.0) for e in per_rep_excl]) for op in ops}
            medb = {op: statistics.median([e.get(op, 0.0) for e in per_rep_busy]) for op in ops}
            res[idx][side] = {"excl": med, "busy": medb, "span": statistics.median(spans), "nrep": len(spans)}
            t0 = kern_rows[0][0]
            prev_end = t0
            for a, b, op, role, nm, l in kern_rows:
                detail.append(f"| {idx} | {side} | {(a - t0) / 1e3:7.1f} | {(b - a) / 1e3:6.1f} | {max(0, a - prev_end) / 1e3:5.1f} |"
                              f" {role} | {op} | {l} | {short(nm)} |")
                if role == "main":
                    prev_end = max(prev_end, b)
        res[idx]["floor"] = floors(info)
    report(res, meta, detail)
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "floor"}
                                             | {"floor": {o: list(x) for o, x in v["floor"].items()}}
                                             for k, v in res.items()}, indent=1))


def fmt(x, w=6, p=1):
    return f"{x:{w}.{p}f}"


# ----------------------------------------------------------------------------------------------
# judge numbers and the what-if engine (shared by `analyze` and `prior`)
# ----------------------------------------------------------------------------------------------
JUDGE_MD = "kernels/results/208b5e1e1ade4698.md"   # k3_v07 judge check, 1.218x
N_DEP = 7   # grid-wide dependencies in one layer (attn<-qkv, o<-attn, norm<-o, router/pooler<-norm,
            # experts<-routing+pooler norm, W2<-W13, unpooler<-experts; the last norm folds into the unpooler pass)
GROUPS = {"qkv": ["qkv"], "attn_block": ["qknorm_kv", "attention", "combine", "o_norm_gate"], "o_proj": ["o_proj"],
          "attn_boundary": ["attn_boundary"], "router": ["router"], "pooler": ["pooler"],
          "pre_expert_norm": ["pre_expert_norm", "v8_tail"], "experts": ["moe_align", "w13", "swiglu", "w2", "finalize", "experts_fused"],
          "unpooler": ["unpooler"], "ffn_boundary": ["ffn_boundary"], "gaps": ["gaps"]}
GROUP_ORDER = list(GROUPS)
K2A_LEVERS = {1: 34, 8: 86, 16: 116, 32: 152, 128: 260, 256: 352}   # k2a_gap_analysis.md §3, all four levers
BIG_GROUPS = {"qkv", "attn_block", "o_proj", "pooler", "experts", "unpooler"}


def judge_table(path=JUDGE_MD):
    rows = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("| decode") or ln.startswith("| verify"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            rows.append({"load": c[0], "shape": c[1], "weight": float(c[2]), "stock": float(c[3]), "ours": float(c[4])})
    return rows


def group_floors(f):
    """Per group: (standalone floor = one kernel at floor, marginal = bytes or FLOP time only)."""
    out = {}
    for g, ops in GROUPS.items():
        if g == "experts":
            sf, mf = f["experts_fused"][0] + LAUNCH, f["experts_fused"][1]   # one fused kernel (+ ids/align launch)
        elif g == "attn_block":
            mf = sum(f[o][1] for o in ops)
            sf = mf + RAMP                                                    # one K1-style kernel
        else:
            sf = sum(f[o][0] for o in ops)
            mf = sum(f[o][1] for o in ops)
        out[g] = (sf, mf)
    return out


def layer_speedup(shapes, times):
    """shapes: [{load, weight, stock}], times: per-shape total µs. Geomean over loads of weighted sums."""
    loads = defaultdict(lambda: [0.0, 0.0])
    for sh, t in zip(shapes, times):
        loads[sh["load"]][0] += sh["weight"] * sh["stock"]
        loads[sh["load"]][1] += sh["weight"] * t
    return math.exp(sum(math.log(b / y) for b, y in loads.values()) / len(loads))


def fused_ideal(fl):
    """Every op at its marginal floor, one ramp, one ~1 µs grid-wide sync per dependency."""
    return sum(m for g, (s, m) in fl.items() if g != "gaps") + RAMP + N_DEP * LAUNCH


def whatifs(shapes, P=print, order=None, big=None):
    """shapes: [{load, weight, stock, ours: {group: µs}, floor: {group: (standalone, marginal)}}] (ours calibrated)."""
    GROUP_ORDER = order or list(GROUPS)  # noqa: N806
    BIG_GROUPS = big or {"qkv", "attn_block", "o_proj", "pooler", "experts", "unpooler"}  # noqa: N806
    tot = lambda s, o: sum(o.values())  # noqa: E731
    cur = layer_speedup(shapes, [tot(s, s["ours"]) for s in shapes])
    P(f"Current (calibrated to the judge): **{cur:.3f}x**\n")
    P("### One group at a time (others as ours)\n")
    P("| group | weighted ours µs/step | weighted gap to standalone floor µs/step | layer x if group -> standalone floor | layer x if group -> marginal floor | layer x if group -> 0 | group speedup needed for 1.75x | for 2.0x |")
    P("|---|---|---|---|---|---|---|---|")
    for g in GROUP_ORDER:
        if not any(g in s["ours"] for s in shapes):
            continue
        wo = sum(s["weight"] * s["ours"].get(g, 0) for s in shapes) / 6
        wg = sum(s["weight"] * max(0.0, s["ours"].get(g, 0) - s["floor"].get(g, (0, 0))[0]) for s in shapes) / 6

        def with_(fn):
            return layer_speedup(shapes, [tot(s, s["ours"]) - s["ours"].get(g, 0) + fn(s) for s in shapes])

        x_sf = with_(lambda s: min(s["ours"].get(g, 0), s["floor"].get(g, (0, 0))[0]))
        x_mf = with_(lambda s: min(s["ours"].get(g, 0), s["floor"].get(g, (0, 0))[1]))
        x_0 = with_(lambda s: 0.0)
        need = []
        for tgt in (1.75, 2.0):
            if x_0 < tgt:
                need.append("impossible alone")
                continue
            lo, hi = 1.0, 1e6
            for _ in range(80):
                mid = math.sqrt(lo * hi)
                if with_(lambda s: s["ours"].get(g, 0) / mid) >= tgt:
                    hi = mid
                else:
                    lo = mid
            need.append(f"{hi:.2f}x")
        P(f"| {g} | {wo:.0f} | {wg:.0f} | {x_sf:.3f} | {x_mf:.3f} | {x_0:.3f} | {need[0]} | {need[1]} |")
    P("")
    P("### Scenarios\n")
    sc = {}
    sc["all gaps -> 0"] = [tot(s, s["ours"]) - s["ours"].get("gaps", 0) for s in shapes]
    small = [g for g in GROUP_ORDER if g not in BIG_GROUPS and g != "gaps"]
    sc["small ops (norms, router, boundaries) fused away: -> marginal floor, gaps -> 0"] = [
        sum(v for g, v in s["ours"].items() if g in BIG_GROUPS) + sum(s["floor"][g][1] for g in small) for s in shapes]
    sc["router folded into the pooler pass (router exposed -> 0) and gaps -> 0, rest as ours"] = [
        tot(s, s["ours"]) - s["ours"].get("router", 0) - s["ours"].get("gaps", 0) for s in shapes]
    sc["realistic glue: big groups except experts at standalone floor, small ops fused, gaps 0, experts as ours"] = [
        sum(s["floor"][g][0] for g in BIG_GROUPS if g != "experts") + sum(s["floor"][g][1] for g in small)
        + s["ours"].get("experts", 0) for s in shapes]
    sc["realistic glue but attention as ours"] = [
        sum(s["floor"][g][0] for g in BIG_GROUPS if g not in ("experts", "attn_block")) + sum(s["floor"][g][1] for g in small)
        + s["ours"].get("experts", 0) + s["ours"].get("attn_block", 0) for s in shapes]
    sc["every group at its standalone floor (no fusion)"] = [sum(s["floor"][g][0] for g in GROUP_ORDER if g != "gaps")
                                                             for s in shapes]
    sc["CEILING with current experts: everything else fused-ideal, experts as ours"] = [
        fused_ideal({g: v for g, v in s["floor"].items() if g != "experts"}) + s["ours"].get("experts", 0)
        for s in shapes]
    sc["CEILING with experts at the K2A 'all four levers' estimate (k2a_gap_analysis §3, standalone µs)"] = [
        fused_ideal({g: v for g, v in s["floor"].items() if g != "experts"}) + K2A_LEVERS[s["tokens"]]
        for s in shapes] if all("tokens" in s for s in shapes) else None
    sc = {k: v for k, v in sc.items() if v is not None}
    sc["fused-ideal layer (physics: all bytes at 4.3 TB/s, KV at 3.6, one ramp, 7 syncs)"] = [
        fused_ideal(s["floor"]) for s in shapes]
    for k, ts in sc.items():
        P(f"- {k}: **{layer_speedup(shapes, ts):.3f}x**")
    P("")
    P("### Experts speedup needed (rest fixed at a scenario)\n")
    P("| rest of the layer at | experts speedup needed for 1.75x | for 2.0x | experts µs/step weighted now |")
    P("|---|---|---|---|")
    wexp = sum(s["weight"] * s["ours"].get("experts", 0) for s in shapes) / 6
    rests = {"as ours": lambda s: tot(s, s["ours"]) - s["ours"].get("experts", 0),
             "gaps -> 0": lambda s: tot(s, s["ours"]) - s["ours"].get("experts", 0) - s["ours"].get("gaps", 0),
             "small ops fused away": lambda s: sum(v for g, v in s["ours"].items() if g in BIG_GROUPS and g != "experts")
             + sum(s["floor"][g][1] for g in small),
             "realistic glue (big at standalone floor, small fused, gaps 0)": lambda s: sum(
                 s["floor"][g][0] for g in BIG_GROUPS if g != "experts") + sum(s["floor"][g][1] for g in small),
             "realistic glue, attention as ours": lambda s: sum(
                 s["floor"][g][0] for g in BIG_GROUPS if g not in ("experts", "attn_block"))
             + sum(s["floor"][g][1] for g in small) + s["ours"].get("attn_block", 0),
             "router folded + gaps 0, rest as ours": lambda s: tot(s, s["ours"]) - s["ours"].get("experts", 0)
             - s["ours"].get("router", 0) - s["ours"].get("gaps", 0),
             "fused-ideal": lambda s: fused_ideal({g: v for g, v in s["floor"].items() if g != "experts"})}
    for name, rest in rests.items():
        need = []
        for tgt in (1.75, 2.0):
            f = lambda k: layer_speedup(shapes, [rest(s) + s["ours"].get("experts", 0) / k for s in shapes])  # noqa: E731
            if f(1e9) < tgt:
                need.append("impossible")
                continue
            lo, hi = 0.01, 1e9
            for _ in range(100):
                mid = math.sqrt(lo * hi)
                lo, hi = (lo, mid) if f(mid) >= tgt else (mid, hi)
            need.append(f"{hi:.2f}x")
        P(f"| {name} | {need[0]} | {need[1]} | {wexp:.0f} |")
    P("")
    P("### Uniform speedup of all of ours needed: "
      + ", ".join(f"{t}x -> {t / cur:.3f}x on every op" for t in (1.75, 2.0)) + "\n")


def report(res, meta, detail):
    P = print
    P(f"router_seed {meta['router_seed']}; replay overhead {meta['overhead_us']:.2f} us; sub {meta['sub']}\n")
    jt = judge_table()
    P("## Per-shape budgets (µs; nsys graph replay, cold L2; exclusive time: main-stream kernel > side-stream kernel"
      " > idle, so rows sum to the GPU span)\n")
    shapes = []
    for idx, r in sorted(res.items(), key=lambda kv: int(kv[0])):
        info = r["info"]
        if "base" not in r or "sub" not in r:
            continue
        b, s, f = r["base"], r["sub"], r["floor"]
        j = jt[int(idx)]
        ev_b, ev_s = statistics.median(info["base_us"]), statistics.median(info["sub_us"])
        name = f"{info['load']} {info['layer']} (T={info['tokens']}, w {info['weight']})"
        tgt = FIELDS[info["load"]]
        P(f"### [{idx}] {name}")
        P(f"- box paired event timing: SGLang {info['base_med']:.1f} µs, ours {info['sub_time']:.1f} µs, x{info['speedup']:.3f}"
          f" (judge 208b5e: {j['stock']:.1f} / {j['ours']:.1f}, x{j['stock'] / j['ours']:.3f})")
        P(f"- nsys span: SGLang {b['span']:.1f} µs ({b['span'] / ev_b:.2f}x event median {ev_b:.1f}), ours {s['span']:.1f} µs"
          f" ({s['span'] / ev_s:.2f}x event median {ev_s:.1f}); {b['nrep']} replays")
        P(f"- routing on the timed inputs: local routes {info['routes_mean']:.1f} (task.yml target {tgt['router_load']}),"
          f" distinct local experts {info['distinct_mean']:.2f} (target {tgt['router_distinct']})\n")
        P("| op | stock µs | ours µs | standalone floor µs | marginal floor µs | stock − floor | ours − floor | stock busy | ours busy |")
        P("|---|---|---|---|---|---|---|---|---|")
        ops_present = [op for op in OPS if b["excl"].get(op, 0) > 0.05 or s["excl"].get(op, 0) > 0.05]
        extra = sorted((set(b["excl"]) | set(s["excl"])) - set(OPS))
        tb = ts = 0.0
        for op in ops_present + extra:
            eb, es = b["excl"].get(op, 0.0), s["excl"].get(op, 0.0)
            fl = f.get(op, (0.0, 0.0, 0))
            tb += eb
            ts += es
            P(f"| {op} | {fmt(eb)} | {fmt(es)} | {fmt(fl[0])} | {fmt(fl[1])} | {fmt(eb - fl[0])} | {fmt(es - fl[0])} |"
              f" {fmt(b['busy'].get(op, 0))} | {fmt(s['busy'].get(op, 0))} |")
        gf = group_floors(f)
        P(f"| **total** | **{tb:.1f}** | **{ts:.1f}** | {sum(v[0] for v in gf.values()):.1f} (groups standalone) |"
          f" {fused_ideal(gf):.1f} (fused-ideal) | | | | |\n")
        j_ours = j["ours"] if meta["sub"].endswith("k3_v07.py") else info["sub_time"] * j["stock"] / info["base_med"]
        cb, cs = j["stock"] / tb, j_ours / ts
        grp = lambda d, c: {g: c * sum(d.get(o, 0.0) for o in ops) for g, ops in GROUPS.items()}  # noqa: E731
        j = dict(j, ours=j_ours)
        shapes.append({"idx": idx, "load": info["load"], "weight": info["weight"], "stock": j["stock"], "tokens": info["tokens"],
                       "ours": grp(s["excl"], cs), "stock_ops": grp(b["excl"], cb), "floor": gf, "cal": (cb, cs),
                       "name": name})
    if shapes:
        P("## Grouped, calibrated to the judge (each side's exclusive µs scaled so the shape total = judge µs)\n")
        P("| shape | " + " | ".join(GROUP_ORDER) + " | total | cal |")
        P("|---|" + "---|" * (len(GROUP_ORDER) + 2))
        for sh in shapes:
            for side, key, c in (("stock", "stock_ops", sh["cal"][0]), ("ours", "ours", sh["cal"][1])):
                P(f"| {sh['name']} {side} | " + " | ".join(f"{sh[key][g]:.1f}" for g in GROUP_ORDER)
                  + f" | {sum(sh[key].values()):.1f} | {c:.3f} |")
            P(f"| {sh['name']} floor (standalone/marginal) | " + " | ".join(f"{sh['floor'][g][0]:.1f}/{sh['floor'][g][1]:.1f}"
                                                                         for g in GROUP_ORDER) + " | | |")
        P("")
        P("## Judge-weighted (46 sliding / 16 full) totals per group, µs per step / 6 loads\n")
        P("| group | stock | ours | standalone floor | ours − floor |")
        P("|---|---|---|---|---|")
        for g in GROUP_ORDER:
            ws = sum(x["weight"] * x["stock_ops"][g] for x in shapes) / 6
            wo = sum(x["weight"] * x["ours"][g] for x in shapes) / 6
            wf = sum(x["weight"] * x["floor"][g][0] for x in shapes) / 6
            P(f"| {g} | {ws:.0f} | {wo:.0f} | {wf:.0f} | {wo - wf:.0f} |")
        P("")
        whatifs(shapes, P)
    if detail:
        P("## Kernel trace, first replay of each graph (start µs from first kernel, dur, idle before on main)\n")
        P("| shape | side | t | dur | gap | stream | op | label | kernel |")
        P("|---|---|---|---|---|---|---|---|---|")
        for d in detail:
            P(d)



# ----------------------------------------------------------------------------------------------
# prior (laptop): a first budget from the judge's own per-kernel profiles (no box)
# ----------------------------------------------------------------------------------------------
PRIOR_ORDER = ["dense", "attn_block", "norms", "experts", "router", "other"]
PRIOR_BIG = {"dense", "attn_block", "experts"}


def parse_profile(path):
    """okc profile txt -> {shape idx: {"rows": [(name, µs, calls)], "total": µs}} (top-12 rows + total)."""
    out, cur = {}, None
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("profile "):
            cur = int(ln.split()[1].rstrip(":"))
            out[cur] = {"rows": [], "total": None}
        elif cur is not None and ln.startswith("total"):
            out[cur]["total"] = float(ln.split()[1])
        elif cur is not None and " µs " in ln and not ln.startswith("GPU busy") and not ln.startswith("kernel"):
            head, tail = ln.rsplit(" µs ", 1)
            name, us = head.rsplit(None, 1)
            out[cur]["rows"].append((name.strip(), float(us), int(tail.split()[-1])))
    return out


def prior_group(name):
    n = name
    if any(k in n for k in ("k2a2", "marlin_moe", "moe_align", "act_and_mul", "_post_expert_norm", "indexSelect",
                            "_scatter_gather")):
        return "experts"
    if "flash::" in n or "k1::" in n:
        return "attn_block"
    if any(k in n for k in ("dot_kernel", "sm80_xmma_gemm_f32f32", "routing_kernel")) or ("nvjet_sm90_tss" in n and "_NN" in n):
        return "router"
    if "splitKreduce" in n:
        return "router" if "float, fl" in n else "dense"
    if "nvjet" in n:
        return "dense"
    if "vectorized_layer_norm" in n or "norm_residual" in n:
        return "norms"
    return "other"


def prior(argv):
    stock_p = argv[0] if argv else "kernels/muse_decode_layer/profiles/r0_baseline.d487f513b5fc476e.txt"
    ours_p = argv[1] if len(argv) > 1 else "kernels/muse_decode_layer/profiles/k3_v07.208b5e1e1ade4698.txt"
    benches = json.loads(Path("problems/muse_decode_layer/meta.json").read_text())["benchmarks"]
    jt = judge_table()
    prof = {"stock": parse_profile(stock_p), "ours": parse_profile(ours_p)}
    P = print
    P(f"Sources: stock profile {stock_p}, ours profile {ours_p}, judge µs {JUDGE_MD}.\n")
    P("Per shape, in-situ kernel GPU time from the judge's profile (top 12 kernels per shape; PDL waits are inside a"
      " kernel's time). `other` = judge µs − listed non-router groups: the hidden small kernels (qk-norm, casts,"
      " store_cache, routing glue, silu, ...), idle gaps and exposed router time. `router` is listed but not counted"
      " in the total (side stream, overlaps the pooler).\n")
    P("| shape | side | dense GEMMs (qkv+o+pooler+unpooler) | attn block | norms | experts | router (side) | other | judge µs |")
    P("|---|---|---|---|---|---|---|---|---|")
    shapes = []
    for i, b in enumerate(benches):
        L = b.get("q_len", 1)
        info = {"tokens": b["batch"] * L, "batch": b["batch"], "q_len": L, "layer": b["layer"],
                "seq_lens": [b["ctx"]] * b["batch"], "routes_mean": FIELDS[b["load"]]["router_load"],
                "distinct_mean": FIELDS[b["load"]]["router_distinct"]}
        f = floors(info)
        gf = group_floors(f)
        pf = {"dense": tuple(sum(gf[g][k] for g in ("qkv", "o_proj", "pooler", "unpooler")) for k in (0, 1)),
              "attn_block": gf["attn_block"], "experts": gf["experts"], "router": gf["router"],
              "norms": tuple(sum(gf[g][k] for g in ("attn_boundary", "pre_expert_norm", "ffn_boundary")) for k in (0, 1)),
              "other": (0.0, 0.0)}
        rec = {"load": b["load"], "weight": b["weight"], "stock": jt[i]["stock"], "tokens": info["tokens"], "floor": pf}
        for side in ("stock", "ours"):
            g = defaultdict(float)
            for name, us, calls in prof[side][i]["rows"]:
                g[prior_group(name)] += us
            j = jt[i][side]
            listed = sum(v for k, v in g.items() if k not in ("router", "other"))
            row = {k: g.get(k, 0.0) for k in PRIOR_ORDER if k not in ("router", "other")}
            row["other"] = j - listed
            rec[side if side == "ours" else "stock_ops"] = row
            P(f"| {b['load']} {b['layer']} | {side} | {row['dense']:.1f} | {row['attn_block']:.1f} | {row['norms']:.1f} |"
              f" {row['experts']:.1f} | {g.get('router', 0):.1f} | {row['other']:.1f} | {j:.1f} |")
        P(f"| {b['load']} {b['layer']} | floor standalone/marginal | {pf['dense'][0]:.1f}/{pf['dense'][1]:.1f} |"
          f" {pf['attn_block'][0]:.1f}/{pf['attn_block'][1]:.1f} | {pf['norms'][0]:.1f}/{pf['norms'][1]:.1f} |"
          f" {pf['experts'][0]:.1f}/{pf['experts'][1]:.1f} | {pf['router'][0]:.1f}/{pf['router'][1]:.1f} | 0 |"
          f" fused-ideal {fused_ideal(gf):.1f} |")
        shapes.append(rec)
    P("")
    P("### Judge-weighted per-group µs per step (Σ weight × µs / 6 loads)\n")
    P("| group | stock | ours | standalone floor | ours − floor |")
    P("|---|---|---|---|---|")
    for g in PRIOR_ORDER:
        if g == "router":
            continue
        ws = sum(x["weight"] * x["stock_ops"][g] for x in shapes) / 6
        wo = sum(x["weight"] * x["ours"][g] for x in shapes) / 6
        wf = sum(x["weight"] * x["floor"][g][0] for x in shapes) / 6
        P(f"| {g} | {ws:.0f} | {wo:.0f} | {wf:.0f} | {wo - wf:.0f} |")
    P("")
    order = [g for g in PRIOR_ORDER if g != "router"]
    for x in shapes:
        x["floor"] = {g: x["floor"][g] for g in order}
    whatifs(shapes, P, order=order, big=PRIOR_BIG)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run":
        run(sys.argv[2:])
    elif mode == "analyze":
        analyze(sys.argv[2:])
    elif mode == "prior":
        prior(sys.argv[2:])
    else:
        raise SystemExit(__doc__)

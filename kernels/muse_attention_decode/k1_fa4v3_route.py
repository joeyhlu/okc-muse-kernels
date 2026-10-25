"""K1-FA4 v3 (decode probe): the v2 kernel (k1_fa4v2.fwd, imported unchanged) on q_len = 1 decode steps, full
kernel_fn path (r1's FusedQKNorm/RoPE + FP8 KV store prologue, rms_norm_gate epilogue), against K1 v4 (fused
decode kernel) and r1 in one process.

At L = 1, pack-GQA gives 8 query rows per KV head; FA4's sm90 forward needs tile_m to be a multiple of 64 (one
wgmma warpgroup per 64 rows), so the tile is 64 x 128 with 8 live rows. The compute is wasted but small; the
cost is per-tile latency, which split-KV spreads.

  python3 kernels/muse_attention_decode/k1_fa4v3_route.py [--shapes 0,2,4,1,3,5] [--splits 2,4,8] [--reps 40]
"""
import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROB = ROOT / "problems" / "muse_attention_decode"
sys.path.insert(0, str(HERE))
import r1_glue as R1  # noqa: E402

_STATE = {"k2": None}


def _k2():
    if _STATE["k2"] is None:
        import k1_fa4v2
        _STATE["k2"] = k1_fa4v2
    return _STATE["k2"]


def route_ok(cfg):
    return (cfg["q_len"] in (1, 8) and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"])


def route(data, splits=None):
    """r1 prologue -> k1_fa4v2.fwd (any q_len the kernel's pack-GQA tile holds: 1 or 8) -> r1 epilogue."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    width = hkv * d
    k_stage = torch.empty((tokens + 1, width), dtype=R1.FP8, device=device)
    v_stage = torch.empty((tokens + 1, width), dtype=R1.FP8, device=device)
    kv_store = (k_stage, v_stage, R1._stage_loc(tokens, device))
    if cfg["rope_theta"]:
        q8, _ = R1._qk_norm(hq, hkv, d).forward_triton(
            qkv, data["positions"], R1._cos_sin(cfg["rope_theta"], d, device), kv_store)
    else:
        q8, _ = R1._qk_norm(hq, hkv, d).forward_triton(qkv, None, None, kv_store)
    R1.store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                   k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                   data["cache_loc"], reserved_skip_index=-1)
    k2 = _k2()
    max_k = data["page_table"].shape[1]
    keys = min(max_k, window + length) if window else max_k
    ns = splits or k2.num_splits_for(batch, hkv, keys)
    out = k2.fwd(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"], data["cache_seqlens"],
                 max_k, cfg["sm_scale"] * cfg["qk_scale_factor"], ns, window=window)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = R1.rms_norm_gate(out.reshape(tokens, hq * d), gate, None, head_dim=d, eps=R1.RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)


SELF_TEST_SHAPES = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding", page=1, seed=31),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding", page=1, seed=32),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1, seed=33),
    dict(batch=9, ctx=5000, heads="tp8", layer="sliding", spread="mixed", page=1, seed=34),
    dict(batch=4, ctx=3000, heads="tp8", layer="full", page=1, seed=35),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1, seed=36),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="0,2,4,1,3,5")
    ap.add_argument("--splits", default="2,4,8")
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args()
    sys.path.insert(0, str(PROB))
    import reference
    dev = torch.device("cuda")

    t0 = time.time()
    ok_all = True
    for kw in SELF_TEST_SHAPES:
        try:
            data = reference.generate_input(**kw)
            out = route(data)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
        except Exception as ex:  # noqa: BLE001
            ok, msg = False, f"{type(ex).__name__}: {str(ex)[:300]}"
            print("\n".join("TB| " + ln for ln in traceback.format_exc()[-2000:].splitlines()), flush=True)
        ok_all &= bool(ok)
        print(f"[k1fa4v3] self-test {kw}: {'PASS' if ok else 'FAIL'} {str(msg)[:200]}", flush=True)
    print(f"[k1fa4v3] self-test {'PASS' if ok_all else 'FAIL'} in {time.time() - t0:.1f}s", flush=True)

    t0 = time.time()
    spec = importlib.util.spec_from_file_location("k1v4_lab", str(HERE / "k1_v4.py"))
    V4 = importlib.util.module_from_spec(spec)
    sys.modules["k1v4_lab"] = V4
    spec.loader.exec_module(V4)
    v4_ok = V4._self_test(dev)
    print(f"[k1fa4v3] k1_v4 build + self-test {'PASS' if v4_ok else 'FAIL'} in {time.time() - t0:.1f}s", flush=True)

    benches = json.loads((PROB / "meta.json").read_text())["benchmarks"]
    flush_buf = torch.ones(256 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
    sink = torch.empty((), dtype=torch.float32, device=dev)

    def time_graph(g, reps):
        ts = []
        for _ in range(reps):
            torch.sum(flush_buf, dim=0, out=sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        return statistics.median(ts)

    eg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(eg):
        pass
    floor = time_graph(eg, 50)
    for i in [int(x) for x in args.shapes.split(",")]:
        bm = benches[i]
        kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
        variants = [("r1", R1.kernel_fn), ("v4", V4.kernel_fn), ("fa4auto", lambda d: route(d))]
        for s in [int(x) for x in args.splits.split(",") if x]:
            variants.append((f"fa4s{s}", lambda d, s=s: route(d, splits=s)))
        row = []
        for name, fn in variants:
            try:
                data = reference.generate_input(**kw)
                out = fn(data)
                torch.cuda.synchronize()
                ok, msg = reference.check_implementation(data, reference.collect(data, out))
                for _ in range(2):
                    fn(data)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn(data)
                t = time_graph(g, args.reps) - floor
                row.append(f"{name} {t:7.2f}us{'' if ok else ' FAIL(' + str(msg)[:80] + ')'}")
                del g, data
            except Exception as ex:  # noqa: BLE001
                row.append(f"{name} ERR {type(ex).__name__}: {str(ex)[:200]}")
                print("\n".join("TB| " + ln for ln in traceback.format_exc()[-2000:].splitlines()), flush=True)
            torch.cuda.empty_cache()
        print(f"BENCH [{i:2d}] {bm['load']} {bm['layer']} (full kernel_fn, graph replay, cold L2) | "
              + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()

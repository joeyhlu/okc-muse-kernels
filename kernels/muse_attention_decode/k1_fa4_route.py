"""K1-FA4 drop-in route for L=8 (verify) full-attention layers: r1_glue.kernel_fn with the FA3 call replaced by
k1_fa4.fwd (FA4 CuTe-DSL sm90 forward, 2 CTAs/SM, FP8 K/V gathered and upcast to F16 in smem, F16 wgmma).
Everything outside the route (decode, sliding, page 64, scaled caches, dp heads, B < ROUTE_MIN_BATCH) goes
to r1_glue.kernel_fn unchanged. Design and measurements: k1_fa4_design.md.

Prologue and epilogue are r1's (FusedQKNorm Triton with the FP8 KV-store epilogue + store_cache, then
rms_norm_gate). Queries: r1's E4M3 q (stock precision, the A16 assumption) cast exactly to F16.

  python3 kernels/muse_attention_decode/k1_fa4_route.py [--reps 30]   # self-test (forced route) + timing
"""
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_glue as R1  # noqa: E402

ROUTE_MIN_BATCH = 16  # measured at B 16 / 32 only (k1_fa4_design.md); smaller batches stay on r1
MAX_SPLITS = 128
_STATE = {"ready": None, "k1": None}


def _k1():
    if _STATE["k1"] is None:
        import k1_fa4
        _STATE["k1"] = k1_fa4
    return _STATE["k1"]


def route_ok(cfg, force=False):
    return (cfg["q_len"] == 8 and cfg["window"] == 0 and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0
            and cfg["head_dim"] == 64 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
            and (force or cfg["batch"] >= ROUTE_MIN_BATCH))


def _route(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    width = hkv * d
    k_stage = torch.empty((tokens + 1, width), dtype=R1.FP8, device=device)
    v_stage = torch.empty((tokens + 1, width), dtype=R1.FP8, device=device)
    kv_store = (k_stage, v_stage, R1._stage_loc(tokens, device))
    q8, _ = R1._qk_norm(hq, hkv, d).forward_triton(qkv, None, None, kv_store)
    R1.store_cache(k_stage[1:].view(torch.uint8), v_stage[1:].view(torch.uint8),
                   k_cache.view(-1, width).view(torch.uint8), v_cache.view(-1, width).view(torch.uint8),
                   data["cache_loc"], reserved_skip_index=-1)
    q16 = q8.view(batch, length, hq, d).to(torch.float16)
    k1 = _k1()
    max_k = data["page_table"].shape[1]  # shape-derived bound (page size 1): no host sync, graph-safe
    ns = min(MAX_SPLITS, k1.num_splits_for(batch, hkv, max_k))
    out = k1.fwd(q16, k_cache, v_cache, data["page_table"], data["cache_seqlens"], max_k,
                 cfg["sm_scale"] * cfg["qk_scale_factor"], ns)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = R1.rms_norm_gate(out.reshape(tokens, hq * d), gate, None, head_dim=d, eps=R1.RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)


def kernel_fn(data):
    if _STATE["ready"] and route_ok(data["config"]):
        return _route(data)
    return R1.kernel_fn(data)


SELF_TEST_SHAPES = [
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1, seed=11),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges", page=1, seed=12),
    dict(batch=3, ctx=100, heads="tp8", layer="full", q_len=8, page=1, seed=13),
    dict(batch=5, ctx=5000, heads="tp8", layer="full", q_len=8, spread="mixed", page=1, seed=14),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", q_len=8, spread="mixed", dist="coherent", page=1,
         seed=15),
]


def self_test(shapes=None, verbose=True):
    """Forced route vs the problem's reference.check_implementation; sets the route live only if all pass."""
    sys.path.insert(0, str(HERE.parents[1] / "problems" / "muse_attention_decode"))
    import reference
    ok_all = True
    try:
        for kw in shapes or SELF_TEST_SHAPES:
            data = reference.generate_input(**kw)
            out = _route(data)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
            ok_all &= bool(ok) and bool(torch.isfinite(out).all())
            if verbose:
                print(f"[k1fa4] self-test {kw}: {'PASS' if ok else 'FAIL'} {str(msg)[:240]}", flush=True)
            del data
    except Exception as ex:  # noqa: BLE001
        print(f"[k1fa4] self-test ERROR {type(ex).__name__}: {str(ex)[:400]}", flush=True)
        ok_all = False
    _STATE["ready"] = ok_all
    return ok_all


def prepare(*_a, **_k):
    """Import + JIT-compile (one small forced-route call) + self-test; the route goes live only on PASS."""
    return self_test(SELF_TEST_SHAPES[:3], verbose=True)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--shapes", default="9,11")
    args = ap.parse_args()
    import time
    t0 = time.time()
    st = self_test()
    print(f"[k1fa4] self-test {'PASS' if st else 'FAIL'} in {time.time() - t0:.1f}s (incl. JIT compile)",
          flush=True)
    sys.path.insert(0, str(HERE.parents[1] / "problems" / "muse_attention_decode"))
    import reference
    benches = json.loads((HERE.parents[1] / "problems" / "muse_attention_decode" / "meta.json").read_text())[
        "benchmarks"]
    dev = torch.device("cuda")
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
        row = []
        for name, fn in (("r1", R1.kernel_fn), ("k1fa4", kernel_fn)):
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
            torch.cuda.empty_cache()
        print(f"BENCH [{i:2d}] {bm['load']} {bm['layer']} (full kernel_fn, graph replay, cold L2) | "
              + " | ".join(row), flush=True)

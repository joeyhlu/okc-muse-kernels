"""K1-FA4 v4 drop-in route for L=8 (verify) blocks: as k1_fa4v2_route, with k1_fa4v4.fwd (FP8 QK and FP8 PV with E4M3
P, stock FA3's numerics per A16) in place of k1_fa4v2.fwd. Same gating, prologue/epilogue, self-test and fallback.

  python3 kernels/muse_attention_decode/k1_fa4v4_route.py [--reps 30] [--shapes 6,8,10,9,11] [--with-v2]
"""
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r1_glue as R1  # noqa: E402

ROUTE_FULL_MIN_BATCH = 16     # as v2 (k1_fa4_design.md section 8)
ROUTE_SLIDING = True          # decided by the measurement in section 8
ROUTE_SLIDING_MIN_BATCH = 1   # wins at B 1 / 16 / 32 (section 8)
SPLITS = {("sliding", 16): 4}  # (layer, batch) -> num_splits override; else k1_fa4v2.num_splits_for
_STATE = {"ready": None, "k2": None}


def _k2():
    if _STATE["k2"] is None:
        import k1_fa4v4
        _STATE["k2"] = k1_fa4v4
    return _STATE["k2"]


def route_ok(cfg, force=False):
    if not (cfg["q_len"] == 8 and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]):
        return False
    if force:
        return True
    if cfg["window"]:
        return ROUTE_SLIDING and cfg["batch"] >= ROUTE_SLIDING_MIN_BATCH
    return cfg["batch"] >= ROUTE_FULL_MIN_BATCH


IWO = {"full": False, "sliding": False}  # FA4 intra-WG overlap per layer type (section 8)


def _route(data, splits=None, iwo=None):
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
    max_k = data["page_table"].shape[1]  # shape-derived bound (page size 1): no host sync, graph-safe
    keys = min(max_k, window + length) if window else max_k
    ns = splits or SPLITS.get(("sliding" if window else "full", batch)) or k2.num_splits_for(batch, hkv, keys)
    out = k2.fwd(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"], data["cache_seqlens"],
                 max_k, cfg["sm_scale"] * cfg["qk_scale_factor"], ns, window=window)
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = R1.rms_norm_gate(out.reshape(tokens, hq * d), gate, None, head_dim=d, eps=R1.RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)


def kernel_fn(data):
    if _STATE["ready"] is None and route_ok(data["config"]) and not torch.cuda.is_current_stream_capturing():
        prepare()  # standalone use without a prepare hook: one-time compile + self-test on the first eager call
    if _STATE["ready"] and route_ok(data["config"]):
        return _route(data)
    return R1.kernel_fn(data)


SELF_TEST_SHAPES = [
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1, seed=11),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges", page=1, seed=12),
    dict(batch=3, ctx=100, heads="tp8", layer="full", q_len=8, page=1, seed=13),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", q_len=8, spread="mixed", dist="coherent", page=1,
         seed=15),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1, seed=21),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1, seed=22),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1, seed=23),
    dict(batch=3, ctx=1000, heads="tp8", layer="sliding", q_len=8, page=1, seed=24),
    dict(batch=2, ctx=100000, heads="tp8", layer="sliding", q_len=8, page=1, seed=25),
]


def self_test(shapes=None, verbose=True):
    """Forced route vs the problem's reference.check_implementation; the route goes live only if all pass."""
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
                print(f"[k1fa4v4] self-test {kw}: {'PASS' if ok else 'FAIL'} {str(msg)[:200]}", flush=True)
            del data
    except Exception as ex:  # noqa: BLE001
        print(f"[k1fa4v4] self-test ERROR {type(ex).__name__}: {str(ex)[:600]}", flush=True)
        ok_all = False
    _STATE["ready"] = ok_all
    return ok_all


def prepare(*_a, **_k):
    """Import + JIT-compile (forced-route calls, full and sliding) + self-test; route live only on PASS."""
    return self_test([SELF_TEST_SHAPES[i] for i in (0, 2, 5, 7)], verbose=True)


if __name__ == "__main__":
    import argparse
    import json
    import time
    import traceback

    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--shapes", default="8,10,9,11")
    ap.add_argument("--sliding-splits", default="2,4,8,16")
    ap.add_argument("--with-v2", action="store_true", help="also time k1_fa4v2_route (v2) in the same process")
    ap.add_argument("--raw", type=int, default=0, help="raw V ring depth for the self-test and v4 (0 = default)")
    args = ap.parse_args()
    if args.raw:
        _k2().set_raw(args.raw)
    t0 = time.time()
    st = self_test()
    print(f"[k1fa4v4] self-test {'PASS' if st else 'FAIL'} in {time.time() - t0:.1f}s (incl. JIT compile)",
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
    v2 = None
    if args.with_v2:
        try:
            import k1_fa4v2_route as v2
        except Exception as ex:  # noqa: BLE001
            print(f"[k1fa4v4] v2 import failed: {ex}", flush=True)
            v2 = None
    for i in [int(x) for x in args.shapes.split(",")]:
        bm = benches[i]
        kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
        variants = [("r1", R1.kernel_fn)]
        if v2 is not None:
            variants.append(("v2", v2._route))
        variants.append(("v4", lambda d: _route(d)))
        variants.append(("v4raw3", "raw3"))
        if bm["layer"] == "full":
            variants.append(("v4st4", "st4"))
        else:
            variants.append(("v4st3", "st3"))
            for s in [int(x) for x in args.sliding_splits.split(",")]:
                variants.append((f"v4s{s}", lambda d, s=s: _route(d, splits=s)))
        row = []
        for name, fn in variants:
            raw_default = args.raw or 2
            _k2().set_raw(3 if fn == "raw3" else raw_default)
            if fn == "raw3":
                fn = (lambda d: _route(d))
            if fn in ("st3", "st4"):  # same route, other K/V pipeline depth for this layer type
                lt = "sliding" if bm["layer"] == "sliding" else "full"
                _k2().set_stages(int(fn[2]), lt)
                fn = (lambda d: _route(d))
            else:
                _k2().set_stages(3, "full")
                _k2().set_stages(2, "sliding")
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

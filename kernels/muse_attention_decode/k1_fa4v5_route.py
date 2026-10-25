"""K1-FA4 v5 route + lab: fused prologue -> FA4 v2 core -> fused combine/norm-gate (k1_fa4v5.run), chained with PDL,
against K1 v4 (decode), the v2 route (r1 prologue/epilogue around FA4), and r1, in one process.

Checks per shape:
- the attention problem's check_implementation;
- the stock-relative rule: rel L2 vs reference.ref_kernel <= 1.25 x r1's own + 5e-3;
- new cache bytes bit-exact vs r1 (and vs K1 v4 on decode shapes);
- an input whose first new token is stored in cache row 0. SGLang's FusedQKNorm store skips that row, which is why
  r1 stages; v5 writes it.

  python3 kernels/muse_attention_decode/k1_fa4v5_route.py [--shapes 0,2,4,6,8,10,7,9,11,1,3,5] [--reps 40]
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

# Routes (k1_fa4_design.md section 10): sliding L=1 and L=8 at every batch; full L=8 at every batch; full L=1 only at
# B <= FULL_L1_MAX_BATCH (K1 v4 stays faster at B16/32 full decode).
FULL_L1_MAX_BATCH = 1
_STATE = {"ready": None, "v5": None}


def _v5():
    if _STATE["v5"] is None:
        import k1_fa4v5
        _STATE["v5"] = k1_fa4v5
    return _STATE["v5"]


def route_ok(cfg):
    if not _v5().route_ok(cfg):
        return False
    if cfg["window"] or cfg["q_len"] == 8:
        return True
    return cfg["batch"] <= FULL_L1_MAX_BATCH


def prepare(*_a, **_k):
    """Compile + correctness gate (attention-problem check, stock-relative rule, bit-exact cache vs r1, a cache-row-0
    case); the route goes live only if every shape passes."""
    sys.path.insert(0, str(PROB))
    import reference
    ok_all = True
    try:
        for kw, row0 in [(SELF_TEST[0], False), (SELF_TEST[4], False), (SELF_TEST[7], False), (SELF_TEST[10], False),
                         (dict(SELF_TEST[3], seed=60), True)]:
            res = check(reference, kw, [("r1", R1.kernel_fn), ("v5", lambda d: _v5().run(d))], row0=row0)
            ok5, _msg, e5, kb, vb = res["v5"]
            ok_all &= (ok5 and e5 <= 1.25 * res["r1"][2] + 5e-3 and kb is not None
                       and torch.equal(kb, res["r1"][3]) and torch.equal(vb, res["r1"][4]))
    except Exception as ex:  # noqa: BLE001
        print(f"[k1fa4v5] prepare ERROR {type(ex).__name__}: {str(ex)[:300]}", flush=True)
        ok_all = False
    _STATE["ready"] = ok_all
    print(f"[k1fa4v5] prepare {'PASS: route live' if ok_all else 'FAIL: r1 fallback'}", flush=True)
    return ok_all


def kernel_fn(data):
    if _STATE["ready"] is None and route_ok(data["config"]) and not torch.cuda.is_current_stream_capturing():
        prepare()
    if _STATE["ready"] and route_ok(data["config"]):
        return _v5().run(data)
    return R1.kernel_fn(data)


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm())


def force_row0(data):
    """Make the first new token of request 0 live in cache row 0 by swapping it with whichever (request, position)
    owns row 0, contents included (page size 1)."""
    pt, kc, vc = data["page_table"], data["k_cache"], data["v_cache"]
    p_new = int(data["seq_lens"][0])
    row_new = int(pt[0, p_new])
    if row_new == 0:
        return data
    hit = (pt == 0).nonzero()
    kf, vf = kc.view(kc.shape[0], -1), vc.view(vc.shape[0], -1)
    loc = data["cache_loc"]
    if hit.numel():  # row 0 belongs to some (request, position): swap roles and contents
        b0, p0 = int(hit[0, 0]), int(hit[0, 1])
        pt[b0, p0] = row_new
        for f in (kf, vf):
            tmp = f[0].clone()
            f[0] = f[row_new]
            f[row_new] = tmp
        loc[loc == 0] = -7  # mark, then remap
    else:  # row 0 is unassigned: just move the token there
        for f in (kf, vf):
            f[0] = f[row_new]
    pt[0, p_new] = 0
    loc[loc == row_new] = 0
    loc[loc == -7] = row_new
    assert int(loc[0]) == 0
    return data


def check(reference, kw, fns, row0=False):
    """Returns {name: (ok, msg, err, k_bytes, v_bytes)} for each fn, plus r1's err."""
    def gen():
        d = reference.generate_input(**kw)
        return force_row0(d) if row0 else d
    d_ref = gen()
    ref_out = reference.ref_kernel(d_ref)[0]
    res = {}
    for name, fn in fns:
        d = gen()
        try:
            out = fn(d)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(d, reference.collect(d, out))
            hkv = d["config"]["num_kv_heads"]
            rows = d["cache_loc"].long()
            kb = d["k_cache"].view(-1, hkv * 64).view(torch.uint8)[rows].clone()
            vb = d["v_cache"].view(-1, hkv * 64).view(torch.uint8)[rows].clone()
            res[name] = (bool(ok), str(msg)[:120], _rel(out.view_as(ref_out), ref_out), kb, vb)
        except Exception as ex:  # noqa: BLE001
            res[name] = (False, f"{type(ex).__name__}: {str(ex)[:200]}", float("nan"), None, None)
            print("\n".join("TB| " + ln for ln in traceback.format_exc()[-2000:].splitlines()), flush=True)
        del d
    return res


SELF_TEST = [
    dict(batch=1, ctx=10, heads="tp8", layer="sliding", page=1, seed=41),
    dict(batch=3, ctx=2047, heads="tp8", layer="sliding", page=1, seed=42),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1, seed=43),
    dict(batch=9, ctx=5000, heads="tp8", layer="sliding", spread="mixed", page=1, seed=44),
    dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1, seed=45),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1, seed=46),
    dict(batch=3, ctx=1000, heads="tp8", layer="sliding", q_len=8, page=1, seed=47),
    dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1, seed=48),
    dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges", page=1, seed=49),
    dict(batch=4, ctx=80000, heads="tp8", layer="full", q_len=8, spread="mixed", dist="coherent", page=1, seed=50),
    dict(batch=4, ctx=3000, heads="tp8", layer="full", page=1, seed=51),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="0,2,4,6,8,10,7,9,11,1,3,5")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--no-v4", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(PROB))
    import reference
    dev = torch.device("cuda")
    v5 = _v5()
    V4 = None
    if not args.no_v4:
        t0 = time.time()
        spec = importlib.util.spec_from_file_location("k1v4_lab5", str(HERE / "k1_v4.py"))
        V4 = importlib.util.module_from_spec(spec)
        sys.modules["k1v4_lab5"] = V4
        spec.loader.exec_module(V4)
        print(f"[k1fa4v5] k1_v4 build + self-test {'PASS' if V4._self_test(dev) else 'FAIL'} "
              f"in {time.time() - t0:.1f}s", flush=True)
    import k1_fa4v2_route as V2R
    import k1_fa4v3_route as V3R

    # ---- correctness ------------------------------------------------------------------------------------------
    all_ok = True
    t0 = time.time()
    for kw in SELF_TEST + [dict(SELF_TEST[3], seed=60), dict(SELF_TEST[7], seed=61)]:
        row0 = kw.get("seed") in (60, 61)
        fns = [("r1", R1.kernel_fn), ("v5", lambda d: v5.run(d))]
        if V4 is not None and kw.get("q_len", 1) == 1:
            fns.append(("v4", V4.kernel_fn))
        res = check(reference, kw, fns, row0=row0)
        e_r1 = res["r1"][2]
        ok5, msg5, e5, kb5, vb5 = res["v5"]
        rule = e5 <= 1.25 * e_r1 + 5e-3
        exact_r1 = kb5 is not None and torch.equal(kb5, res["r1"][3]) and torch.equal(vb5, res["r1"][4])
        exact_v4 = "-"
        if "v4" in res and res["v4"][3] is not None and kb5 is not None:
            exact_v4 = str(torch.equal(kb5, res["v4"][3]) and torch.equal(vb5, res["v4"][4]))
        ok = ok5 and rule and exact_r1
        all_ok &= ok
        print(f"[k1fa4v5] check {'PASS' if ok else 'FAIL'} {kw}{' row0' if row0 else ''}: check={ok5} "
              f"err v5 {e5:.4e} r1 {e_r1:.4e} (rule {rule}) cache==r1 {exact_r1} cache==v4 {exact_v4} | {msg5}",
              flush=True)
    print(f"[k1fa4v5] correctness {'PASS' if all_ok else 'FAIL'} in {time.time() - t0:.1f}s", flush=True)
    _STATE["ready"] = all_ok

    # ---- timing -----------------------------------------------------------------------------------------------
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

    def v5_variant(pdl, trig, splits=None, st_sl=2):
        def fn(d):
            v5.set_pdl(pdl, trig)
            v5.set_stages(st_sl, "sliding")
            return v5.run(d, splits=splits, pdl=pdl)
        return fn

    for i in [int(x) for x in args.shapes.split(",")]:
        bm = benches[i]
        kw = {k: v for k, v in bm.items() if k not in ("load", "weight")}
        L = bm.get("q_len", 1)
        variants = [("r1", R1.kernel_fn)]
        if L == 1:
            if V4 is not None:
                variants.append(("v4", V4.kernel_fn))
            variants.append(("fa4+r1io", lambda d: V3R.route(d)))
        else:
            variants.append(("v2route", lambda d: V2R._route(d)))
        variants += [("v5", v5_variant(True, "epi")), ("v5_notrig", v5_variant(True, False)),
                     ("v5_nopdl", v5_variant(False, False))]
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
                row.append(f"{name} {t:7.2f}us{'' if ok else ' FAIL(' + str(msg)[:60] + ')'}")
                del g, data
            except Exception as ex:  # noqa: BLE001
                row.append(f"{name} ERR {type(ex).__name__}: {str(ex)[:160]}")
                print("\n".join("TB| " + ln for ln in traceback.format_exc()[-2000:].splitlines()), flush=True)
            torch.cuda.empty_cache()
        v5.set_pdl(True, "epi")
        v5.set_stages(2, "sliding")
        print(f"BENCH [{i:2d}] {bm['load']} {bm['layer']} (full kernel_fn, graph replay, cold L2) | "
              + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()

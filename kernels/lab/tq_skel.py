"""tq_skel.py (moe-tq step 1, R-tq-1 in notes/memory/gpu_requests.md): ticket-queue SKELETON falsifier. LAB, box only.

Builds from kernels/lab/k3_tq_s.py (tools/tq_patches.py skel; base k3_gc_21): static null bf2n / w72n and TQ null bf2q / w72q =
the shipped VML producer, rings, real TMA weight + x streams, gather / self-fill, epilogues (zero accumulators), K-split fix-ups
and doneA releases as shipped; the consumers do no math (wait + release every slot, optional spin per stage).
Routings: A2.layer_routing (sliding layer case, input seed case + 1000) at T = 32 / 128 / 256 x router seeds.
Arms per routing (spin = emulated consumer cycles per stage from the GATED tier paces TRUTH; plan = the static planner's table):
  S   static, plan = shipped VML_COST, spin TRUTH      (control: the shipped static plan)
  Si  static, plan = TRUTH, spin TRUTH                (informed static: the cost-table share of any gain)
  Q8 / QM8  TQ A-then-B / merged (B block k after A ticket 32 (k + 1) + G), claim at consumer stage ~8 (main arms)
  Q0 / QM0  claim at unit start (claim-ahead 1); Q15 late claim (control); Q8r = Q8 with synchronous claims (RTT histogram)
  Sn / Sf / Q8n / QM8n: pure null (no spin): static shipped table / static flat table / TQ A-then-B / TQ merged
Timing: CUDA graph of vm_main + vm_fin per (routing, arm), judge GpuTimer (L2 flush + 300 us gate), rounds x reps interleaved.
Marks (VM_MKW 64): graph (marks.zero_(), call), --mreps gated replays: max / p50 CTA end, A end, B-gate waits, TQ claim stats.
Checks: err 0, stage counts (A 512 nj, B 256 nj), TQ unit counts (32 nj / 16 nj), scratch counters clean after the runs.
PASS (brief + coordinator R-tq-0 note): for a main TQ arm (Q8 or QM8, spin mode), median over routings of gated (TQ - S) <= -5 us
  at T = 128 and T = 256 and idle-on-claim p50 < 0.3 us (> half of the claims below 300 ns) there; T = 32 reported only (ship keeps
  the static plan there). Also reported: per-CTA end spread (max - p50, sd) and same-routing replay correlation of per-CTA ends,
  S vs Q8 / QM8; claim RTT p50 / p95 inside the stream (Q8r); B-gate wait per CTA.

  python3 -u kernels/lab/tq_skel.py [--rseeds=8677534258059892491,12345,777] [--rounds=3] [--reps=20] [--mreps=5] [--out=box_out/tq1/run.json]
  python3 kernels/lab/tq_skel.py summary box_out/tq1/run.json
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "tools/box", ROOT / "tools/judge"):
    sys.path.insert(0, str(_p))
LAB = ROOT / "kernels/lab/k3_tq_s.py"
MKW = 64
CYC_PER_UNIT = 1.11 / 128 * 1980          # gated tier-L pace 1.11 us per 128 cost units, SM clock 1980 MHz
# gated relative paces (tools/vct_patches.py REFIT = e4 fit of box_out/lat32/m_ship; T = 32 from the R-lat32-1 tier marks
# L 1.11 / M 1.33-1.42 / H 1.38-1.41, B rows assumed at the A ratios: no gated T = 32 B fit exists)
TRUTH = {256: [128, 153, 153, 164, 170, 179, 188, 201, 128, 145, 159, 168, 174, 184, 185, 200],
         128: [128, 151, 154, 166, 172, 184, 193, 200, 128, 144, 161, 168, 174, 185, 188, 201],
         32: [128, 159, 161, 161, 171, 182, 193, 204, 128, 156, 160, 160, 171, 182, 193, 204]}
FLAT = [128] * 16
# arm -> (build kind, plan table, tq lag ('G' = grid size, -1 = A-then-B, None = static), spin, tqat = producer issue stage of the
# claim; the producer runs NS = 3 stages ahead of the consumers, so tqat 11 = consumer stage ~8 (R-tq-0 best), 0 = claim-ahead 1
# (claim at unit start), 15 = late control; | 256 = RTT mode: every claim synchronous, round-trip histogram in the marks)
ARMS = {"S": ("n", "ship", None, True, 11), "Si": ("n", "truth", None, True, 11),
        "Q8": ("q", "ship", -1, True, 11), "QM8": ("q", "ship", "G", True, 11), "Q0": ("q", "ship", -1, True, 0),
        "QM0": ("q", "ship", "G", True, 0), "Q15": ("q", "ship", -1, True, 15), "Q8r": ("q", "ship", -1, True, 11 | 256),
        "Sn": ("n", "ship", None, False, 11), "Sf": ("n", "flat", None, False, 11), "Q8n": ("q", "ship", -1, False, 11),
        "QM8n": ("q", "ship", "G", False, 11)}
MAIN = ("Q8", "QM8")
RBIN = (0.25, 0.75, 1.25, 1.75, 2.5, 3.75, 5.25, 7.0)   # RTT histogram bin centres (us): < 0.5 / 1 / 1.5 / 2 / 3 / 4.5 / 6 / more


def load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(argv):
    import torch
    import judge as J
    import a2_lab as A2
    rounds, reps, mreps = int(A2.arg(argv, "rounds", 3)), int(A2.arg(argv, "reps", 20)), int(A2.arg(argv, "mreps", 5))
    rseeds = [int(s) for s in str(A2.arg(argv, "rseeds", "8677534258059892491,12345,777")).split(",")]
    out = Path(str(A2.arg(argv, "out", "box_out/tq1/run.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    G = load(LAB, "k3_tqskel_lab")
    mods, errs, bt = {}, {}, {}

    def bld(v):
        t0 = time.time()
        try:
            mods[v], _ = G._vml_build(defines=G._VML_DEFS[v])
        except Exception as ex:  # noqa: BLE001
            errs[v] = repr(ex)[-800:]
        bt[v] = round(time.time() - t0)
    ths = [threading.Thread(target=bld, args=(v,)) for v in ("bf2n", "bf2q", "w72n", "w72q")]
    for t in ths:
        t.start()
    prob = J.load_problem(ROOT / "problems/muse_decode_layer")
    ref = prob.ref
    cases = {}
    for b in prob.benches:
        cases.setdefault((b["load"], b["layer"]), b)
    inputs = {}
    t0 = time.time()
    for rs in rseeds:
        w0 = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda", router_seed=rs)
        ctx = {"torch": torch, "ref": ref, "weights": w0}
        for T in (32, 128, 256):
            case = cases[(A2.LAYER_CASE[T], "sliding")]
            inputs[f"L{T}r{rs % 1000}"] = A2.layer_routing(ctx, case, case["seed"] + 1000)
        del w0
        torch.cuda.empty_cache()
    log(f"[tqskel] routings {len(inputs)} in {time.time() - t0:.0f} s")
    E, SW3 = G.K2A_E_LOC, G.K2A33_SW3
    kk = {"ws13": torch.empty((E, 64, 16, SW3), dtype=torch.uint8, device="cuda"),
          "ws2": torch.empty((E, 32, 16, SW3), dtype=torch.uint8, device="cuda"),
          "gsc13": torch.ones(E, device="cuda"), "gsc2": torch.ones(E, device="cuda"), "nw": torch.ones(4096, device="cuda")}
    for t in ths:
        t.join()
    log(f"[tqskel] builds {bt} errs {errs}")
    if errs:
        raise SystemExit("build failed")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    sc = G._vml_alloc("cuda", sms)
    timer = J.GpuTimer()
    marks = torch.zeros(((sms + 1) * MKW,), dtype=torch.int64, device="cuda")

    def call(T, arm, d, mk=None):
        kind, plan, lag, spin, at = ARMS[arm]
        mod = mods[("bf2" if T == 32 else "w72") + kind]
        rep = G._k2a_bucket_of(d["hidden_states"].shape[0])
        assert rep == T, (rep, T)
        tab = {"ship": G.VML_COST[T], "truth": TRUTH[T], "flat": FLAT}[plan]
        cyc = [int(round(c * CYC_PER_UNIT)) for c in G._vml_cost_ext(mod, TRUTH[T])] if spin else None
        lg = -1 if lag is None else (sms if lag == "G" else lag)
        return G._vml_call(mod, kk, sc, d["hidden_states"], d["topk_ids"], d["topk_weights"], 0, marks=mk,
                           flags=G.VML_FLAGS[T], cost=tab, tq=(lg, at, cyc))

    def njobs(T, d):
        ids = d["topk_ids"].long().flatten()
        ids = ids[(ids >= 0) & (ids < 32)]
        jn = 64 if T == 32 else 72
        return sum((c + jn - 1) // jn for c in torch.bincount(ids, minlength=32).tolist())

    def mstats(arm, nj):
        m = marks.view(sms + 1, MKW)[:sms].cpu().tolist()
        st = [r for r in m if r[0] > 0]
        t0_ = min(r[0] for r in st)
        cend = [round((max(r[1:5]) - t0_) / 1e3, 3) if max(r[1:5]) > 0 else None for r in m]
        ends = sorted(x for x in cend if x is not None)
        sA = sum(sum(r[8:24]) for r in m)
        sB = sum(sum(r[24:40]) for r in m)
        res = {"kend": ends[-1], "p50": ends[len(ends) // 2], "aend": (max(r[6] for r in st) - t0_) / 1e3, "sd": statistics.pstdev(ends),
               "bw_us": sum(r[40] for r in m) / 1e3 / sms, "bw_n": sum(r[41] for r in m), "bw_max": max(r[40] for r in m) / 1e3,
               "cend": cend, "ok_st": sA == 512 * nj and sB == 256 * nj}
        kind, at = ARMS[arm][0], ARMS[arm][4]
        if kind == "q":
            res["ok_u"] = sum(r[49] for r in m) == 32 * nj and sum(r[50] for r in m) == 16 * nj
            res["rtt1"] = statistics.median([(r[62] & 0xFFFFFFFF) / 1e3 for r in m])
            if at & 256:
                res.update(rtth=[sum((r[63] >> (8 * b)) & 0xFF for r in m) for b in range(8)], rttmax=max(r[62] >> 32 for r in m) / 1e3)
            else:
                res.update(nc=sum(r[51] for r in m), idle_us=sum(r[55] for r in m) / 1e3, n3=sum(r[63] >> 32 for r in m),
                           n1=sum(r[63] & 0xFFFFFFFF for r in m), idmax=max(r[62] >> 32 for r in m) / 1e3)
        return res

    bad, res = [], {"errs": errs, "builds": bt, "arms": {a: list(v) for a, v in ARMS.items()}, "nj": {}, "mk": {}}
    graphs, mgraphs = {}, {}
    for k, d in inputs.items():
        T = int(k[1:].split("r")[0])
        nj = njobs(T, d)
        res["nj"][k] = nj
        for arm in ARMS:
            sc["err"].zero_()
            marks.zero_()
            call(T, arm, d, marks)
            torch.cuda.synchronize()
            e = int(sc["err"][0])
            ms = mstats(arm, nj)
            if e != 0 or not ms["ok_st"] or not ms.get("ok_u", True):
                bad.append(f"{k}|{arm} err {e} st {ms['ok_st']} u {ms.get('ok_u')}")
                continue
            with torch.cuda.stream(timer.stream):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=timer.stream):
                    call(T, arm, d)
                gm = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gm, stream=timer.stream):
                    marks.zero_()
                    call(T, arm, d, marks)
            torch.cuda.synchronize()
            graphs[(k, arm)], mgraphs[(k, arm)] = g, gm
    dirty = {n: float(sc[n].abs().sum()) for n in ("xready", "doneA", "cnt", "gbar")}
    if any(dirty.values()):
        bad.append(f"dirty {dirty}")
    log(f"[tqskel] checks (err 0, stage / unit counts, clean counters) on {len(inputs)} x {len(ARMS)}: FAIL {bad if bad else 'none'}")
    tm = {}
    t0 = time.time()
    arms = list(ARMS)
    for r in range(rounds):
        for k in inputs:
            order = arms[r % len(arms):] + arms[:r % len(arms)]
            for arm in order:
                if (k, arm) in graphs:
                    tm.setdefault(f"{k}|{arm}", []).extend(timer.time_replay(graphs[(k, arm)]) for _ in range(reps))
        log(f"[tqskel] gated round {r} done at {time.time() - t0:.0f} s")
    for k in inputs:
        for arm in arms:
            if (k, arm) not in mgraphs:
                continue
            rows = []
            for _ in range(mreps):
                timer.time_replay(mgraphs[(k, arm)])
                torch.cuda.synchronize()
                rows.append(mstats(arm, res["nj"][k]))
            res["mk"][f"{k}|{arm}"] = rows
    if int(sc["err"][0]) != 0:
        bad.append(f"err after timing {int(sc['err'][0])}")
    res["gmed"] = {key: statistics.median(v) for key, v in tm.items()}
    res["bad"] = bad
    out.write_text(json.dumps(res, default=str))
    summary(res)


def _corr(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5 if va > 0 and vb > 0 else 0.0


def _rep_corr(rows):
    """mean pairwise correlation of per-CTA ends across replays of one routing (same-routing lateness persistence)."""
    v = [r["cend"] for r in rows]
    idx = [i for i in range(len(v[0])) if all(x[i] is not None for x in v)]
    cs = [_corr([v[a][i] for i in idx], [v[b][i] for i in idx]) for a in range(len(v)) for b in range(a + 1, len(v))]
    return statistics.mean(cs) if cs else 0.0


def _pct(h, q):
    n, acc = sum(h), 0
    for b, c in enumerate(h):
        acc += c
        if acc >= q * n:
            return RBIN[b]
    return RBIN[-1]


def summary(res):
    g, mk = res["gmed"], res["mk"]
    med = statistics.median
    ks = sorted({x.split("|")[0] for x in g}, key=lambda s: (len(s), s))
    print("[tqskel] T | arm vs ref | gated d us median (min..max) | marks kend d | p50 d | ref gated / kend", flush=True)
    dmed, idle = {}, {}
    for T in (32, 128, 256):
        kt = [k for k in ks if k.startswith(f"L{T}r") and f"{k}|S" in g]
        if not kt:
            continue
        for arm in ("Si", "Q8", "QM8", "Q0", "QM0", "Q15", "Sf", "Q8n", "QM8n"):
            ref = "Sn" if arm in ("Sf", "Q8n", "QM8n") else "S"
            kk = [k for k in kt if f"{k}|{arm}" in g and f"{k}|{ref}" in g and mk.get(f"{k}|{arm}") and mk.get(f"{k}|{ref}")]
            if not kk:
                continue
            dg = [g[f"{k}|{arm}"] - g[f"{k}|{ref}"] for k in kk]
            dk = [med([r["kend"] for r in mk[f"{k}|{arm}"]]) - med([r["kend"] for r in mk[f"{k}|{ref}"]]) for k in kk]
            dp = [med([r["p50"] for r in mk[f"{k}|{arm}"]]) - med([r["p50"] for r in mk[f"{k}|{ref}"]]) for k in kk]
            dmed[(T, arm)] = med(dg)
            if arm != "QM8n":
                print(f"[tqskel] {T:3d} | {arm:4s} vs {ref:2s} | {med(dg):+6.2f} ({min(dg):+.1f}..{max(dg):+.1f}) | {med(dk):+6.2f} | {med(dp):+6.2f}"
                      f" | {med([g[f'{k}|{ref}'] for k in kk]):6.1f} / {med([med([r['kend'] for r in mk[f'{k}|{ref}']]) for k in kk]):6.1f}", flush=True)
        for arm in ("S", "Q8", "QM8"):   # per-CTA end spread + same-routing persistence (per routing, then median)
            kk = [k for k in kt if mk.get(f"{k}|{arm}")]
            if kk:
                sp = [med([r["kend"] - r["p50"] for r in mk[f"{k}|{arm}"]]) for k in kk]
                sd = [med([r["sd"] for r in mk[f"{k}|{arm}"]]) for k in kk]
                rc = [_rep_corr(mk[f"{k}|{arm}"]) for k in kk]
                print(f"[tqskel] {T:3d} | {arm:4s} CTA end max-p50 {med(sp):5.2f} us (routings {' '.join(f'{x:.1f}' for x in sp)}) sd {med(sd):.2f}"
                      f" rep-corr {med(rc):.2f}", flush=True)
        for arm in MAIN + ("Q0",):
            rows = [r for k in kt for r in mk.get(f"{k}|{arm}", [])]
            if rows:
                nc = sum(r["nc"] for r in rows)
                f3 = sum(r["n3"] for r in rows) / max(nc, 1)
                idle[(T, arm)] = f3
                print(f"[tqskel] {T:3d} | {arm:4s} claims {nc} idle<0.3us {f3:.2f} <0.1us {sum(r['n1'] for r in rows) / max(nc, 1):.2f} mean "
                      f"{sum(r['idle_us'] for r in rows) / max(nc, 1):.3f} max {max(r['idmax'] for r in rows):.2f} | B-gate us/CTA med "
                      f"{med([r['bw_us'] for r in rows]):.2f} max {max(r['bw_max'] for r in rows):.1f} (S "
                      f"{med([r['bw_us'] for k in kt for r in mk.get(f'{k}|S', [])]):.2f})", flush=True)
        rows = [r for k in kt for r in mk.get(f"{k}|Q8r", [])]
        if rows:
            h = [sum(r["rtth"][b] for r in rows) for b in range(8)]
            print(f"[tqskel] {T:3d} | claim RTT (sync, in stream) p50 ~{_pct(h, 0.5):.2f} p95 ~{_pct(h, 0.95):.2f} max "
                  f"{max(r['rttmax'] for r in rows):.2f} us, hist {h}", flush=True)
    ok = {a: all(dmed.get((T, a), 99) <= -5 and idle.get((T, a), 0) > 0.5 for T in (128, 256)) for a in MAIN}
    best = [a for a, o in ok.items() if o]
    print(f"[tqskel] check failures: {res.get('bad')}", flush=True)
    print(f"[tqskel] STEP1 {'PASS ' + ','.join(best) if best and not res.get('bad') else 'KILL'} (rule at T 128 / 256; T 32 report only) "
          f"{ {a: [round(dmed[(T, a)], 2) for T in (32, 128, 256) if (T, a) in dmed] for a in MAIN} }", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(json.loads(Path(sys.argv[2]).read_text()))
    else:
        main(sys.argv[1:])

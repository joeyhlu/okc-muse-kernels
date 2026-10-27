"""MK0 on paper: byte schedule + timing model of one sliding decoder layer under three structures.

No GPU (user order 2026-10-27: no pods). Inputs are the problem's sizes (problems/muse_decode_layer/reference.py) and
H200 constants we measured (notes/lessons.md, notes/lab.md, notes/study/k2a_tails.md) or that were published
(notes/study/top_teams_scan.md: Hazy megakernel, tie-pilot-qxw PDL reconstruction, MPK). Every constant names its source.

  (a)  one launch per op, no PDL       = sum over ops of the measured one-launch law
  (a2) ideal PDL chain                 = (b) + boundaries x PDL residual (tie-pilot: chain = persistent + 0.33 us/edge)
  (b)  ideal persistent kernel         = ramp + bytes / stream rate + tail + exposed dependency latency
  (c)  pure stream of the same bytes   = ramp + bytes / stream rate + tail

Run: python kernels/lab/mk0/mk0_model.py   (prints the tables used in notes/study/mk0_paper.md)
"""

# ---- problem sizes (reference.py) ----
HIDDEN, D_EXPERT, QKV_SIZE, Q_SIZE, KV_ROW = 8192, 4096, 2304, 1024, 128  # KV_ROW: both KV heads of one key, FP8
EXPERT_BYTES = 3 * D_EXPERT * D_EXPERT * (0.5 + 1 / 16)  # W13 + W2, NVFP4 + FP8 scale per 16 = 28.31 MB
W13_BYTES = EXPERT_BYTES * 2 / 3
DENSE = {"qkv": QKV_SIZE * HIDDEN * 2, "o": HIDDEN * Q_SIZE * 2, "router": 256 * HIDDEN * 4,
         "pooler": D_EXPERT * HIDDEN * 2, "unpool": HIDDEN * D_EXPERT * 2}
WINDOW = 2048

# ---- measured / published constants (us, TB/s = 1e12 B/s) ----
FLOOR = 4.8            # judge floor rate
R_STREAM = 4.45        # 1 GiB LDG / bulk-ring stream 4.29-4.48 (lessons s6, lab M3)
R_LAUNCH = 4.74        # one launch over a matrix: t = 3.05 + B / 4.74 (lessons s6, cold L2, graph replay)
T0_LAUNCH = 3.05       # fixed cost of that law (launch + ramp + tail)
EXTRA_LAUNCH = 0.97    # each extra launch inside a graph (lessons s6)
R_GATHER = 3.6         # random 128 B rows, plain loads, deep (lessons s6)
PDL_HOP = (0.33, 0.50)  # residual per PDL edge: tie-pilot H100 0.33 us; ours H200 0.5 (lab 10-26 k1-sl)
FLAG_HOP = 1.0         # release -> poll -> acquire -> first activation load: ~2 global load/store pairs, ~0.5 us each
                       # (Hazy B200); our K1->K2 per-expert hand-off measured 0.5 (k2a_tails) -> 1.0 is conservative
LIGHT = 1.0            # row-owned norm / top-k / merge / combine body after its inputs land
RAMP, TAIL = 1.0, 1.0  # first-byte latency of a cold stream; last-wave spread of a balanced stream
RING_KB = 192          # smem lookahead per SM (one CTA/SM, 12 x 16 KiB)
L2_PREFETCH_MB = 30.0  # usable L2 lookahead (50 MB L2 minus working set); PF1 says it buys ~0 in practice
SMS = 132
TC_EFF = 0.6 * 989e12  # achievable BF16 tensor rate for the T=256 dense GEMMs (assumption, mma.sync/wgmma mix)

LOADS = {  # name: (tokens, batch, q_len, distinct local experts (brief), local routes (notes/layer_budget.md))
    "decode1": (1, 1, 1, 2, 2.5),
    "decode32": (32, 32, 1, 9, 61.0),
    "verify32": (256, 32, 8, 23, 400.0),
}
# v34 judge medians (kernels/results/2cc140effaa54e71.json): (sliding, full) us per layer, kernels per sliding layer
V34 = {"decode1": (88.01, 100.44, 10), "decode16": (144.82, 238.33, 11), "decode32": (180.61, 360.84, 11),
       "verify1": (125.53, 140.18, 9), "verify16": (308.41, 421.17, 13), "verify32": (399.18, 634.70, 14)}


def us(nbytes, rate_tbs):
    return nbytes / (rate_tbs * 1e6)


def schedule(tokens, batch, q_len, n_exp):
    """Ops in dependency order: (name, bytes, kind). kind: w = weight stream, g = gather, l = light (no weights)."""
    unpool = DENSE["unpool"] // 8 if tokens <= 8 else DENSE["unpool"]
    kv = batch * (WINDOW + q_len - 1) * KV_ROW * 2  # K and V rows, both heads per 128 B row
    return [("qkv", DENSE["qkv"], "w"), ("attn", kv, "g"), ("merge", 0, "l"), ("o", DENSE["o"], "w"),
            ("norm1", 0, "l"), ("router", DENSE["router"], "w"), ("topk", 0, "l"), ("pooler", DENSE["pooler"], "w"),
            ("w13", n_exp * W13_BYTES, "w"), ("w2", n_exp * (EXPERT_BYTES - W13_BYTES), "w"),
            ("combine", 0, "l"), ("unpool", unpool, "w"), ("norm2", 0, "l")]


def model(name):
    tokens, batch, q_len, n_exp, routes = LOADS[name]
    ops = schedule(tokens, batch, q_len, n_exp)
    total = sum(b for _, b, _ in ops)
    floor = us(total, FLOOR)
    c = RAMP + us(total, R_STREAM) + TAIL
    a = sum((T0_LAUNCH + us(b, R_LAUNCH)) if k == "w" else (T0_LAUNCH + us(b, R_GATHER)) if k == "g"
            else (EXTRA_LAUNCH + LIGHT) for _, b, k in ops)
    # (b): HBM streams continuously except where a consumer-side latency chain outlasts the producer's lookahead.
    ring_us = us(RING_KB * 1024 * SMS, R_STREAM)
    # Middle chain after the last qkv byte: attention body (KV already prefetched: it does not depend on q),
    # 4 flag hops (qkv->attn->merge->o->norm1->router/pooler) and 2 light bodies (merge, norm1).
    mid = 1.5 + 4 * FLAG_HOP + 2 * LIGHT
    exposed_mid = max(0.0, mid - ring_us)                                   # smem ring only
    exposed_mid_l2 = max(0.0, mid - ring_us - us(L2_PREFETCH_MB * 1e6, R_STREAM))
    # Router -> top-k -> expert addresses: hidden while the pooler streams (router placed before pooler).
    pooler_us = us(DENSE["pooler"], R_STREAM)
    exposed_route = max(0.0, (2 * FLAG_HOP + LIGHT) - pooler_us)
    # End chain after the last weight byte. T <= 8: last bytes = the 8 MB unpooler slice, streamed during the
    # W2 -> combine -> unpool -> norm2 chain (3 hops + 3 bodies). T > 8: unpool weights cover W2 -> combine; only
    # unpool drain + 1 hop + norm2 remain.
    if tokens <= 8:
        end = max(0.0, 3 * FLAG_HOP + 2 * LIGHT + 0.3 - us(DENSE["unpool"] / 8, R_STREAM))
    else:
        end = 0.5 + FLAG_HOP + LIGHT
    b = RAMP + us(total, R_STREAM) + exposed_mid + exposed_route + end
    # T = 256: the dense GEMMs are tensor-bound; only the pooler -> expert edge can bank lookahead into a
    # memory-bound phase (ring + L2), the rest of the excess compute stays exposed in every structure.
    excess = 0.0
    if tokens >= 128:
        for n_, b_, k_ in ops:
            if n_ in ("qkv", "o", "router", "pooler", "unpool"):
                params = b_ / (4 if n_ == "router" else 2) * (2 if n_ == "router" else 1)  # router: BF16 hi + lo
                excess += max(0.0, 2 * tokens * params / TC_EFF * 1e6 - us(b_, R_STREAM))
    excess_banked = max(0.0, excess - min(excess, ring_us + us(L2_PREFETCH_MB * 1e6, R_STREAM)))
    a += excess  # one launch per op cannot bank lookahead across the tensor-bound GEMMs
    excess = excess_banked
    boundaries = sum(1 for _ in ops) - 1
    return dict(name=name, ops=ops, total=total, floor=floor, c=c, a=a, b=b, b_l2=b - exposed_mid + exposed_mid_l2,
                b_tc=b + excess, a2=(b + boundaries * PDL_HOP[0], b + boundaries * PDL_HOP[1]),
                mid=mid, ring_us=ring_us, end=end, boundaries=boundaries)


def score_gain(delta_us_per_layer):
    """Overall board gain (geomean over the 6 loads, each load = 46 sliding + 16 full layers) from saving
    delta_us_per_layer (dict load -> us, applied to sliding and full layers alike), relative to v34."""
    import math
    logs = []
    for load, (sl, fu, _) in V34.items():
        t = 46 * sl + 16 * fu
        d = 62 * delta_us_per_layer.get(load, 0.0)
        logs.append(math.log(t / (t - d)))
    return math.exp(sum(logs) / len(logs)) - 1


def main():
    print("== MK0 paper model: one sliding layer (us). floor = bytes @ 4.8 TB/s ==")
    print(f"{'load':9} {'MB':>6} {'floor':>6} {'(c)':>6} {'(b)':>6} {'(b)L2':>6} {'(a2)':>11} {'(a)':>6}"
          f" {'b/c':>5} {'b/flr':>5} {'a/b':>5} {'v34':>6}")
    rows = {}
    for name in LOADS:
        r = model(name)
        rows[name] = r
        v = V34[name][0]
        b_show = r["b_tc"]
        print(f"{name:9} {r['total'] / 1e6:6.1f} {r['floor']:6.1f} {r['c']:6.1f} {b_show:6.1f} "
              f"{r['b_l2'] + (r['b_tc'] - r['b']):6.1f} {r['a2'][0] + (r['b_tc'] - r['b']):5.1f}-"
              f"{r['a2'][1] + (r['b_tc'] - r['b']):5.1f} {r['a']:6.1f} {b_show / r['c']:5.2f}"
              f" {b_show / r['floor']:5.2f} {r['a'] / b_show:5.2f} {v:6.1f}")
    print("v34 minus ideal PDL chain (= per-kernel body quality, not structure): "
          + ", ".join(f"{n} {V34[n][0] - r['a2'][1] - (r['b_tc'] - r['b']):.0f} us" for n, r in rows.items()))
    print("(b) at verify32 includes tensor-bound dense excess beyond ring+L2 lookahead (b without it:"
          f" {rows['verify32']['b']:.1f}).")
    print(f"ring lookahead {rows['decode1']['ring_us']:.1f} us; middle chain {rows['decode1']['mid']:.1f} us;"
          f" boundaries per layer {rows['decode1']['boundaries']}")
    print("\n== byte schedule (MB) ==")
    for name, r in rows.items():
        print(name, " ".join(f"{n}:{b / 1e6:.1f}" for n, b, _ in r["ops"] if b))
    print("\n== value of persistence ACROSS op boundaries over an ideal PDL chain ==")
    for hop in PDL_HOP:
        d = {load: kernels * hop for load, (_, _, kernels) in V34.items()}
        print(f"  PDL residual {hop:.2f} us x v34 kernels/layer (9-14): {100 * score_gain(d):.2f} % overall")
    d = {load: kernels * PDL_HOP[1] + 2.0 for load, (_, _, kernels) in V34.items()}
    print(f"  + 2 us/layer of finer-grained flags everywhere (upper bound): {100 * score_gain(d):.2f} % overall")
    print("\n== MoE-internal levers (T=32 gap 39.2 us, k2a_tails s1) ==")
    # Removable by homogeneous cost-balanced CTAs (no hot/cold roles, no tickets, W2 gated per expert):
    # K1 tail 15.4 -> ~3, K2 tail 10.5 -> ~3, start excess 3.5 -> ~1, hand-off 0.5 -> 0, K3 5.4 -> ~2.5.
    lo, hi = 15.4 + 10.5 - 2 * 4.0 + 2.0 + 0.5 + 2.0, 15.4 + 10.5 - 2 * 2.0 + 3.0 + 0.5 + 3.4
    print(f"  T=32 removable {lo:.0f}-{hi:.0f} us of 39.2 -> MoE ~{95.5 - hi:.0f}-{95.5 - lo:.0f} us"
          f" (consumer-bound floor 255 MB / 3.8 TB/s = {us(9 * EXPERT_BYTES, 3.8):.0f} us + ~4)")
    removable = {"verify1": (8, 13), "decode16": (12, 17), "decode32": (lo, hi), "verify16": (30, 46),
                 "verify32": (25, 40), "decode1": (5, 8)}
    for i, tag in ((0, "low"), (1, "high")):
        print(f"  all loads ({tag}): {100 * score_gain({k: v[i] for k, v in removable.items()}):.1f} % overall")


if __name__ == "__main__":
    main()

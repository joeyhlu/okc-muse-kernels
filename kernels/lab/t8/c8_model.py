"""C8 conveyor paper model (notes/study/t8_design.md s3). No GPU. Run: python -I kernels/lab/t8/c8_model.py

Per-SM model of one persistent conveyor kernel (homogeneous SMs, 132 CTAs). The producer receives weight bytes at the
per-SM stream rate R (4.4 TB/s / 132, stream5 / r0b measured) from t0 on, as long as the SMEM ring has room. The consumer
drains a phase's bytes only after that phase's dependency time, at k * R (bytes already staged). Dependencies are the
completion times of earlier phases plus L2 hop latencies (counter or flag round trips). Optimistic in one way (no
granule imbalance, no start skew) and pessimistic in another (hops are serial, no overlap of a hop with compute).
"""
R = 4.4e12 / 132 / 1e9   # KB per us per SM (33.3)
KB = 1e3


def run(phases, ring_kb, t0=1.0, k=2.5, dt=0.002, hop=1.0, tail=0.0):
    n = len(phases)
    left_arr = [p["kb"] for p in phases]
    left_con = [p["kb"] for p in phases]
    done = [None] * n
    t, occ, stall, peak, ai, ci = 0.0, 0.0, 0.0, 0.0, 0, 0
    while ci < n and t < 400:
        if t >= t0 and ai < n:
            room = ring_kb - occ
            a = min(R * dt, room, left_arr[ai])
            if room < R * dt:
                stall += dt
            left_arr[ai] -= a
            occ += a
            if left_arr[ai] <= 1e-9:
                ai += 1
        dep = phases[ci]["dep"](done, hop)
        if dep is not None and t >= dep:
            arrived = phases[ci]["kb"] - left_arr[ci] if ci >= ai else phases[ci]["kb"]
            avail = arrived - (phases[ci]["kb"] - left_con[ci])
            c = min(k * R * dt, avail, left_con[ci])
            left_con[ci] -= c
            occ -= c
            if left_con[ci] <= 1e-9:
                done[ci] = t
                ci += 1
        peak = max(peak, occ)
        t += dt
    return done[-1] + tail, stall, peak, done


def front(T, ring_kb, hop=1.0, attn=None, k=2.5, qkv_first=True):
    """C8-F: qkv split-K 8 (q/k/v tiles first), 8 attention SMs (KS1-like chain: 3 hops + compute), o, boundary 1
    (moments hop; h all-gather hop for the pooler), router K-split, pooler split-K 4; end = p counter hop."""
    attn = (2.0 if T == 1 else 2.6) if attn is None else attn
    if qkv_first:
        ph = [dict(name="qkv q/k/v", kb=15.7e6 / 132 / KB, dep=lambda d, h: 0.0),
              dict(name="qkv gate", kb=12.6e6 / 132 / KB, dep=lambda d, h: 0.0)]
    else:
        ph = [dict(name="qkv", kb=28.3e6 / 132 / KB, dep=lambda d, h: 0.0)]
    j = len(ph)
    ph += [dict(name="o", kb=12.6e6 / 132 / KB, dep=lambda d, h: None if d[0] is None else d[0] + 3 * h + attn),
           dict(name="router", kb=8.4e6 / 132 / KB, dep=lambda d, h: None if d[j] is None else d[j] + h + 0.2),
           dict(name="pooler", kb=50.3e6 / 132 / KB, dep=lambda d, h: None if d[j] is None else d[j] + 2 * h + 0.2)]
    return ph, run(ph, ring_kb, hop=hop, k=k, tail=hop)


def back(E, ring_kb, hop=1.0, k=2.5):
    """C8-B: routing + x_e prologue (0.5 us after the PDL wait), W13 then W2 granules (per-expert counters: no wait at
    E >= 2, a full dependency at E = 1), unpooler tiles streamed during the y/ss hop; tail = m + moments hop + writes."""
    ph = [dict(name="W13", kb=E * 18.87e6 / 132 / KB, dep=lambda d, h: 0.5),
          dict(name="W2", kb=E * 9.44e6 / 132 / KB, dep=lambda d, h: 0.5 + (0.0 if E >= 2 else 1.5)),
          dict(name="unpooler", kb=6.3e6 / 132 / KB, dep=lambda d, h: None if d[1] is None else d[1] + h)]
    return ph, run(ph, ring_kb, t0=1.5, hop=hop, k=k, tail=0.6 + hop + 0.3)


if __name__ == "__main__":
    for T, ring in ((1, 192), (8, 120)):
        ph, (end, stall, peak, done) = front(T, ring)
        print(f"C8-F T={T} ring {ring} KB/SM: end {end:.1f} us, stall {stall:.2f}, peak {peak:.0f} KB | "
              + ", ".join(f"{p['name']} {x:.1f}" for p, x in zip(ph, done)))
    for E, ring in ((2.5, 184), (1, 184), (7.75, 110)):
        ph, (end, stall, peak, done) = back(E, ring)
        print(f"C8-B E={E} ring {ring} KB/SM: end {end:.1f} us, stall {stall:.2f}, peak {peak:.0f} KB | "
              + ", ".join(f"{p['name']} {x:.1f}" for p, x in zip(ph, done)))
    print("front-half sensitivity at T=8, ring 120 KB (end us / producer stall us): q/k/v-first vs natural qkv order")
    for hop in (0.7, 1.0, 1.5, 2.0):
        for attn in (2.6, 3.5):
            for k in (2.5, 1.3):
                _, a = front(8, 120, hop, attn, k)
                _, b = front(8, 120, hop, attn, k, qkv_first=False)
                print(f"  hop {hop:.1f} attn {attn:.1f} consumer x{k:.1f}: {a[0]:5.1f} / {a[1]:4.2f}   natural {b[0]:5.1f} / {b[1]:4.2f}")

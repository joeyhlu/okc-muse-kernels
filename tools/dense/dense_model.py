"""Per-shape budget + phase predictions for muse_bf16_proj (paper model, no GPU)."""
import math

LAYERS = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192),
          "unpooler": (8192, 4096), "router": (256, 8192)}
LOADS = [("decode1", 1), ("decode16", 16), ("decode32", 32), ("verify1", 8), ("verify16", 128), ("verify32", 256)]
ORDER = ["qkv_gate", "o", "pooler", "unpooler", "router"]

# judge check 15fa0da01e2246f6 (sglang, ours), per load in ORDER
SGL = {"decode1": [13.60, 6.67, 21.30, 6.82, 10.40], "decode16": [13.76, 6.67, 21.10, 18.24, 14.66],
       "decode32": [14.27, 6.98, 21.73, 18.69, 15.42], "verify1": [13.98, 6.72, 20.96, 6.98, 26.50],
       "verify16": [16.86, 7.23, 25.50, 19.74, 16.32], "verify32": [19.68, 9.31, 31.15, 24.42, 17.44]}
OURS = {"decode1": [11.56, 6.30, 15.65, 5.17, 11.19], "decode16": [13.54, 7.56, 22.53, 15.99, 11.09],
        "decode32": [16.07, 8.72, 26.49, 17.88, 15.24], "verify1": [13.38, 7.80, 16.35, 7.09, 14.72],
        "verify16": [17.18, 7.39, 24.56, 19.66, 12.82], "verify32": [20.31, 9.41, 30.68, 24.46, 15.25]}
# NathenFernandes v18 (proj 1.5186, public repo results/v18_shapes.csv): theirs / their-run sglang
NAT = {"decode1": [9.34, 5.10, 13.29, 3.72, 3.94], "decode16": [10.25, 5.48, 13.98, 13.74, 4.23],
       "decode32": [12.16, 5.81, 16.13, 15.60, 5.04], "verify1": [9.69, 5.50, 13.61, 4.28, 4.34],
       "verify16": [13.99, 6.89, 20.49, 19.37, 6.19], "verify32": [17.09, 8.98, 24.19, 23.32, 8.44]}
NAT_SGL = {"decode1": [14.10, 7.04, 21.25, 6.78, 10.46], "decode16": [14.32, 6.85, 21.66, 18.72, 14.66],
           "decode32": [14.62, 6.91, 21.84, 18.78, 15.07], "verify1": [14.56, 6.94, 21.30, 7.01, 26.30],
           "verify16": [16.83, 7.42, 25.09, 19.98, 16.29], "verify32": [19.50, 9.34, 30.96, 24.29, 17.71]}

A0 = 2.2      # us, one-launch intercept in judge units (stream lab S1-il fit; Nathen v18 matches it)
R = 4.54e12   # B/s asymptote of that fit
B12 = 12.01   # P12 bits/weight incl. metadata
B11 = 11.30   # P11 (3-bit window + Delta escapes), phase 3


def nk(layer, m):
    n, k = LAYERS[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024
    return n, k


def stream_us(nbytes, m, extra=0.0):
    a = A0 + (0.1 if m == 8 else 0.0) + (0.3 if m == 16 else 0.0) + (0.5 if m == 32 else 0.0)
    return a + nbytes / R * 1e6 + extra


def flop_us(layer, m, tf):
    n, k = nk(layer, m)
    f = 2.0 * m * n * k * (2 if layer == "router" else 1)
    return f / (tf * 1e12) * 1e6


def pred(layer, m, phase):
    n, k = nk(layer, m)
    qkv_pen = 0.5 if layer == "qkv_gate" else 0.0   # qkv36M-type deficit seen in stream lab + Nathen
    bits = B11 if (phase >= 3 and m <= 32 and layer != "router") else B12
    if layer == "router":
        if m <= 8:
            return stream_us(n * k * 4, m)                      # FP32 weights, FFMA, cluster-8 reduce
        if m <= 32:
            return stream_us(n * k * (B12 + 16) / 8, m)         # hi P12 + lo raw BF16, pairs summed in DSMEM
        if phase < 2:
            return None
        return 2.5 + max(flop_us(layer, m, 500), (n * k * 3.5 + m * k * 2) / R * 1e6) + 0.8  # K-concat hi|lo GEMM
    if m <= 32:
        return stream_us(n * k * bits / 8, m, qkv_pen)
    if m == 128:
        if phase < 2:
            return None
        return 2.5 + max(n * k * B12 / 8 / R * 1e6, flop_us(layer, m, 650)) + 0.5 + qkv_pen
    if m == 256:
        if phase < 3:
            return None
        return NAT["verify32"][ORDER.index(layer)]          # phase 3 target = public v18 level
    return None


def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    print("== per-shape budget (us) ==")
    print(f"{'load':9s} {'layer':9s} {'BF16MB':>7s} {'P12MB':>6s} {'floor':>6s} {'sgl':>6s} {'ours':>6s} "
          f"{'nat':>6s} {'#50u':>6s} {'p1':>6s} {'p2':>6s} {'p3':>6s}")
    phases = {1: {}, 2: {}, 3: {}}
    for load, m in LOADS:
        for i, layer in enumerate(ORDER):
            n, k = nk(layer, m)
            wb = 4 if layer == "router" else 2
            bf = n * k * wb / 1e6
            p12 = n * k * B12 / 8 / 1e6 if layer != "router" else float("nan")
            floor = 1e6 * max((n * k * wb + m * k * 2 + m * n * wb) / 4.8e12, 2.0 * m * n * k / 989e12)
            row = []
            for ph in (1, 2, 3):
                p = pred(layer, m, ph)
                if p is None:          # keep the incumbent route
                    p = OURS[load][i]
                p = min(p, OURS[load][i]) if ph >= 1 else p
                phases[ph].setdefault(load, []).append(p)
                row.append(p)
            print(f"{load:9s} {layer:9s} {bf:7.2f} {p12:6.2f} {floor:6.2f} {SGL[load][i]:6.2f} {OURS[load][i]:6.2f} "
                  f"{NAT[load][i]:6.2f} {SGL[load][i] / 1.530:6.2f} {row[0]:6.2f} {row[1]:6.2f} {row[2]:6.2f}")
    print()
    print("== per-load speedups (vs this check's SGLang column) ==")
    res = {}
    for name, table in [("ours", OURS), ("nathen*", None), ("phase1", phases[1]), ("phase2", phases[2]),
                        ("phase3", phases[3])]:
        sp = []
        for load, m in LOADS:
            if name == "nathen*":
                sp.append(sum(NAT_SGL[load]) / sum(NAT[load]))
            else:
                sp.append(sum(SGL[load]) / sum(table[load]))
        res[name] = sp
        print(f"{name:8s} " + " ".join(f"{s:6.3f}" for s in sp) + f"   geomean {geo(sp):.3f}")
    # haircut: realise only h of the predicted saving vs ours on every shape
    print()
    for h in (1.0, 0.8, 0.6):
        for ph in (2, 3):
            sp = []
            for load, m in LOADS:
                t = [o - h * (o - p) for o, p in zip(OURS[load], phases[ph][load])]
                sp.append(sum(SGL[load]) / sum(t))
            print(f"haircut {h:.1f} phase{ph}: " + " ".join(f"{s:6.3f}" for s in sp) + f"   geomean {geo(sp):.3f}")
    # value of single levers on top of ours
    print()
    def lever(sel):
        sp = []
        for load, m in LOADS:
            t = list(OURS[load])
            for i, layer in enumerate(ORDER):
                if sel(load, m, layer):
                    t[i] = phases[3][load][i] if m == 256 else phases[2][load][i]
            sp.append(sum(SGL[load]) / sum(t))
        return geo(sp)
    base = geo(res["ours"])
    for name, sel in [("router all T", lambda l, m, y: y == "router"),
                      ("PS T<=32 dense", lambda l, m, y: m <= 32 and y != "router"),
                      ("PT T=128 dense", lambda l, m, y: m == 128 and y != "router"),
                      ("T=256 dense v18-level", lambda l, m, y: m == 256 and y != "router"),
                      ("router + PS<=32", lambda l, m, y: y == "router" or m <= 32),
                      ("router + PS<=32 + PT128", lambda l, m, y: y == "router" or m <= 128)]:
        g = lever(sel)
        print(f"lever {name:26s} geomean {g:.3f}  (x{g / base:.3f} over ours)")


if __name__ == "__main__":
    main()

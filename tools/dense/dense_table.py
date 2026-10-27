import importlib.util, sys
spec = importlib.util.spec_from_file_location("dm", "dense_model.py"); dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)
rows = []
print("| load (m) | proj | BF16 MB | P12 MB | floor | SGLang | ours | v18 | #50 unif. | target p2 | x SGLang | p3 | x SGLang |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for load, m in dm.LOADS:
    for i, layer in enumerate(dm.ORDER):
        n, k = dm.nk(layer, m)
        wb = 4 if layer == "router" else 2
        bf = n * k * wb / 1e6
        if layer == "router":
            p12 = "8.39 f32" if m <= 8 else "7.34 hi/lo"
        else:
            p12 = f"{n*k*dm.B12/8/1e6:.2f}"
        floor = 1e6 * max((n*k*wb + m*k*2 + m*n*wb) / 4.8e12, 2.0*m*n*k/989e12)
        s = dm.SGL[load][i]; o = dm.OURS[load][i]; v = dm.NAT[load][i]
        p2 = dm.pred(layer, m, 2); p2 = o if p2 is None else min(p2, o)
        p3 = dm.pred(layer, m, 3); p3 = o if p3 is None else min(p3, o)
        nm = {"qkv_gate": "qkv", "o": "o", "pooler": "pooler", "unpooler": "unpool" + (" slice" if m <= 8 else ""), "router": "router"}[layer]
        print(f"| {load} ({m}) | {nm} | {bf:.1f} | {p12} | {floor:.1f} | {s:.2f} | {o:.2f} | {v:.2f} | {s/1.530:.1f} | {p2:.1f} | {s/p2:.2f} | {p3:.1f} | {s/p3:.2f} |")

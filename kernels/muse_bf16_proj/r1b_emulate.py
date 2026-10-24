"""CPU checks for r1b_router_gemv.py (not a submission; Mac, torch 2.5.1, no GPU).

  A. L1 router hi||lo numerics: the judge's own generate_input / check_implementation (problems/muse_bf16_proj,
     run on CPU) against BF16 x BF16 products with FP32 accumulation for the kcat / ncat / two modes, plus the
     FP32-direct GEMV numerics, on the 9 test shapes' router analogues and every scored router m.
  B. L2 GEMV: a literal NumPy port of gemv_kernel's index math (CTA/warp/lane/chunk/u/j -> n, k; x stage;
     shuffle-xor tree; red[] smem layout; split partials; last-CTA fixup) for every route on its real shapes:
     B1 coverage (every (n, k) multiplied exactly once per token, every output written exactly once),
     B2 integer data -> bit-exact x @ W^T, B3 real data (BF16 weights / FP32 router, probes, outliers) through the
     judge's check_implementation.
  C. Dispatch: which path every scored and test shape takes.
Usage: python3 -I r1b_emulate.py [--quick]
"""

import importlib.util
import math
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
PROB = HERE.parent.parent / "problems" / "muse_bf16_proj"
sys.path.insert(0, str(PROB))
import reference as R  # noqa: E402  (the emulator may; the submission never does)

spec = importlib.util.spec_from_file_location("r1b", HERE / "r1b_router_gemv.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

QUICK = "--quick" in sys.argv
SCORED_M = (1, 8, 16, 32, 128, 256)
TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"),
         (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]

t_start = time.time()
torch.manual_seed(0)
W = R.generate_weights(R.WEIGHTS_SEED, device="cpu")
print(f"weights generated in {time.time() - t_start:.1f}s")
FAIL = []


def check(name, ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'} {name}: {msg}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------------------------------------------
# A. router numerics
# ------------------------------------------------------------------------------------------------
print("\n== A. router: hi||lo (L1) and FP32-direct (L2) vs FP64, judge's checker ==")
w32 = W["router"]
hi, lo = S.split_router(w32)
recon = hi.double() + lo.double()
rel_w = float(((recon - w32.double()).abs() / w32.double().abs().clamp(min=1e-30)).max())
print(f"  split: max |hi+lo-w|/|w| = {rel_w:.2e} (bound 2^-17 = {2**-17:.2e}); hi exact-bf16, lo exact-bf16")
check("split bound", rel_w <= 2 ** -17, f"{rel_w:.2e}")
cat_k = torch.cat((hi, lo), 1).float()
cat_n = torch.cat((hi, lo), 0).float()


def router_kcat(x):   # cuBLAS BF16 GEMM, K = 16384, FP32 accumulate: products are exact in FP32
    return torch.cat((x, x), 1).float() @ cat_k.T


def router_ncat(x):
    y = x.float() @ cat_n.T
    return y[:, :256] + y[:, 256:]


def router_two(x):
    acc = x.float() @ hi.float().T
    acc += x.float() @ lo.float().T
    return acc


def router_fp32(x):   # L2 GEMV on the FP32 weight
    return x.float() @ w32.T


def router_bf16_only(x):   # must FAIL (the TASK says so) -> proves the check has teeth
    return x.float() @ hi.float().T


router_ms = sorted(set(SCORED_M) | {8, 9, 2, 5, 33, 512})
for m in router_ms:
    for dist in ("normal", "outliers"):
        d = R.generate_input(m, "router", seed=100 + m, dist=dist, device="cpu")
        x = d["x"]
        res = []
        for name, fn in (("kcat", router_kcat), ("ncat", router_ncat), ("two", router_two), ("fp32", router_fp32)):
            out = fn(x)
            ok, msg = R.check_implementation(d, out, W)
            ref = R.ref_kernel(d, W).double()
            tot = float((out.double() - ref).norm() / ref.norm())
            rows = R.probe_rows(x)
            pr = float(((out[rows].double() - ref[rows]).norm(dim=-1) / ref[rows].norm(dim=-1)).max()) if len(rows) else 0.0
            res.append(f"{name} {tot:.1e}/p{pr:.1e}")
            if not ok:
                check(f"router {name} m={m} {dist}", False, msg)
        okb, _ = R.check_implementation(d, router_bf16_only(x), W)
        res.append(f"bf16-only {'passes(!)' if okb else 'fails(ok)'}")
        print(f"  m={m:4d} {dist:8s}: " + "  ".join(res) + f"   [tol {R.ROUTER_TOL:g}, probe {R.PROBE_TOL:g}]")
        if okb:
            FAIL.append(f"bf16-only passed m={m}")

# ------------------------------------------------------------------------------------------------
# B. GEMV kernel emulation (literal port of gemv_kernel)
# ------------------------------------------------------------------------------------------------
U = S.UNROLL
NW = S.NWARP


def run_kernel(x, w, n, route, vec, fixup=True, count_visits=False):
    """Port of gemv_kernel + fixup/reduce. x f32 [M,K]; w f32 [Nw,K]. Returns out (f32, pre-rounding),
    out write counts [M, n], visit counts [n, K] per token tile (if asked)."""
    M, K = x.shape
    mt, rpw, WK, split = route
    g = S.geometry(M, n, K, vec, route)
    assert g is not None
    gx, gy, gz = g["grid"]
    kslice, CH, nch, rows_cta = g["kslice"], g["chunk"], g["nchunk"], g["rows_cta"]
    assert CH == 32 * vec * U and rows_cta == (NW // WK) * rpw
    # x stage per (bz, by): xs[t, kk] = x[t0 + t, kbase + kk] for t < mloc, else 0
    vpr = kslice >> 3
    xs = np.zeros((gz, gy, mt, kslice), np.float32)
    for bz in range(gz):
        t0 = bz * mt
        mloc = min(mt, M - t0)
        for by in range(gy):
            kbase = by * kslice
            for i in range(mt * vpr):     # literal staging loop (vector granularity)
                t, v = i // vpr, i - (i // vpr) * vpr
                if t < mloc:
                    xs[bz, by, t, v * 8:(v + 1) * 8] = x[t0 + t, kbase + v * 8: kbase + v * 8 + 8]
    # lanes: [gx, gy, gz, NW, 32]
    bx = np.arange(gx)[:, None, None, None, None]
    by = np.arange(gy)[None, :, None, None, None]
    bz = np.arange(gz)[None, None, :, None, None]
    warp = np.arange(NW)[None, None, None, :, None]
    lane = np.arange(32)[None, None, None, None, :]
    wk = warp % WK
    wr = warp // WK
    row0 = bx * rows_cta + wr * rpw
    kbase = by * kslice
    shape = (gx, gy, gz, NW, 32)
    acc = np.zeros(shape + (rpw, mt), np.float32)
    visits = np.zeros((gz, n, K), np.int32) if count_visits else None
    BX, BY, BZ, WP, LN = np.broadcast_arrays(bx, by, bz, warp, lane)
    for it in range((nch + WK - 1) // WK + 1):
        c = wk + it * WK                                  # warp's it-th chunk
        active = np.broadcast_to(c < nch, shape)
        if not active.any():
            break
        for u in range(U):
            kk = c * CH + u * 32 * vec + lane * vec       # offset inside the K slice
            kk = np.broadcast_to(kk, shape)
            for r in range(rpw):
                rows = np.broadcast_to(row0 + r, shape)
                kg = np.broadcast_to(kbase, shape) + kk   # global k of element j = 0
                for j in range(vec):
                    a_rows, a_k = rows[active], kg[active] + j
                    wf = w[a_rows, a_k]
                    if count_visits:
                        np.add.at(visits, (BZ[active], a_rows, a_k), 1)
                    for t in range(mt):
                        xf = xs[BZ[active], BY[active], t, kk[active] + j]
                        sel = acc[..., r, t]
                        sel[active] = sel[active] + wf * xf    # fp32 (kernel: fmaf)
    # shuffle-xor tree
    for off in (16, 8, 4, 2, 1):
        acc = acc + acc[:, :, :, :, np.arange(32) ^ off]
    red = np.zeros((gx, gy, gz, NW * rpw * mt), np.float32)
    for wp in range(NW):
        for r in range(rpw):
            for t in range(mt):
                red[:, :, :, (wp * rpw + r) * mt + t] = acc[:, :, :, wp, 0, r, t]
    out = np.zeros(M * n, np.float32)
    part = np.zeros((split, M * n), np.float32)
    wcount = np.zeros(M * n, np.int32)
    pcount = np.zeros((split, M * n), np.int32)
    for X in range(gx):
        for Y in range(gy):
            for Z in range(gz):
                t0 = Z * mt
                mloc = min(mt, M - t0)
                for i in range(rows_cta * mt):
                    t, rr = i // rows_cta, i % rows_cta
                    if t >= mloc:
                        continue
                    wr2, r2 = rr // rpw, rr % rpw
                    s = np.float32(0)
                    for q in range(WK):
                        s = np.float32(s + red[X, Y, Z, ((wr2 * WK + q) * rpw + r2) * mt + t])
                    o = (t0 + t) * n + X * rows_cta + rr
                    if split == 1:
                        out[o] = s
                        wcount[o] += 1
                    else:
                        part[Y, o] = s
                        pcount[Y, o] += 1
    if split > 1:
        assert (pcount == 1).all(), "partials written != once"
        # fixup (last CTA per tile, any arrival order: the sum order is q = 0..S-1 either way) or reduce kernel
        for o in range(M * n):
            if fixup:
                s = np.float32(0)
                for q in range(split):
                    s = np.float32(s + part[q, o])
            else:
                s = part[0, o]
                for q in range(1, split):
                    s = np.float32(s + part[q, o])
            out[o] = s
            wcount[o] += 1
    return out.reshape(M, n), wcount.reshape(M, n), visits


def bf16_round(t):
    return t.to(torch.bfloat16).float()


print("\n== B. GEMV emulation per route ==")
route_cases = []
for (layer, b), route in S.GEMV_ROUTES.items():
    ms = {1: [1], 8: [8, 5, 7, 2] if not QUICK else [8], 32: [9, 16, 32] if not QUICK else [16]}[b]
    if layer == "router" and b == 8:
        ms = [1] + ms
    for m in ms:
        route_cases.append((layer, b, route, m))

for layer, b, route, m in route_cases:
    t0 = time.time()
    n_full, K = R.LAYERS[layer]
    n = R.out_features(layer, m)
    fp32 = layer == "router"
    vec = 4 if fp32 else 8
    g = S.geometry(m, n, K, vec, route)
    if g is None:
        check(f"{layer}/{b} m={m}", False, f"geometry rejects route {route}")
        continue
    wl = W[layer][:n].float()
    # B1+B2: small-integer data -> every product and partial sum exact in fp32
    gen = torch.Generator().manual_seed(7 * m + n)
    xi = torch.randint(-3, 4, (m, K), generator=gen).float()
    wi = torch.randint(-3, 4, (n, K), generator=gen).float()
    want_visits = (m <= 8 and n * K <= 4096 * 8192)
    out_i, wc, vis = run_kernel(xi.numpy(), wi.numpy(), n, route, vec, fixup=S.R1B_FIXUP, count_visits=want_visits)
    exact = np.array_equal(out_i, (xi.double() @ wi.double().T).numpy().astype(np.float32))
    once = bool((wc == 1).all())
    cov = "skipped"
    if want_visits:
        cov = "exactly-once" if bool((vis == 1).all()) else f"BAD min {vis.min()} max {vis.max()}"
    # B3: real data through the judge's checker
    d = R.generate_input(m, layer, seed=300 + m, dist="outliers" if m in (5, 9) else "normal", device="cpu")
    xr = d["x"].float()
    out_r, _, _ = run_kernel(xr.numpy(), wl.numpy(), n, route, vec, fixup=S.R1B_FIXUP)
    out_t = torch.from_numpy(out_r)
    out_t = out_t if fp32 else out_t.to(torch.bfloat16)
    ok, msg = R.check_implementation(d, out_t, W)
    gx, gy, gz = g["grid"]
    print(f"  {layer:8s} b={b:2d} m={m:2d} route={route} grid=({gx},{gy},{gz}) kslice={g['kslice']} "
          f"chunks={g['nchunk']} smem={g['smem'] // 1024}KiB ({time.time() - t0:.1f}s)")
    check(f"{layer}/{b} m={m} int-exact", exact, "bit-exact vs x@W^T" if exact else "MISMATCH")
    check(f"{layer}/{b} m={m} writes", once, "every output written once" if once else "write count != 1")
    if want_visits:
        check(f"{layer}/{b} m={m} coverage", cov == "exactly-once", cov)
    check(f"{layer}/{b} m={m} judge", ok, msg)

# non-fixup reduce path on one split route
layer, route, m = "router", S.GEMV_ROUTES[("router", 8)], 8
n, K = 256, 8192
xi = torch.randint(-3, 4, (m, K)).float()
wi = torch.randint(-3, 4, (n, K)).float()
o2, wc2, _ = run_kernel(xi.numpy(), wi.numpy(), n, route, 4, fixup=False)
check("reduce-kernel path router m=8", np.array_equal(o2, (xi.double() @ wi.double().T).numpy().astype(np.float32))
      and bool((wc2 == 1).all()), "bit-exact, written once")

# geometry edge: N tail / bad splits must be rejected (fall back to cuBLAS), never launched
check("geometry rejects N%rows", S.geometry(1, 1000, 8192, 8, (1, 2, 1, 1)) is None, "N=1000, 16 rows/CTA")
check("geometry rejects kslice%chunk", S.geometry(1, 2304, 8192, 8, (1, 1, 1, 16)) is None, "kslice 512 < chunk 1024")
check("geometry rejects smem", S.geometry(8, 2304, 8192, 8, (8, 2, 1, 1)) is None, "x stage 128 KiB")

# ------------------------------------------------------------------------------------------------
# C. dispatch
# ------------------------------------------------------------------------------------------------
print("\n== C. dispatch (GEMV module assumed built, all routes live) ==")
fake = {name: (torch.empty((1, 1), dtype=torch.float32) if name == "router" else torch.empty((1, 1), dtype=torch.bfloat16))
        for name in R.LAYERS}
fake.update(mod=object(), routes=dict(S.GEMV_ROUTES), router_mode=S.R1B_ROUTER_SPLIT_MODE)
for name in R.LAYERS:
    fake[name] = W[name]
shapes = [(m, l, "scored") for m in SCORED_M for l in R.LAYERS] + [(m, l, f"test {dist}") for m, l, dist in TESTS]
for m, layer, tag in shapes:
    n = R.out_features(layer, m)
    k = R.LAYERS[layer][1]
    route = S._gemv_route(fake, layer, m, n, k)
    if route is not None:
        path = f"GEMV {route}"
    elif layer == "router":
        path = f"L1 router split ({S.R1B_ROUTER_SPLIT_MODE})"
    else:
        path = "cuBLAS (SGLang dispatch)"
    print(f"  {tag:15s} m={m:4d} {layer:8s} n={n:5d}: {path}")

print(f"\nDONE in {time.time() - t_start:.0f}s; failures: {FAIL if FAIL else 'none'}")
sys.exit(1 if FAIL else 0)

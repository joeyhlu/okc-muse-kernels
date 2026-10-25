# p1 lab (round 3): P12 lossless BF16 weights + p1s streaming GEMV for T <= 8 (LAB ONLY, never submit)
"""P1 lab, round 3. The kernel, packer, verify and layer API live in kernels/muse_bf16_proj/p1_module.py (the
integration module); this file times them. In ONE process (CUDA graph per call, 256 MB L2 flush before every replay,
paired alternating order): pack + bit-exact verify, GEMV checks (one-hot escape columns, random x vs FP64, repeat
bit-identical, split-K determinism stress with the FP32-atomic negative control), then per (projection, T in {1, 8}):
today's k3_v12 path vs p1s on BF16 vs p1s packed (each swept over single-wave geometries) and the pure-LDG stream
probe at the packed winner's grid (the ceiling). Prints the table and paste-ready P1_GEOMETRY / P1_ROUTE dicts.

    python3 tools/box/sass_gate.py kernels/muse_bf16_proj/p1_module.py        # compile gate (the CUDA lives there)
    python3 kernels/muse_bf16_proj/p1_lab.py 2>&1 | tee box_out/p1_lab3.log
"""

import argparse
import importlib.util
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

_spec = importlib.util.spec_from_file_location("p1_module", os.path.join(HERE, "p1_module.py"))
P1M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1M)

P1_TS = (1, 8)          # timed: decode 1 (T = 1) and verify 1 (T = 8); T >= 16 stays on cuBLAS
P1_EXTRA_TS = (2, 5)    # correctness only; T = 9 must be refused by the host check
REPS_SWEEP = 9
REPS_FINAL = 41
GATE = 1.15             # packed must be >= 15% faster (net of the graph replay floor) than today's path
N_SMS = 132
SWEEP_W = (2, 4, 6, 8, 12, 16)
SWEEP_S = (1, 2, 3, 4, 5, 6, 7, 8)
SWEEP_D = (1, 2, 3)
INST = {(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (2, 3), (3, 3)}   # (depth, mode) built


def p1_build():
    return P1M.p1_build(verbose=True)


pack_p12 = P1M.p1_pack
describe = P1M.p1_describe
mode_of = P1M.p1_mode_of
p1s_call = P1M.p1_call


def unpack(mod, P, use_esc=True):
    return P1M.p1_unpack(mod, P, use_esc)


def make_weights(dev, seed):
    """The judge's generator (problems/muse_decode_layer/reference.py) when importable, else the same recipe here."""
    try:
        sys.path.insert(0, os.path.join(REPO, "problems", "muse_decode_layer"))
        import reference as R   # lab only (never in a submission)
        w = R.generate_weights(seed, device=str(dev))
        print(f"[p1] weights: reference.generate_weights(seed={seed})", flush=True)
        return {k: w[k] for k in ("qkv_gate", "o", "pooler", "unpooler", "router")}
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] reference.py unavailable ({ex!r}); using the same recipe locally", flush=True)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def dense(n, k):
        return (torch.randn((n, k), generator=gen, device=dev) / math.sqrt(k)).to(torch.bfloat16)
    mean = torch.randn((8192,), generator=torch.Generator().manual_seed(5354)).to(dev)
    rows = 0.045 * torch.randn((256, 8192), generator=gen, device=dev) + \
        0.07 * torch.randn((256, 1), generator=gen, device=dev) * mean[None, :] / math.sqrt(8192)
    return {"qkv_gate": dense(2304, 8192), "o": dense(8192, 1024), "pooler": dense(4096, 8192),
            "unpooler": dense(8192, 4096), "router": rows.contiguous()}


def stress_matrix(dev, n=512, k=1024, seed=99):
    """Held-out style: per-row log-normal scales, Student-t(2) entries, 1% zeros, subnormals, inf, NaN, outliers."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    sc = torch.exp2(-6.0 + 1.5 * torch.randn((n, 1), generator=gen, device=dev))
    z = torch.randn((n, k), generator=gen, device=dev)
    chi = (torch.randn((n, k), generator=gen, device=dev) ** 2 + torch.randn((n, k), generator=gen, device=dev) ** 2) / 2
    w = (sc * z / chi.clamp_min(1e-12).sqrt()).to(torch.bfloat16)
    w[torch.rand((n, k), generator=gen, device=dev) < 0.01] = 0
    bits = w.view(torch.int16)
    specials = torch.tensor([0x0001, -32767, 0x007F, 0x7F80, -128, 0x7FC0, -63, 0x7F7F, 0x0080, -32768], dtype=torch.int16,
                            device=dev)   # -128 = 0xFF80 (-inf), -63 = 0xFFC1 (NaN), -32767 = 0x8001, -32768 = -0
    pos = torch.randint(0, n * k, (200,), generator=gen, device=dev)
    bits.view(-1)[pos] = specials[torch.arange(200, device=dev) % specials.numel()]
    # one fully subnormal 16-row tile and one huge-outlier tile
    bits[32:48] = (bits[32:48] & -32768) | (bits[32:48] & 0x7F)
    w[64:80, :8] = 3.0e38
    return w


def stress_mild(dev, n=512, k=8192, seed=98):
    """Generator-like rows plus sparse specials (zeros, +-subnormals, +-inf, NaN) that stay within the patch budget,
    and one unscalable value (3e38) that forces its 256-row group to RAW: exercises the in-register patch path."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = (torch.randn((n, k), generator=gen, device=dev) / math.sqrt(k)).to(torch.bfloat16)
    bits = w.view(torch.int16)
    specials = torch.tensor([0x0000, -32768, 0x0001, -32767, 0x0003, 0x007F, 0x7F80, -128, 0x7FC0], dtype=torch.int16,
                            device=dev)
    pos = torch.randint(0, (n // 2) * k, (150,), generator=gen, device=dev)          # first 256-row group: patched
    bits.view(-1)[pos] = specials[torch.arange(150, device=dev) % specials.numel()]
    w[300, 5] = 3.0e38                                                                  # second group: RAW
    return w


# ================================================================================================ timing
def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_pair(ga, gb, flush, reps):
    """Paired alternating order, cold L2 (256 MB flush) before every replay. -> (median a, median b, median a/b)."""
    ta, tb, rr = [], [], []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for r in range(reps):
        order = (0, 1) if r % 2 == 0 else (1, 0)
        t = [0.0, 0.0]
        for which in order:
            flush.add_(1.0)
            ev[0].record()
            (ga if which == 0 else gb).replay()
            ev[1].record()
            torch.cuda.synchronize()
            t[which] = ev[0].elapsed_time(ev[1]) * 1e3
        ta.append(t[0]); tb.append(t[1]); rr.append(t[0] / t[1])
    return statistics.median(ta), statistics.median(tb), statistics.median(rr)


def time_one(g, flush, reps):
    ts = []
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        flush.add_(1.0)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(ts)


# ================================================================================================ today's path
def load_r1b():
    try:
        spec = importlib.util.spec_from_file_location("r1b_v2_lab", os.path.join(HERE, "r1b_v2.py"))
        r1b = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(r1b)
        mod = r1b._build()
        dev = torch.device("cuda")
        st = {"mod": mod, "dummy": torch.zeros((4,), dtype=torch.float32, device=dev),
              "counters": torch.zeros((r1b.COUNTERS,), dtype=torch.int32, device=dev)}
        return r1b, st
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] R1B unavailable -> T=1 'today' rows use cuBLAS (k3_v12 uses R1B there): {ex!r}", flush=True)
        return None, None


def bf16_gemm_fn():
    try:
        from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
        return lambda x, w: _bf16_gemm_dispatch_impl(x, w, None), "sglang _bf16_gemm_dispatch_impl"
    except Exception as ex:  # noqa: BLE001
        print(f"[p1] SGLang dispatch unavailable ({ex!r}); F.linear", flush=True)
        return (lambda x, w: F.linear(x, w)), "F.linear"




# ================================================================================================ calls (round 2)


# ================================================================================================ geometry sweep
def geometries(P, n_rows, min_ctas=1):
    """(warps per CTA, split-K CTAs per row tile, ring depth) accepted by the host checks (correctness sweeps)."""
    nrt = n_rows // 16
    out = []
    for warps in (4, 8, 16):
        for split in (1, 2, 4, 8):
            if split * warps > P["nsb"] or nrt * split < min_ctas:
                continue
            for depth in (1, 2):
                out.append((warps, split, depth))
    return out


_OCC = {}


def occ_of(mod, depth, mode, warps):
    key = (depth, mode, warps)
    if key not in _OCC:
        _OCC[key] = tuple(mod.p1s_info(depth, mode, warps))
    return _OCC[key]


def residency(mod, P, n_rows, geo, mode=None):
    """(regs/thread, local bytes, CTAs/SM by occupancy, average warps/SM, weight bytes in flight per SM, waves)."""
    warps, split, depth = geo
    mode = mode_of(P, n_rows) if mode is None else mode
    regs, local, occ = occ_of(mod, depth, mode, warps)
    ctas = (n_rows // 16) * split
    waves = ctas / max(1, occ * N_SMS)
    avg_warps = min(occ * warps, ctas * warps / N_SMS)
    sbb = 2048 if mode == 1 else 1536
    sb_per_warp = P["nsb"] / (split * warps)
    return regs, local, occ, avg_warps, avg_warps * min(depth, sb_per_warp) * sbb, waves


def sweep_geometries(mod, P, n_rows, limit=28):
    """Round 3: single-wave candidates (every CTA resident at once), ranked by weight bytes in flight per SM, plus the
    round-2b incumbent (16, 1, 2)."""
    mode = mode_of(P, n_rows)
    cands = []
    for warps in SWEEP_W:
        for split in SWEEP_S:
            if split * warps > P["nsb"]:
                continue
            for depth in SWEEP_D:
                if (depth, mode) not in INST:
                    continue
                regs, local, occ, wsm, infl, waves = residency(mod, P, n_rows, (warps, split, depth), mode)
                if occ < 1 or waves > 1.0 or local > 0:
                    continue
                cands.append((-infl, split, (warps, split, depth)))
    cands.sort()
    out = [c[2] for c in cands[:limit]]
    if (16, 1, 2) not in out and 16 * 1 <= P["nsb"]:
        out.append((16, 1, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5353)
    ap.add_argument("--quick", action="store_true", help="fewer geometries / reps")
    args = ap.parse_args()
    reps_sweep, reps_final = (5, 15) if args.quick else (REPS_SWEEP, REPS_FINAL)
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"[p1] torch {torch.__version__} cuda {torch.version.cuda} device {torch.cuda.get_device_name(0)}", flush=True)
    t_all = time.time()
    mod = p1_build()
    for depth, mode in sorted(INST):
        for warps in (4, 8, 16):
                regs, local, occ = mod.p1s_info(depth, mode, warps)
                print(f"[p1] p1s_kernel<D={depth}, MODE={mode}> warps {warps}: {regs} regs/thread, {local} B local, "
                      f"{occ} CTAs/SM ({occ * warps} warps/SM)", flush=True)
    W = make_weights(dev, args.seed)
    hi = W["router"].to(torch.bfloat16)
    lo = (W["router"] - hi.float()).to(torch.bfloat16)
    wcat = torch.cat((W["pooler"], hi, lo), 0).contiguous()             # k3_v12 "cat": [pooler; hi; lo]
    mats = {"qkv": W["qkv_gate"].contiguous(), "o": W["o"].contiguous(), "cat": wcat,
            "unpooler": W["unpooler"].contiguous()}

    # ---------------------------------------------------------------- 1. pack + bit-exact verification
    print("\n===== P1 PACK + VERIFY =====", flush=True)
    P, ok_all = {}, True
    for name, w in list(mats.items()) + [("stress", stress_matrix(dev)), ("stress_mild", stress_mild(dev))]:
        pk = pack_p12(w, name=name)
        u = unpack(mod, pk)
        torch.cuda.synchronize()
        mism = int((u.view(torch.int16) != w.view(torch.int16)).sum())
        neg = int((unpack(mod, pk, use_esc=False).view(torch.int16) != w.view(torch.int16)).sum())
        neg_ok = (neg > 0) if pk["n_esc"] > 0 else True
        # the per-k-tile patch needs every row tile's entries sorted by (sb, kt)
        e = pk["esc"][:, 0].long()
        keyv = (e & 0xFFFF) * 4 + ((e >> 23) & 3)
        eo = pk["eoff"].long().cpu().tolist()
        kc = keyv.cpu().tolist()
        sorted_ok = all(kc[j] <= kc[j + 1] for r in range(pk["nrt"]) for j in range(eo[r], eo[r + 1] - 1))
        ok = mism == 0 and neg_ok and sorted_ok
        ok_all &= ok
        print(f"[p1] {describe(pk)}", flush=True)
        print(f"[p1]   unpack mismatches {mism} (must be 0) | negative control without patches: {neg} mismatches "
              f"(must be > 0 when escapes exist) | entries key-sorted {sorted_ok} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        P[name] = pk
        if name.startswith("stress"):
            mats[name] = w
    PR = {}   # the same kernel on uncompressed BF16 (every tile RAW): separates structure from compression
    for name in ("qkv", "o", "cat", "unpooler"):
        PR[name] = pack_p12(mats[name], esc_max=-1, name=name + "_raw")
        mism = int((unpack(mod, PR[name]).view(torch.int16) != mats[name].view(torch.int16)).sum())
        ok_all &= mism == 0 and PR[name]["n_raw_tiles"] == PR[name]["nrt"]
        print(f"[p1] {describe(PR[name])} | unpack mismatches {mism}", flush=True)
    if not ok_all:
        print("[p1] GATE FAIL: unpack is not bit-exact; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 2. GEMV correctness
    print("\n===== P1 GEMV CHECKS =====", flush=True)
    gen = torch.Generator(device=dev).manual_seed(7)

    def check_geos(pk, n_rows):
        geos = geometries(pk, n_rows)
        pick = [geos[0]]
        pick += [g for g in geos if g[1] > 1 and g[2] == 2][:1]       # split-K atomics + fixup path
        pick += [g for g in geos if g[1] == 1 and g[2] == 2][-1:]     # deepest S = 1 path
        return list(dict.fromkeys(pick))

    def one_hot_cols(pk, w, n_rows, cols, mb):
        """x = e_k rows: y[t, :] must equal W[:n_rows, k_t] exactly (one exact product per output). Rows with an
        inf/NaN weight are skipped (0 * inf = NaN in any GEMM); subnormals read back as 0 are counted apart
        (tensor-core input flush: the same on raw BF16 through cuBLAS)."""
        bad = flushed = 0
        fin = torch.isfinite(w[:n_rows].float()).all(1)
        for geo in check_geos(pk, n_rows):
            for i in range(0, len(cols), mb):
                cc = cols[i:i + mb]
                x = torch.zeros((len(cc), pk["K"]), dtype=torch.bfloat16, device=dev)
                x[torch.arange(len(cc), device=dev), torch.tensor(cc, device=dev)] = 1.0
                out = torch.empty((len(cc), n_rows), dtype=torch.bfloat16, device=dev)
                p1s_call(mod, pk, x, out, n_rows, n_rows, geo)
                ref = w[:n_rows, torch.tensor(cc, device=dev)].t().float()
                got = out.float()
                mis = (got != ref) & ~(torch.isnan(got) & torch.isnan(ref)) & fin[None, :]
                sub = mis & (got == 0) & (ref.abs() < 2.0 ** -126)
                flushed += int(sub.sum())
                bad += int((mis & ~sub).sum())
        return bad, flushed

    gemm_ok = True
    for name in ("qkv", "o", "cat", "unpooler", "stress", "stress_mild"):
        pk, w = P[name], mats[name]
        cols_esc = []
        if pk["n_esc"]:
            cols_esc = sorted(set(P1M.p1_esc_columns(pk["esc"]).tolist()))[:384]   # round-3 k-permuted lane map
        cols = cols_esc + torch.randint(0, pk["K"], (32,), generator=gen, device=dev).tolist()
        n_rows = pk["N"]
        bad8, fl8 = one_hot_cols(pk, w, n_rows, cols, 8)
        bad1, fl1 = one_hot_cols(pk, w, n_rows, cols[:6], 1)
        print(f"[p1] one-hot probe {name} (mode {mode_of(pk, n_rows)}): {len(cols)} columns ({len(cols_esc)} escape "
              f"columns) x {len(check_geos(pk, n_rows))} geometries: {bad8 + bad1} mismatches (must be 0); subnormals "
              f"flushed by the tensor core {fl8 + fl1}", flush=True)
        gemm_ok &= (bad8 + bad1) == 0
        if name.startswith("stress"):
            continue
        for var, pv in (("packed", pk), ("raw", PR[name])):
            for m in P1_TS + P1_EXTRA_TS:
                x = torch.randn((m, pv["K"]), generator=gen, device=dev).to(torch.bfloat16)
                ref = x.double() @ w[:n_rows].double().t()
                errs = []
                geos = geometries(pv, n_rows)
                for geo in geos[:: (3 if args.quick else 1)]:
                    out = torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
                    p1s_call(mod, pv, x, out, n_rows, n_rows, geo)
                    out2 = out.clone()
                    p1s_call(mod, pv, x, out, n_rows, n_rows, geo)
                    rel = float((out.double() - ref).norm() / ref.norm())
                    errs.append((rel, bool(torch.equal(out, out2)), geo))
                worst = max(errs)
                good = worst[0] < 4e-3 and all(er[1] for er in errs)
                gemm_ok &= good
                if m in P1_TS or not good:
                    print(f"[p1] gemv {name} {var} M={m}: {len(errs)} geometries, worst rel-L2 vs fp64 {worst[0]:.2e} "
                          f"(tol 4e-3) at {worst[2]}, repeat bit-identical {all(er[1] for er in errs)} -> "
                          f"{'ok' if good else 'FAIL'}", flush=True)
    # determinism stress (round 2 failed "repeat bit-identical" for M >= 2 at split 4/8: the FP32 atomic split-K sum
    # is order-dependent once S >= 3). Largest split, M = 8, 8 repeats: red_mode 0 (rank-ordered slots) must give ONE
    # result; red_mode 1 (round 2's atomics) is the negative control and is expected to give several.
    for name in ("qkv", "cat", "unpooler", "o"):
        for var, pv in (("packed", P[name]), ("raw", PR[name])):
            n_rows = pv["N"]
            geos = geometries(pv, n_rows)
            smax = max(g[1] for g in geos)
            geo = [g for g in geos if g[1] == smax and g[2] == 2][0]
            x = torch.randn((8, pv["K"]), generator=gen, device=dev).to(torch.bfloat16)
            distinct = {}
            for rm in (0, 1):
                res = []
                for _ in range(8):
                    out = torch.empty((8, n_rows), dtype=torch.bfloat16, device=dev)
                    p1s_call(mod, pv, x, out, n_rows, n_rows, geo, red_mode=rm)
                    res.append(out)
                torch.cuda.synchronize()
                uniq = []
                for r_ in res:
                    if not any(torch.equal(r_, u) for u in uniq):
                        uniq.append(r_)
                distinct[rm] = len(uniq)
                if rm == 1:
                    agree = float((res[0].float() - res[1].float()).abs().max())
            good = distinct[0] == 1
            gemm_ok &= good
            print(f"[p1] determinism {name} {var} {geo} M=8 x8: rank-ordered split-K {distinct[0]} distinct result(s) "
                  f"(must be 1) -> {'ok' if good else 'FAIL'} | negative control, FP32 atomics: {distinct[1]} distinct "
                  f"(expected > 1; max |diff| {agree:.1e})", flush=True)
    try:
        x9 = torch.randn((9, 8192), device=dev).to(torch.bfloat16)
        p1s_call(mod, P["qkv"], x9, torch.empty((9, 2304), dtype=torch.bfloat16, device=dev), 2304, 2304, (8, 1, 2))
        print("[p1] M=9 was accepted (expected a refusal -> current path)", flush=True)
        gemm_ok = False
    except Exception:  # noqa: BLE001
        print("[p1] M=9 refused by the host check (T > 8 stays on the current path): ok", flush=True)
    if not gemm_ok:
        print("[p1] GATE FAIL: GEMV checks failed; stopping before timing", flush=True)
        return 1

    # ---------------------------------------------------------------- 3. timing: today vs p1s on BF16 vs p1s packed
    print("\n===== P1 TIMING (CUDA graph, 256 MB flush per replay, paired alternating order) =====", flush=True)
    gemm, gemm_name = bf16_gemm_fn()
    r1b, r1b_st = load_r1b()
    flush = torch.empty((64 * 1024 * 1024,), dtype=torch.float32, device=dev)
    zbuf = torch.zeros((8, 8192), dtype=torch.bfloat16, device=dev)
    tiny = torch.zeros((1,), device=dev)
    floor_us = time_one(graph_of(lambda: tiny.add_(1.0)), flush, 21)
    print(f"[p1] graph replay floor (one tiny kernel): {floor_us:.2f} us; cuBLAS path: {gemm_name}", flush=True)

    def today(name, m, x):
        """(callable, label) = k3_v12's projection path at this T (T <= 8)."""
        if r1b is not None and m == 1 and name in ("qkv", "cat", "unpooler"):
            layer, n, w = {"qkv": ("qkv_gate", 2304, mats["qkv"]), "cat": ("pooler", 4096, W["pooler"]),
                           "unpooler": ("unpooler", 1024, mats["unpooler"])}[name]
            route = r1b.GEMV_ROUTES[(layer, 1)]
            return (lambda: r1b._gemv(r1b_st, x, w, n, route)), f"R1B GEMV {route}"
        if name == "qkv":
            return (lambda: gemm(x, mats["qkv"])), "cuBLAS"
        if name == "o":
            return (lambda: gemm(x, mats["o"])), "cuBLAS"
        if name == "cat":
            return (lambda: gemm(x, W["pooler"])), "cuBLAS pooler"
        ws_ = mats["unpooler"][:1024]
        return (lambda: torch.mm(x, ws_.t(), out=zbuf[:m, :1024])), "cuBLAS slice->zbuf"

    rows = []
    shapes = [("qkv", 8192, 2304), ("o", 1024, 8192), ("cat", 8192, 4096), ("unpooler", 4096, 1024)]
    for name, k, n_rows in shapes:
        for m in P1_TS:
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ldo = 8192 if name == "unpooler" else n_rows
            out = zbuf[:m] if name == "unpooler" else torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
            fa, label = today(name, m, x)
            ga = graph_of(fa)
            bests = {}
            for var, pv in (("packed", P[name]), ("raw", PR[name])):
                geos = sweep_geometries(mod, pv, n_rows, limit=12 if args.quick else 28)
                sweep, best = [], None
                for geo in geos:
                    try:
                        p1s_call(mod, pv, x, out, n_rows, ldo, geo)
                        gb = graph_of(lambda geo=geo, pv=pv: p1s_call(mod, pv, x, out, n_rows, ldo, geo))
                    except Exception as ex:  # noqa: BLE001
                        print(f"[p1]   {name} M={m} {var} geo {geo}: launch failed {ex!r}", flush=True)
                        continue
                    _, tb, _ = time_pair(ga, gb, flush, reps_sweep)
                    sweep.append((tb, geo))
                    if best is None or tb < best[0]:
                        best = (tb, geo, gb)
                sweep.sort()
                print(f"[p1] {name} M={m} {var} sweep (us, (warps,split,depth)): " +
                      ", ".join(f"{t:.1f} {g}" for t, g in sweep[:6]), flush=True)
                if best is not None:
                    bests[var] = best
            if "packed" not in bests:
                print(f"[p1] {name} M={m}: no packed geometry ran", flush=True)
                continue
            best = bests["packed"]
            t_a, t_b, ratio = time_pair(ga, best[2], flush, reps_final)
            net = (t_a - floor_us) / max(t_b - floor_us, 1e-3)
            t_r = c_ratio = float("nan")
            if "raw" in bests:
                t_r, _, c_ratio = time_pair(bests["raw"][2], best[2], flush, reps_final)
            mb = P[name]["bytes"] * n_rows / P[name]["N"] / 1e6
            regs, local, occ, wsm, infl, waves = residency(mod, P[name], n_rows, best[1])
            # pure-LDG stream probe at the packed winner's grid (same loads, XOR consume): the achievable ceiling
            pgeo = (best[1][0], best[1][1], max(2, best[1][2]))
            t_p = float("nan")
            try:
                p1s_call(mod, P[name], x, out, n_rows, ldo, pgeo, mode=3)
                gp = graph_of(lambda: p1s_call(mod, P[name], x, out, n_rows, ldo, pgeo, mode=3))
                t_p, _, _ = time_pair(gp, best[2], flush, reps_final)
            except Exception as ex:  # noqa: BLE001
                print(f"[p1]   {name} M={m}: probe failed {ex!r}", flush=True)
            rows.append((name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, best[1], regs, wsm, infl, waves, mb, t_p))
            print(f"[p1] {name} M={m}: today {label} {t_a:.2f} us | p1s BF16 {t_r:.2f} us | packed {t_b:.2f} us "
                  f"{best[1]} | paired x{ratio:.3f} | net x{net:.3f} | compression alone x{c_ratio:.3f} | "
                  f"{mb / max(t_b - floor_us, 1e-3):.2f} TB/s net | {regs} regs, {wsm:.1f} warps/SM, "
                  f"{infl / 1024:.0f} KB/SM in flight, {waves:.2f} waves | stream probe {t_p:.2f} us = "
                  f"{mb / max(t_p - floor_us, 1e-3):.2f} TB/s ceiling", flush=True)

    # ---------------------------------------------------------------- 3b. PDL lever (needs an early trigger to matter)
    print("\n===== P1 PDL (producer x.copy_ -> p1s, paired: pdl off / pdl on) =====", flush=True)
    for (name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, geo, *_rest) in rows:
        if m != 1:
            continue
        k, n_rows = {"qkv": (8192, 2304), "o": (1024, 8192), "cat": (8192, 4096), "unpooler": (4096, 1024)}[name]
        ldo = 8192 if name == "unpooler" else n_rows
        out = zbuf[:m] if name == "unpooler" else torch.empty((m, n_rows), dtype=torch.bfloat16, device=dev)
        xsrc = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
        x = torch.empty_like(xsrc)
        try:
            g0 = graph_of(lambda: (x.copy_(xsrc), p1s_call(mod, P[name], x, out, n_rows, ldo, geo, 0)))
            g1 = graph_of(lambda: (x.copy_(xsrc), p1s_call(mod, P[name], x, out, n_rows, ldo, geo, 1)))
            t0_, t1_, r_ = time_pair(g0, g1, flush, reps_final)
            print(f"[p1] PDL {name} M={m}: off {t0_:.2f} us | on {t1_:.2f} us | x{r_:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"[p1] PDL {name} M={m}: failed {ex!r}", flush=True)

    # ---------------------------------------------------------------- 4. table + gate
    print("\n===== P1 SUMMARY (event us incl. graph replay; net = minus the one-kernel floor "
          f"{floor_us:.2f} us; TB/s = packed MB / net us) =====", flush=True)
    print(f"| proj | T | today's path | today us | p1s BF16 us | p1s packed us | probe us | paired x | net x | "
          f"compression x | geometry (W,S,D) | regs | warps/SM | KB/SM in flight | waves | packed MB | achieved TB/s | "
          f"probe TB/s | gate x{GATE} |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    gate_ok = True
    geo_tab, route_tab = {}, {}
    for (name, m, label, t_a, t_r, t_b, ratio, net, c_ratio, geo, regs, wsm, infl, waves, mb, t_p) in rows:
        g = net >= GATE
        gate_ok &= g
        proj = {"cat": "pooler"}.get(name, name)
        geo_tab[(proj, m)] = geo
        route_tab[(proj, m)] = bool(net >= P1M.P1_ROUTE_MIN)
        print(f"| {proj} | {m} | {label} | {t_a:.2f} | {t_r:.2f} | {t_b:.2f} | {t_p:.2f} | "
              f"{ratio:.3f} | {net:.3f} | {c_ratio:.3f} | {geo} | {regs} | {wsm:.1f} | {infl / 1024:.0f} | "
              f"{waves:.2f} | {mb:.1f} | {mb / max(t_b - floor_us, 1e-3):.2f} | {mb / max(t_p - floor_us, 1e-3):.2f} | "
              f"{'PASS' if g else 'fail'} |")
    print("\n# paste into kernels/muse_bf16_proj/p1_module.py (route = net x >= "
          f"{P1M.P1_ROUTE_MIN} vs today's path, same process)")
    print("P1_GEOMETRY = {" + ", ".join(f"{k!r}: {v!r}" for k, v in sorted(geo_tab.items())) + "}")
    print("P1_ROUTE = {" + ", ".join(f"{k!r}: {v!r}" for k, v in sorted(route_tab.items())) + "}")
    n_shapes = len(rows)
    print(f"\n[p1] GATE (bit-exact unpack on all weights; packed >= x{GATE} net vs today on qkv, o, pooler, unpooler at "
          f"T in {P1_TS}): {'PASS' if (gate_ok and ok_all and n_shapes == 2 * len(shapes)) else 'FAIL'} "
          f"({sum(r[7] >= GATE for r in rows)}/{n_shapes} shapes pass); total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

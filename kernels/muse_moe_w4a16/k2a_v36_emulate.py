"""CPU emulation of K2A v3.6's new index math (k2a_v36.py, HK2 variants): routing (per-expert pass width), the two
balance modes (dual range / role split), gemm36's (part, segment, item, stage) walk shared by producer, consumers and
epilogue, the K1 X-row-to-lane map, the flush's (lane, chunk) -> (token, column) map for NQ chunks, split-item
counters, named-barrier ids and the smem budget. Every function mirrors the CUDA line by line (names kept).

    python kernels/muse_moe_w4a16/k2a_v36_emulate.py            # all checks + negative controls (exit 0 = PASS)
    python kernels/muse_moe_w4a16/k2a_v36_emulate.py --quick    # fewer routings

Standard library only (numpy is not needed; integer index math). Negative controls: each plants one bug (pass width
mismatch between route and gemm, an off-by-one range end, a missing role clamp, a wrong flush token map, a wrong X
lane map) and must be caught by a check; the run fails if any is NOT caught.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SW, XROW, KDIM, M3 = 18432, 528, 4096, 320
STATIC_SMEM_K1 = 7168          # s_list (5120 B) + Route + alignment: A2 build log, K1 smem 75184 - 68016 dynamic
SM_SMEM, CTA_RESERVED = 233472, 1024


# ------------------------------------------------------------------------------------------------ device mirrors
def route(counts, NC=1, TPH=32, HGE=True, hot_t=8):
    """k3::route<NC, TPH, HGE>: per expert cnt / off / np, stage prefixes sb (all), sbh (hot), sbc (cold), nrb = 1 unit
    (the caller scales by 16 * nrb)."""
    cnt, off, np_, hot = [0] * 32, [0] * 32, [0] * 32, [False] * 32
    o = 0
    for e in range(32):
        c = counts[e]
        h = (c > hot_t) if HGE else (hot_t > 0 and c > hot_t)
        cnt[e], off[e], hot[e] = c, o, h
        np_[e] = -(-c // TPH) if h else -(-c // (8 * NC))
        o += c
    return {"cnt": cnt, "off": off, "np": np_, "hot": hot, "R": o}


def prefixes(S, nrb):
    sb, sbh, sbc = [0] * 33, [0] * 33, [0] * 33
    for e in range(32):
        n = 16 * nrb * S["np"][e]
        sb[e + 1] = sb[e] + n
        sbh[e + 1] = sbh[e] + (n if S["hot"][e] else 0)
        sbc[e + 1] = sbc[e] + (0 if S["hot"][e] else n)
    return sb, sbh, sbc


def my_range2(Stot, cta, ncta):
    return Stot * cta // ncta, Stot * (cta + 1) // ncta


def ranges36(S, sbh, sbc, G, c, mode, hot_t, hot_w4, NCH, bug=None):
    Sh, Sc = sbh[32], sbc[32]
    if mode & 1:
        hb, he = my_range2(Sh, c, G)
        cb, ce = my_range2(Sc, c, G)
        if bug == "range_off_by_one":
            he = min(Sh, he + 1) if c % 7 == 3 else he
        cold_first = bool(mode & 2) and 2 * c >= G
        k0 = 0 if cold_first else 1
        b0, e0 = (cb, ce) if cold_first else (hb, he)
        b1, e1 = (hb, he) if cold_first else (cb, ce)
        return (b0, e0, k0, b1, e1, 1 - k0)
    Gh = 0
    if Sh > 0:
        Wh = Wc = 0
        for x in range(32):
            n = S["cnt"][x]
            if n == 0:
                continue
            if n > hot_t:
                tp = 8 * NCH
                full, rem = n // tp, n - (n // tp) * tp
                Wh += full * (16 + hot_w4 * (NCH - 1)) + ((16 + hot_w4 * (((rem + 7) >> 3) - 1)) if rem else 0)
            else:
                Wc += 16 * ((n + 7) >> 3)
        Gh = (Wh * G + (Wh + Wc) // 2) // (Wh + Wc) if Sc > 0 else G
        if bug != "no_role_clamp":
            Gh = max(1, min(Gh, G - 1 if Sc > 0 else G))
        else:
            Gh = G
    hr = c < Gh
    if hr:
        b0, e0 = my_range2(Sh, c, Gh)
    else:
        b0, e0 = my_range2(Sc, c - Gh, G - Gh) if G - Gh > 0 else (0, 0)
    return (b0, e0, 1 if hr else 0, 0, 0, 0)


def tail_first2(sb, beg, end):
    if end <= beg:
        return beg
    e = 0
    while sb[e + 1] <= end - 1:
        e += 1
    m = sb[e] + ((end - 1 - sb[e]) & ~15)
    return max(beg, m)


def segs36(sbh, sbc, parts):
    """k3::segs36: per part, [mid, end) (the tail-first item) then [begin, mid) -> four (begin, end, kind) segments."""
    b0, e0, k0, b1, e1, k1 = parts
    out = []
    for b, e, k in ((b0, e0, k0), (b1, e1, k1)):
        mid = tail_first2(sbh if k else sbc, b, e)
        out += [(mid, e, k), (b, mid, k)]
    return out


HO_BUDGET = 10


def walk(S, sbh, sbc, parts, NCH, bug=None, mode1=True):
    """gemm36's loop nest over segs36's four segments: one dict per item visit, in the order producer / consumers /
    epilogue see them."""
    out = []
    npart = 0
    for xb, xe, k in segs36(sbh, sbc, parts):
        if xe <= xb:
            continue
        hk = k != 0
        sb = sbh if hk else sbc
        TP = 8 * NCH if hk else 8
        if bug == "tp_mismatch" and hk:
            TP = 8 * (NCH // 2) if NCH > 2 else 4
        e = 0
        x = xb
        while x < xe:
            while sb[e + 1] <= x:
                e += 1
            np_ = S["np"][e]
            l = x - sb[e]
            item = l >> 4
            s0 = l & 15
            s1 = min(16, s0 + (xe - x))
            rb, j = item // np_, item - (item // np_) * np_
            ntok = min(TP, S["cnt"][e] - j * TP)
            c0 = S["off"][e] + j * TP
            nch = (ntok + 7) >> 3
            whole = s0 == 0 and s1 == 16
            # K1 hot items with >= 3 chunks: SwiGLU by the epilogue warp (NOSWI); whole ones take an id while npart < 10
            noswi = mode1 and hk and nch >= 3
            ho = noswi and npart < (HO_BUDGET if bug != "no_ho_budget" else 99)
            out.append({"hk": hk, "e": e, "rb": rb, "j": j, "s0": s0, "s1": s1, "ntok": ntok, "c0": c0, "nch": nch,
                        "whole": whole, "bar": 2 + npart if (not whole or ho) else None,
                        "exhausted": noswi and whole and not ho})
            if not whole or ho:
                npart += 1
            x += s1 - s0
    return out


def consumer_path(v, NCH):
    """which consumer body gemm36 dispatches to"""
    if v["nch"] <= 1:
        return "cold_item"
    if NCH >= 4:
        return "hc_item<%d>" % min(4, v["nch"]) if v["nch"] >= 4 else "hc_item<%d>" % v["nch"]
    return "hc_item<2>"


def flush_cover(NQ, ntok, bug=None):
    """flush_item<NQ>: (token n, row r of the 16-row tile) written per lane (g, t), chunk q, hh; returns a Counter."""
    seen = Counter()
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for q in range(NQ):
            for hh in range(2):
                n = q * 8 + 2 * t + hh if bug != "flush_map" else q * 8 + t + hh
                if n < ntok:
                    seen[(n, g)] += 1        # acc[q][hh]     -> row g   (K1 gate col 8 tile + g / K2 y col 16 tile + g)
                    seen[(n, g + 8)] += 1    # acc[q][2 + hh] -> row g+8 (K1 up col / K2 y col + 8)
    return seen


def xlane_cover(ntok, bug=None):
    """K1 producer: lane (r + 1) & 31 copies X row r (ntok <= 32)."""
    rows = Counter()
    for lane in range(32):
        r = (lane + 31) & 31 if bug != "xlane" else (lane + 31) & 15   # bug: rows 16-31 never copied
        if r < ntok:
            rows[r] += 1
    return rows


# ------------------------------------------------------------------------------------------------ checks
class Fail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def check_kernel(counts, nrb, G, NCH, NST, mode, hot_t, hot_w4, bug=None, stats=None):
    S = route(counts, 1, 8 * NCH, True, hot_t)
    sb, sbh, sbc = prefixes(S, nrb)
    STG = SW + 8 * NCH * XROW
    seen = Counter()
    contrib = {}
    tok_cover = {}
    max_bar, per_cta, exhausted = 0, [], 0
    for c in range(G):
        parts = ranges36(S, sbh, sbc, G, c, mode, hot_t, hot_w4, NCH, bug)
        b0, e0, k0, b1, e1, k1 = parts
        need(0 <= b0 <= e0 and 0 <= b1 <= e1, "range order")
        if not mode & 1:
            need(e1 == b1, "role split: one part per CTA")
        visits = walk(S, sbh, sbc, parts, NCH, bug, mode1=(nrb == 64))
        hs = cs = 0
        for v in visits:
            e = v["e"]
            need(S["hot"][e] == v["hk"], "item kind != expert kind")
            need(0 <= v["j"] < S["np"][e] and v["j"] < 64, "pass index out of range (cnt slot j < 64)")
            need(0 <= v["rb"] < nrb, "row block out of range")
            need(1 <= v["ntok"] <= (8 * NCH if v["hk"] else 8), "ntok out of range for the item kind")
            need(v["ntok"] * XROW <= STG - SW, "X rows overflow the slot")
            need(consumer_path(v, NCH) in ("cold_item", "hc_item<2>", "hc_item<3>", "hc_item<4>"), "dispatch")
            need(v["nch"] <= (NCH if v["hk"] else 1), "nch above the compiled chunk count")
            for s in range(v["s0"], v["s1"]):
                seen[(e, v["rb"], v["j"], s)] += 1
            key = (e, v["rb"], v["j"])
            contrib.setdefault(key, []).append(v["s1"] - v["s0"])
            tok_cover.setdefault((e, v["rb"]), set()).add((v["j"], v["c0"], v["ntok"]))
            if v["bar"] is not None:
                max_bar = max(max_bar, v["bar"])
            if v["exhausted"]:
                exhausted += 1
            if v["hk"]:
                hs += v["s1"] - v["s0"]
            else:
                cs += v["s1"] - v["s0"]
        per_cta.append((hs, cs))
    # every stage of every (e, rb, j) exactly once
    for e in range(32):
        for rb in range(nrb):
            for j in range(S["np"][e]):
                for s in range(16):
                    need(seen[(e, rb, j, s)] == 1, f"stage (e{e} rb{rb} j{j} s{s}) covered {seen[(e, rb, j, s)]}x")
    need(sum(seen.values()) == sb[32], "extra stages visited")
    # split items: contributions sum to 16 (the counter reaches 16 exactly once, the last contributor finalizes)
    for key, v in contrib.items():
        need(sum(v) == 16, f"item {key} contributions {v}")
    # passes of a row block partition the expert's compact rows [off, off + cnt)
    for (e, rb), ps in tok_cover.items():
        rows = []
        for j, c0, n in sorted(ps):
            rows += list(range(c0, c0 + n))
        need(rows == list(range(S["off"][e], S["off"][e] + S["cnt"][e])), f"token partition e{e} rb{rb}")
    need(max_bar <= 15, f"named barrier id {max_bar} > 15")
    if sum(counts) <= 8 * 48:   # HK2 is only routed for M <= 48 (v36_call refuses more)
        need(exhausted == 0, f"{exhausted} whole K1 hot items past the hand-off budget (error flag path)")
    if stats is not None:
        stats["exhausted_big"] = stats.get("exhausted_big", 0) + (exhausted if sum(counts) > 8 * 48 else 0)
    if mode & 1:   # dual range: every CTA within one stage of every other, per kind
        hsv, csv = [h for h, _ in per_cta], [c for _, c in per_cta]
        need(max(hsv) - min(hsv) <= 1 and max(csv) - min(csv) <= 1, "dual range not balanced to one stage")
    if stats is not None:
        stats["max_bar"] = max(stats.get("max_bar", 0), max_bar)
        stats["items"] = stats.get("items", 0) + len(contrib)
        stats["split"] = stats.get("split", 0) + sum(1 for v in contrib.values() if len(v) > 1)
    return S


def check_flush_maps(bug=None):
    for NQ in (1, 2, 3, 4):
        for ntok in range(8 * (NQ - 1) + 1, 8 * NQ + 1):
            seen = flush_cover(NQ, ntok, bug)
            for n in range(ntok):
                for r in range(16):
                    need(seen[(n, r)] == 1, f"flush NQ{NQ} ntok{ntok}: (n{n}, row{r}) written {seen[(n, r)]}x")
            need(sum(seen.values()) == 16 * ntok, "flush writes outside the pass")
    for ntok in range(1, 33):
        rows = xlane_cover(ntok, bug)
        need(all(rows[r] == 1 for r in range(ntok)) and sum(rows.values()) == ntok, f"X lane map ntok {ntok}")


def check_smem():
    out = []
    for NCH, NST in ((4, 3), (2, 3), (2, 4), (4, 4)):
        dyn = NST * (SW + 8 * NCH * XROW) + 2 * NST * 8
        per_cta = dyn + STATIC_SMEM_K1 + CTA_RESERVED
        out.append((NCH, NST, dyn, per_cta, 2 * per_cta <= SM_SMEM))
    return out


# ------------------------------------------------------------------------------------------------ routings
def lab_routings():
    p = ROOT / "box_out/a2_lab/summary.json"
    if not p.exists():
        return []
    s = json.loads(p.read_text())
    return [(f"a2 bs{r['bs']} {r['src']} {r['seed']}", list(r["counts"])) for r in s["routings"]]


def synth_counts(bs, rnd, kind):
    counts = [0] * 32
    w = [rnd.paretovariate(1.2) for _ in range(256)] if kind == "skew" else [1.0 + 0.1 * rnd.random() for _ in range(256)]
    for _ in range(bs):
        if kind == "hot8":
            picks = list(range(8))
        else:
            picks = set()
            while len(picks) < 8:
                picks.add(rnd.choices(range(256), weights=w)[0])
        for e in picks:
            if e < 32:
                counts[e] += 1
    return counts


def all_routings(quick):
    rnd = random.Random(36)
    rs = lab_routings()
    for bs in ((1, 8, 12, 16, 20, 24, 32, 33, 48) if quick else (1, 2, 5, 8, 9, 12, 16, 17, 20, 24, 31, 32, 33, 40, 48, 64)):
        for kind in ("skew", "flat", "hot8"):
            rs.append((f"synth bs{bs} {kind}", synth_counts(bs, rnd, kind)))
    for bs in ((320,) if quick else (128, 256, 320)):   # beyond HK2's M <= 48: the barrier budget must still hold
        rs.append((f"synth bs{bs} hot8", synth_counts(bs, rnd, "hot8")))
    rs.append(("one expert x32", [32] + [0] * 31))
    rs.append(("empty", [0] * 32))
    return rs


# ------------------------------------------------------------------------------------------------ driver
def run_checks(quick=False, bug=None, log=print):
    routings = all_routings(quick)
    configs = []
    for NCH, NST in ((4, 3), (2, 3)):
        for mode in (3, 1, 0):
            for hot_t, hot_w4 in ((8, 8), (0, 8), (16, 1), (8, 16)):
                configs.append((NCH, NST, mode, hot_t, hot_w4))
    if quick:
        configs = [c for c in configs if c[3] == 8 or c[2] == 3]
    grids = (264, 263, 132)
    stats = {}
    n = 0
    for name, counts in routings:
        for (NCH, NST, mode, hot_t, hot_w4) in configs:
            for G in (grids if name.startswith("a2") else grids[:1]):
                for nrb in (64, 32):
                    check_kernel(counts, nrb, G, NCH, NST, mode, hot_t, hot_w4, bug, stats)
                    n += 1
    check_flush_maps(bug)
    return n, stats, len(routings)


def model_print(log=print):
    """Gh and hot share at the lab's bs16/32 routings (role split, hot_w4 8) - the bench prints the measured side."""
    for name, counts in lab_routings():
        if " bs16 " not in name + " " and " bs32 " not in name + " ":
            continue
        S = route(counts, 1, 32, True, 8)
        _, sbh, sbc = prefixes(S, 64)
        parts = [ranges36(S, sbh, sbc, 264, c, 0, 8, 8, 4) for c in range(264)]
        gh = sum(1 for p in parts if p[2] == 1)
        hot = sorted([c for c in counts if c > 8], reverse=True)
        log(f"  {name:22s} hot {str(hot):14s} K1 stages hot/cold {sbh[32]:5d}/{sbc[32]:5d} (NC1 {prefixes(route(counts, 1, 8, True, 999), 64)[0][32]:5d}); role split Gh {gh}")


def main():
    quick = "--quick" in sys.argv
    n, stats, nr = run_checks(quick)
    print(f"[emu] positive: {n} kernel walks over {nr} routings x configs (G 264/263/132, K1+K2, NCH 2/4, modes 3/1/0, "
          f"hot_t 0/8/16): every stage once, pass partition, split counters, barrier ids <= {stats['max_bar']} (<= 15), "
          f"dual balance <= 1 stage; K1 hand-off budget never exhausted at M <= 48 (at M > 48: {stats.get('exhausted_big', 0)} "
          f"items, HK2 not routed there); items {stats['items']} of which split {stats['split']} -> PASS")
    print("[emu] flush (lane, chunk) -> (token, row) maps for NQ 1-4 and X row -> lane map for ntok 1-32 -> PASS")
    for NCH, NST, dyn, per, ok in check_smem():
        print(f"[emu] smem NCH{NCH} NST{NST}: dynamic {dyn} B, per CTA with static + reserve {per} B -> 2 CTAs/SM "
              f"{'fit' if ok else 'DO NOT fit'} ({2 * per} of {SM_SMEM})")
    need(check_smem()[0][4], "the shipped HK2 variant (NCH4 NST3) must fit 2 CTAs/SM")
    caught = {}
    for bug in ("tp_mismatch", "range_off_by_one", "no_role_clamp", "flush_map", "xlane", "no_ho_budget"):
        try:
            run_checks(True, bug)
            caught[bug] = False
        except Fail as ex:
            caught[bug] = str(ex)[:90]
    for bug, msg in caught.items():
        print(f"[emu] negative control {bug:16s}: {'caught: ' + msg if msg else 'NOT CAUGHT'}")
    print("[emu] model (lab bs16/32 routings):")
    model_print()
    ok = all(caught.values())
    print("[emu] RESULT:", "PASS" if ok else "FAIL (a negative control was not caught)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except Fail as ex:
        print("[emu] RESULT: FAIL:", ex)
        sys.exit(1)

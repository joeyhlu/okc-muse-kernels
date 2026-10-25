"""K1 v9c (C1) CPU emulation: the cluster-shaped split plan and the DSMEM cluster merge of k1_v9c.py.

  python3 -I k1_c1_emulate.py            # part A (numpy only) + part B when torch is importable
  python3 -I k1_c1_emulate.py --no-e2e   # part A only

Part A (numpy, no torch, seconds):
  0. Source anchors: every formula mirrored below is asserted to appear verbatim in k1_v9c.py's CUDA_SRC / host code, so
     the emulator fails loudly if the kernel drifts from it.
  1. Plan: cluster_plan() (device: request b = cluster c / CS, split = rank, C_b = max(min_chunk, ceil(nt/CS)),
     S_b = ceil(nt/C_b), balanced boundaries) on 3000 random cases: every tile once, one last split per request,
     S_b <= CS, splits <= C_b tiles and non-empty. Host pick (pick_cs) on the scored shapes, and the cluster plan
     against the v4 plan (plan_v4 = k1_emulate.plan) on dec1/16/32 sliding (agentic lengths): identical splits.
  2. Merge index math, thread by thread (rank rk < S_b, tid < 256, item k < KC): every (row, quad) of the 16L rows merged
     exactly once; reads only from ranks s < S_b (never an idle rank), only inside the peer's ost / lse_s regions,
     16-B aligned; items only in warps whose warp-uniform predicate is set, the predicate uniform per warp, the 16
     quads of a row in one half-warp; gate / output addresses equal the v4 merge's.
  3. Operand order: per output (row, quad) the cluster merge feeds merge_finalize's per-item code the same (lse, O)
     values in the same split order as the v4 fold merge reading global partials -> same instruction sequence per
     item -> bit-identical outputs on the GPU whenever the plans agree (all scored sliding shapes, part 1).
  4. Numerics on dec1/dec16/dec32 sliding (B 1/16/32, 2048-key window, soft keys): per-split partials (O / l, log2
     lse, float32) -> cluster merge with the kernel's float32 op order -> vs float64 attention over all keys, before
     and after the norm-gate epilogue; plus -inf lse rows. Bound: rel L2 <= 1e-5 before the epilogue, bf16 after.
  5. Negative controls (each must FAIL a check): rows split by CS instead of S_b; peer row off by one; lse ignored;
     lse read from the own CTA (no mapa); per-lane instead of warp-uniform predicate; S_b = CS regardless of nt.
Part B (torch, the pod CPU): k1_emulate.emulate_k1 end to end with its plan and fold merge swapped for the cluster
  versions (tile pipeline unchanged) on dec1/16/32 sliding inputs from k1_emulate.generate_input (the judge's
  generator copy): judge rule, exact key coverage, and equality with the v4-plan fold output.
"""
import ast
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KFILE = HERE / "k1_v9c.py"

TILE, NWARP, NTHR, ROWB, P_OFFSET = 64, 8, 256, 128, 8.0
LOG2E = 1.4426950408889634
N_SM = 132
CL_MAX = 16
WINDOW = 2048
MIN_CHUNK = {("sliding", 1): 2, ("sliding", 8): 4, ("full", 1): 2, ("full", 8): 2}   # k1_v9c K1_MIN_CHUNK_T
CPS = {("sliding", 1): 1, ("sliding", 8): 2, ("full", 1): 2, ("full", 8): 2}         # k1_v9c K1_CPS_T
OCC = {1: 2, 8: 2}                                                                      # k1_main occupancy (box)


# ------------------------------------------------------------------------------------------------
# 0. Source anchors
# ------------------------------------------------------------------------------------------------
ANCHORS = [
    # cluster plan
    "const int CS = p.cs, bq = c / CS, rk = c - bq * CS;",
    "const int nt = old_tiles(__ldg(p.seq_lens + bq), p.W);",
    "const int C = max(p.min_chunk, (nt + CS - 1) / CS);",
    "plan_s[0] = shape_ok ? bq : -1; plan_s[1] = rk; plan_s[2] = split_count(nt, C); plan_s[3] = nt;",
    "const int t0 = (int)(((long long)s_idx * nt_b) / S_b);",
    "const int ntl = (int)(((long long)(s_idx + 1) * nt_b) / S_b) - t0;",
    "const bool last = (s_idx == S_b - 1);",
    "if (p.cs && s_idx >= S_b) {",
    # merge rows / items / predicates / addresses
    "template <int L> constexpr int c1_kc() { return (8 * L + 15) / 16; }",
    "__device__ __forceinline__ int c1_rlo(int rk, int S, int rows) { return (rk * rows) / S; }",
    "const int r_lo = c1_rlo(rk, S, ROWS), r_hi = c1_rlo(rk + 1, S, ROWS);",
    "const int r0 = threadIdx.x >> 4, u = threadIdx.x & 15, wr = 2 * (threadIdx.x >> 5);",
    "const int r = r_lo + r0 + 16 * k;",
    "const bool mine = r < r_hi;",
    "const bool wk = r_lo + wr + 16 * k < r_hi;",
    "if (mine && s < S) {",
    "lv[q] = cl_ld(cl_map(l_sa + 4u * (unsigned)r, (unsigned)s));",
    "ov[q] = cl_ld4(cl_map(o_sa + 4u * (unsigned)(r * 64 + 4 * u), (unsigned)s));",
    "constexpr int UN = (L == 1) ? 16 : 8;              // merge_finalize's batch",
    "if (k == KC - 1) cl_arrive_sel(p.c1_flags & 2);",
    # merge math (shared with merge_finalize)
    "const float sc = ex2(M - Mu);",
    "const float w = ex2(lv[q] - Mu);",
    "const float scl = (ws > 0.f) ? vsc / ws : 0.f;",
    "const int qh = (2 * hp + hh) * 8 + gi;",
    # smem layout of the partials (L=1: after ow / mw / lw in the ring; L=8: ring start)
    "float* mw = ow + NWARP * 8 * 64;              // [8][8]",
    "float* lw = mw + NWARP * 8;                   // [8][8]",
    "ost = lw + NWARP * 8;                         // [16][64]",
    "lse_s = ost + ROWS * 64;                      // [16]",
    "ost = reinterpret_cast<float*>(ring);            // [128][64] (ring is drained; newk/vh are separate)",
    # protocol: two arrive/wait pairs on every path
    "if (S_b > 1) {\n      merge_cluster<L>(p, ost, lse_s, S_b, rk, b, hp, c1g, c1vsc);",
]
HOST_ANCHORS = [
    "cap = min(K1_CLUSTER_MAX, (_sms(dev) * cps) // max(1, B * hp))",
    "nt_max = -(-(cfg[\"window\"] - 1) // K1_TILE)",
    "cap = min(cap, -(-nt_max // min_chunk))",
    "gx = B * ccs",
    "K1_CLUSTER_T = {(\"sliding\", 1): True}",
]


def source_anchor_test():
    text = KFILE.read_text()
    tree = ast.parse(text)
    cuda = next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CUDA_SRC")
    miss = [a for a in ANCHORS if a not in cuda] + [a for a in HOST_ANCHORS if a not in text]
    assert not miss, "k1_v9c.py drifted from the emulator; missing:\n  " + "\n  ".join(miss)
    # merge_cluster's per-item math must be merge_finalize's (same lines, modulo the [kg] / per-item names)
    def body(name):
        i = cuda.index(name)
        return cuda[i:cuda.index("\n}\n", i)]
    fin = body("__device__ __forceinline__ void merge_finalize(")
    clu = body("__device__ __forceinline__ void merge_cluster(")
    norm = lambda s: " ".join(s.replace("[kg]", "").replace("acc[kg]", "acc").split())   # noqa: E731
    for line in ("bm = fmaxf(bm, lv[q]);", "const float Mu = (bm == neg_inf()) ? 0.f : bm;",
                 "acc.x *= sc; acc.y *= sc; acc.z *= sc; acc.w *= sc;", "ws *= sc;", "ws += w;",
                 "acc.x = fmaf(w, ov[q].x, acc.x); acc.y = fmaf(w, ov[q].y, acc.y);",
                 "acc.z = fmaf(w, ov[q].z, acc.z); acc.w = fmaf(w, ov[q].w, acc.w);", "M = bm;",
                 "float ss = a4.x * a4.x + a4.y * a4.y + a4.z * a4.z + a4.w * a4.w;",
                 "const float rs = rsqrtf(ss * (1.f / 64.f) + 1e-5f);"):
        assert line in norm(fin) and line in norm(clu), f"merge math differs: {line}"
    return len(ANCHORS) + len(HOST_ANCHORS)


# ------------------------------------------------------------------------------------------------
# 1. Plans
# ------------------------------------------------------------------------------------------------
def cdiv(a, b):
    return -(-a // b)


def old_tiles(seq, W):
    n_old = min(seq, W - 1) if W else seq
    return cdiv(max(n_old, 0), TILE)


def split_count(nt, C):
    return max(1, cdiv(nt, C))


def cluster_plan(seq_lens, W, CS, min_chunk):
    """device plan of the cluster path -> per request (S_b, C_b, nt_b) and per CTA c in [0, B*CS) its work."""
    reqs = []
    for seq in seq_lens:
        nt = old_tiles(seq, W)
        C = max(min_chunk, cdiv(nt, CS))
        reqs.append((split_count(nt, C), C, nt))
    return reqs


def cta_work_cluster(c, CS, reqs):
    b, rk = c // CS, c % CS
    S, C, nt = reqs[b]
    if rk >= S:
        return None                                   # idle rank
    t0 = (rk * nt) // S
    t1 = ((rk + 1) * nt) // S
    return b, rk, t0, t1, rk == S - 1


def plan_v4(seq_lens, W, Gx, min_chunk):
    """k1_emulate.plan (the v4 device plan_C / plan_find): (nt, C, S, off)."""
    B = len(seq_lens)
    nt = [old_tiles(s, W) for s in seq_lens]
    T = sum(nt)
    lo = max(min_chunk, cdiv(T, Gx))
    hi = max(lo, cdiv(T, max(1, Gx - B)))
    while lo < hi:
        mid = (lo + hi) >> 1
        if sum(split_count(n, mid) for n in nt) <= Gx:
            hi = mid
        else:
            lo = mid + 1
    S = [split_count(n, lo) for n in nt]
    off = list(np.cumsum([0] + S[:-1]))
    return nt, lo, S, off


def pick_cs(kind, L, B, hp=1, n_sm=N_SM):
    """k1_v9c._pick_cluster without the occupancy query: min(16, SMs*cps/(B*hp), useful splits of a window), >= 2."""
    cps = max(1, min(CPS[(kind, L)], OCC[L]))
    cap = min(CL_MAX, (n_sm * cps) // max(1, B * hp))
    if kind == "sliding":
        cap = min(cap, cdiv(cdiv(WINDOW - 1, TILE), MIN_CHUNK[(kind, L)]))
    return cap if cap >= 2 else 0


def agentic_lengths(B, ctx, seed):
    """reference._lengths(spread='agentic') in numpy: ctx * (1 + 0.075 (2u - 1)), u uniform."""
    if B == 1:
        return [ctx]
    u = np.random.default_rng(seed).random(B)
    return [int(x) for x in (ctx * (1 + 0.075 * (2 * u - 1))).astype(np.int64)]


def plan_test(n_cases=3000, seed=1, plan=cluster_plan):
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        B = int(rng.integers(1, 70))
        W = WINDOW if rng.random() < 0.5 else 0
        CS = int(rng.integers(2, 17))
        mc = int((1, 2, 4)[rng.integers(0, 3)])
        seq = rng.integers(0, 300000, B)
        if rng.random() < 0.4:
            seq = seq % 300
        reqs = plan(list(map(int, seq)), W, CS, mc)
        cover = [np.zeros(r[2], np.int64) for r in reqs]
        lasts = np.zeros(B, np.int64)
        for c in range(B * CS):
            wk = cta_work_cluster(c, CS, reqs)
            if wk is None:
                continue
            b, rk, t0, t1, last = wk
            S, C, nt = reqs[b]
            assert 1 <= S <= CS, "S_b outside [1, CS]"
            assert t1 - t0 <= C, "split longer than C_b"
            assert t1 > t0 or nt == 0, "empty split for a request with tiles"
            cover[b][t0:t1] += 1
            lasts[b] += int(last)
        assert all((cv == 1).all() for cv in cover), "tile coverage"
        assert (lasts == 1).all(), "one last split per request"
    return n_cases


def scored_plan_test():
    rows = []
    for B, seed in ((1, 601), (16, 603), (32, 605)):
        seq = agentic_lengths(B, 80000, seed)
        cs = pick_cs("sliding", 1, B)
        reqs = cluster_plan(seq, WINDOW, cs, MIN_CHUNK[("sliding", 1)])
        gx_v4 = max(max(1, N_SM * CPS[("sliding", 1)]), B + 1)
        nt, C, S, off = plan_v4(seq, WINDOW, gx_v4, MIN_CHUNK[("sliding", 1)])
        for b in range(B):
            Sc, Cc, ntc = reqs[b]
            assert ntc == nt[b] and Sc == S[b], f"B={B} b={b}: cluster S={Sc} vs v4 S={S[b]}"
            for s in range(Sc):
                assert ((s * ntc) // Sc, ((s + 1) * ntc) // Sc) == ((s * nt[b]) // S[b], ((s + 1) * nt[b]) // S[b])
        assert B * cs <= N_SM, "padded clusters need B*CS <= SMs for one wave"
        rows.append(f"dec{B}: CS={cs} grid={B * cs} (v4 grid {gx_v4}, busy {sum(S)}) S_b={S[0]} tiles/split="
                    f"{nt[0] // S[0]}")
    return rows


# ------------------------------------------------------------------------------------------------
# 2./3. Merge index math and operand order (thread level)
# ------------------------------------------------------------------------------------------------
def smem_offsets(L):
    """(ost, lse_s) byte offsets in the CTA's dynamic smem, from the kernel's layout (ring at offset 0)."""
    if L == 1:
        ow = 0
        mw = ow + NWARP * 8 * 64 * 4
        lw = mw + NWARP * 8 * 4
        ost = lw + NWARP * 8 * 4
    else:
        ost = 0
    return ost, ost + 16 * L * 64 * 4


def merge_items(L, S, rk, uniform_pred=True, split_by=None):
    """Thread-level enumeration of rank rk's merge items: yields (tid, k, r, u, mine, wk)."""
    ROWS, KC = 16 * L, (8 * L + 15) // 16
    div = S if split_by is None else split_by
    r_lo, r_hi = (rk * ROWS) // div, ((rk + 1) * ROWS) // div
    for tid in range(NTHR):
        r0, u, wr = tid >> 4, tid & 15, 2 * (tid >> 5)
        for k in range(KC):
            r = r_lo + r0 + 16 * k
            mine = r < r_hi
            wk = (r_lo + wr + 16 * k < r_hi) if uniform_pred else mine
            yield tid, k, r, u, mine, wk


def index_test(L, S, CS, uniform_pred=True, split_by=None, peer_shift=0):
    """All checks of section 2 for one (L, S_b, CS); returns the operand map {(row, quad): [(rank, byte offsets)]}."""
    ROWS = 16 * L
    ost, lse = smem_offsets(L)
    ring_bytes = 4 * 2 * TILE * ROWB                    # ST = 4 stages
    assert lse + 4 * ROWS <= ring_bytes, "partials outside the drained ring"
    assert ost % 16 == 0, "ost not 16-B aligned"
    got = {}
    for rk in range(S):
        warp_wk = {}
        rows_of_halfwarp = {}
        for tid, k, r, u, mine, wk in merge_items(L, S, rk, uniform_pred, split_by):
            warp_wk.setdefault((tid >> 5, k), set()).add(wk)
            if wk:
                rows_of_halfwarp.setdefault((tid >> 4, k), set()).add(r)
            if not mine:
                continue
            assert wk, "item in a skipped warp"
            ops = []
            for s in range(S):
                peer = (s + peer_shift) % S
                assert 0 <= peer < S < CS + 1, "read from an idle / foreign rank"
                lo = lse + 4 * r
                oo = ost + 4 * (r * 64 + 4 * u)
                assert lse <= lo < lse + 4 * ROWS and ost <= oo < ost + ROWS * 256 and oo % 16 == 0, "DSMEM address"
                ops.append((peer, lo, oo))
            key = (r, u)
            assert key not in got, f"row {r} quad {u} merged twice"
            got[key] = ops
        assert all(len(v) == 1 for v in warp_wk.values()), "warp-uniform predicate not uniform"
        assert all(len(v) == 1 for v in rows_of_halfwarp.values()), "a half-warp spans two rows"
    assert set(got) == {(r, u) for r in range(ROWS) for u in range(16)}, "rows / quads not covered exactly once"
    return got


def fold_operands(L, S):
    """v4 merge_finalize reading global partials po[s][r][64], pl[s][r]: operand map in the same form."""
    ROWS = 16 * L
    ost, lse = smem_offsets(L)
    got = {}
    for tid in range(NTHR):
        r0, u = tid >> 4, tid & 15
        for k in range(ROWS * 16 // NTHR):
            r = r0 + 16 * k
            got[(r, u)] = [(s, lse + 4 * r, ost + 4 * (r * 64 + 4 * u)) for s in range(S)]
    return got


def gate_out_addr_test(L, Hq=16, Hkv=2):
    """gate column / output offset of row r: the cluster merge's (c1_gates / epilogue) == merge_finalize's."""
    R8 = 8 * L
    stride = (2 * Hq + 2 * Hkv) * 64
    for hp in range(Hkv // 2):
        for b in range(3):
            for r in range(16 * L):
                hh, rr = r // R8, r % R8
                j, gi = rr >> 3, rr & 7
                g_cl = (b * L + j) * stride + (Hq + 2 * Hkv) * 64 + ((2 * hp + hh) * 8 + gi) * 64
                qh = (2 * hp + hh) * 8 + gi
                o_cl = (b * L + j) * Hq * 64 + qh * 64
                # merge_finalize: graw at (b*L + j)*stride + (Hq + 2Hkv)*64 + ((2hp+hh)*8+gi)*64; out tok*Hq*64 + qh*64
                g_v4 = (b * L + j) * stride + (Hq + 2 * Hkv) * 64 + ((2 * hp + hh) * 8 + gi) * 64
                assert g_cl == g_v4 and o_cl == (b * L + j) * Hq * 64 + qh * 64
                assert 0 <= qh < Hq and 0 <= j < L


# ------------------------------------------------------------------------------------------------
# 4. Numerics
# ------------------------------------------------------------------------------------------------
f32 = np.float32


def fmaf(a, b, c):
    """fp32 fma: exact product (48 bits) + c in float64, one rounding to fp32 (exact unless the float64 sum rounds,
    which needs exponent gaps > 29 bits: irrelevant here)."""
    return f32(np.float64(a) * np.float64(b) + np.float64(c))


def ex2(x):
    return f32(np.exp2(f32(x)))


def merge_item(lv_all, ov_all, S, UN, vsc):
    """merge_finalize / merge_cluster per item, fp32 op order: lv_all [S], ov_all [S, 4] -> (acc[4] * vsc/ws, ws)."""
    NEG = f32(-np.inf)
    M, ws = NEG, f32(0)
    acc = np.zeros(4, f32)
    for s0 in range(0, S, UN):
        lv = [lv_all[s0 + q] if s0 + q < S else NEG for q in range(UN)]
        ov = [ov_all[s0 + q] if s0 + q < S else np.zeros(4, f32) for q in range(UN)]
        bm = M
        for q in range(UN):
            bm = f32(max(bm, lv[q]))
        Mu = f32(0) if bm == NEG else bm
        sc = ex2(f32(M - Mu))
        acc = (acc * sc).astype(f32)
        ws = f32(ws * sc)
        for q in range(UN):
            w = ex2(f32(lv[q] - Mu))
            ws = f32(ws + w)
            acc = np.array([fmaf(w, ov[q][i], acc[i]) for i in range(4)], f32)
        M = bm
    scl = f32(vsc / ws) if ws > 0 else f32(0)
    return (acc * scl).astype(f32), ws


def epilogue(a_rows, gate_rows):
    """rows [n, 64] fp32 (already * vsc / ws): half-warp RMS (squares per quad, xor 8/4/2/1) * sigmoid(gate) -> bf16."""
    out = np.empty_like(a_rows)
    for i in range(a_rows.shape[0]):
        a4 = a_rows[i].reshape(16, 4)
        ss = (a4 * a4).sum(1).astype(f32)
        for d in (8, 4, 2, 1):                        # butterfly over the 16 quads (same total in every lane)
            ss = (ss + ss[np.arange(16) ^ d]).astype(f32)
        rs = f32(1.0 / math.sqrt(float(ss[0]) / 64.0 + 1e-5))
        out[i] = a_rows[i] * rs / (1 + np.exp(-gate_rows[i].astype(np.float64)))
    return to_bf16(out)


def to_bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16          # RNE
    return (u.astype(np.uint32) << 16).view(np.float32)


def make_request(rng, L, seq, W, n_keys_new):
    """scores in log2 units [16L rows, keys] and values [keys, 2 heads, 64] for one (request, head pair): soft keys
    (shared unit direction per head, c2 in the sliding SOFT_C2 range), unit-RMS normalized queries x 5.473 x 1/8."""
    nk_old = min(seq, W - 1) if W else seq
    nk = nk_old + n_keys_new
    u = rng.standard_normal((2, 64))
    u /= np.sqrt((u * u).mean(1, keepdims=True))
    c2 = rng.uniform(0.3, 0.75, (2, 1))
    z = rng.standard_normal((nk, 2, 64))
    k = np.sqrt(c2) * u + np.sqrt(1 - c2) * z
    q = rng.standard_normal((2, 8 * L, 64)) + 0.6 * u[:, None, :]
    q /= np.sqrt((q * q).mean(-1, keepdims=True))
    v = rng.standard_normal((nk, 2, 64))
    cl2 = 5.4730064863875 * 0.125 * LOG2E
    s2 = np.einsum("hrd,khd->hrk", q, k) * cl2      # [2, 8L, nk]
    # masks (k1_main: ak > seq + j - W and ak <= seq + j): token j drops the j oldest window keys (when the window is
    # full) and sees the new keys j' <= j
    j = np.arange(8 * L) >> 3
    kk = np.arange(nk)
    vis = np.where(kk[None, :] < nk_old, kk[None, :] >= (j[:, None] if (W and seq >= W - 1) else 0),
                   (kk[None, :] - nk_old) <= j[:, None])
    s2 = np.where(vis[None], s2, -np.inf)
    return s2.reshape(16 * L, nk), v, nk_old


def partials(s2, v, L, splits, keys_of):
    """per-split normalized O and log2 lse as k1_main stores them (fp32): O_s = sum p v / sum p, lse_s = log2 sum 2^s2."""
    R8 = 8 * L
    po, pl = [], []
    heads = np.arange(16 * L) // R8
    for s in range(splits):
        ks = keys_of(s)
        x = s2[:, ks]
        m = x.max(1, keepdims=True)
        dead = ~np.isfinite(m[:, 0])                  # every key of the split masked for this row
        p = np.exp2(x - np.where(np.isfinite(m), m, 0.0))
        lt = p.sum(1)
        o = np.einsum("rk,rkd->rd", p, v[ks][:, heads, :].transpose(1, 0, 2)) / np.where(dead, 1.0, lt)[:, None]
        po.append(np.where(dead[:, None], 0.0, o).astype(f32))          # kernel: ost = 0, lse = -inf when lt == 0
        with np.errstate(divide="ignore"):
            pl.append(np.where(dead, -np.inf, m[:, 0] + np.log2(np.where(dead, 1.0, lt))).astype(f32))
    return np.stack(po), np.stack(pl)


def ref_attention(s2, v, L):
    R8 = 8 * L
    heads = np.arange(16 * L) // R8
    m = s2.max(1, keepdims=True)
    p = np.exp2(s2 - m)
    return np.einsum("rk,rkd->rd", p, v[:, heads, :].transpose(1, 0, 2)) / p.sum(1)[:, None]


def cluster_merge_request(po, pl, L, S, vsc, ops_map, lse_from_own=False, ignore_lse=False, row_shift=0):
    """the cluster merge of one (request, head pair) driven by the operand map of index_test (peer, byte offsets):
    returns merged rows [16L, 64] (before the epilogue, already x vsc / ws)."""
    ost, lse = smem_offsets(L)
    UN = 16 if L == 1 else 8
    ROWS = 16 * L
    out = np.zeros((ROWS, 64), f32)
    owner_rank = {}
    for rk in range(S):
        r_lo, r_hi = (rk * ROWS) // S, ((rk + 1) * ROWS) // S
        for r in range(r_lo, r_hi):
            owner_rank[r] = rk
    for (r, u), ops in ops_map.items():
        lv = np.empty(S, f32)
        ov = np.empty((S, 4), f32)
        for i, (peer, lo, oo) in enumerate(ops):
            rr = (lo - lse) // 4
            rq = (oo - ost) // 4
            src = owner_rank[r] if lse_from_own else peer
            lv[i] = f32(0) if ignore_lse else pl[src, rr]
            ro, d0 = divmod(rq, 64)
            ro = (ro + row_shift) % ROWS
            ov[i] = po[peer, ro, d0:d0 + 4]
        a, _ = merge_item(lv, ov, S, UN, vsc)
        out[r, 4 * u:4 * u + 4] = a
    return out


def numerics_case(B, L, seed, CS=None, ctl=None):
    """dec{B} sliding: every request through the cluster plan and merge; returns (rel err pre-epilogue, post)."""
    rng = np.random.default_rng(seed)
    seqs = agentic_lengths(B, 80000, seed)
    cs = CS or pick_cs("sliding", L, B)
    reqs = cluster_plan(seqs, WINDOW, cs, MIN_CHUNK[("sliding", L)])
    if ctl == "S_eq_CS":
        reqs = [(cs, C, nt) for (S, C, nt) in reqs]
    num = den = num_e = den_e = 0.0
    for b in range(B):
        S, C, nt = reqs[b]
        s2, v, nk_old = make_request(rng, L, seqs[b], WINDOW, L)
        tiles_key = lambda t: np.arange(t * TILE, min((t + 1) * TILE, nk_old))   # noqa: E731

        def keys_of(s, S=S, nt=nt):
            t0, t1 = (s * nt) // S, ((s + 1) * nt) // S
            ks = [tiles_key(t) for t in range(t0, t1)]
            if s == S - 1:
                ks.append(np.arange(nk_old, nk_old + L))
            return np.concatenate(ks) if ks else np.zeros(0, np.int64)
        for s in range(S):                            # cover every key once (the plan part of the emulation)
            assert len(keys_of(s)) > 0, "a split with no keys"
        assert sorted(np.concatenate([keys_of(s) for s in range(S)]).tolist()) == list(range(nk_old + L)), "keys"
        po, pl = partials(s2, v, L, S, keys_of)
        kw = dict(uniform_pred=(ctl != "lane_pred"), split_by=(cs if ctl == "rows_by_CS" else None))
        ops = index_test(L, S, cs, **kw)
        got = cluster_merge_request(po, pl, L, S, f32(1.0), ops, lse_from_own=(ctl == "own_lse"),
                                    ignore_lse=(ctl == "no_lse"), row_shift=(1 if ctl == "row_shift" else 0))
        ref = ref_attention(s2, v, L)
        num += float(((got - ref) ** 2).sum())
        den += float((ref ** 2).sum())
        gate = rng.standard_normal((16 * L, 64)).astype(f32)
        e_got = epilogue(got, gate)
        rr = ref / np.sqrt((ref ** 2).mean(1, keepdims=True) + 1e-5) / (1 + np.exp(-gate.astype(np.float64)))
        num_e += float(((e_got - rr) ** 2).sum())
        den_e += float((rr ** 2).sum())
    return math.sqrt(num / den), math.sqrt(num_e / den_e), cs, reqs[0][0]


def neg_inf_rows_test():
    """rows whose lse is -inf in some / all splits (M = -inf path): finite, and equal to the v4 fold merge op for op."""
    rng = np.random.default_rng(5)
    for S in (2, 5, 16):
        lv = rng.uniform(-30, 30, S).astype(f32)
        ov = rng.standard_normal((S, 4)).astype(f32)
        lv[rng.random(S) < 0.5] = -np.inf
        a, ws = merge_item(lv, ov, S, 16, f32(1))
        assert np.isfinite(a).all()
        lv[:] = -np.inf
        a, ws = merge_item(lv, ov, S, 16, f32(1))
        assert ws == 0 and (a == 0).all(), "all -inf rows must give 0 (scl = 0), like merge_finalize"


# ------------------------------------------------------------------------------------------------
# Part B: end to end through k1_emulate (torch)
# ------------------------------------------------------------------------------------------------
def e2e(cases):
    import torch   # noqa: F401
    sys.path.insert(0, str(HERE))
    import k1_emulate as E
    res = []
    for name, kw in cases:
        data = E.generate_input(**kw)
        kc0, vc0 = data["k_cache"].clone(), data["v_cache"].clone()
        out_v4 = E.emulate_k1(data, n_sm=N_SM, cps=CPS[("sliding", 1)], fold=True)
        cfg = data["config"]
        B, L = cfg["batch"], cfg["q_len"]
        cs = pick_cs("sliding" if cfg["window"] else "full", L, B, cfg["num_kv_heads"] // 2)
        mc = MIN_CHUNK[("sliding" if cfg["window"] else "full", L)]
        saved = (E.grid_x, E.plan, E.merge_fold)
        merged_rows = []

        def plan_c(seq_l, W, Gx, min_chunk, cs=cs):
            reqs = cluster_plan(seq_l, W, cs, min_chunk)
            nt = [r[2] for r in reqs]
            S = [r[0] for r in reqs]
            return nt, max(r[1] for r in reqs), S, [b * cs for b in range(len(reqs))]

        def merge_c(pO, pL, S):
            """cluster merge semantics: rank rk merges rows [rk*R/S, (rk+1)*R/S) from every peer's partial, same math."""
            R = pL.shape[1]
            o4 = torch.zeros((R, 16, 4))
            wsum = torch.zeros(R)
            seen = torch.zeros(R, dtype=torch.int64)
            for rk in range(S):
                lo, hi = (rk * R) // S, ((rk + 1) * R) // S
                sl = slice(lo, hi)
                M = pL[:S, sl].max(0).values
                Mu = torch.where(M == float("-inf"), torch.zeros_like(M), M)
                for s_ in range(S):
                    w = torch.exp2(pL[s_, sl] - Mu)
                    wsum[sl] = wsum[sl] + w
                    o4[sl] = o4[sl] + w[:, None, None] * pO[s_, sl].reshape(-1, 16, 4)
                seen[sl] += 1
            assert bool((seen == 1).all()), "cluster merge row coverage"
            merged_rows.append(R)
            return o4, wsum
        try:
            E.grid_x = lambda n_sm, B_, HP, cps=1, cs=cs: B_ * cs
            E.plan = lambda seq_l, W, Gx, min_chunk=mc: plan_c(seq_l, W, Gx, min_chunk)
            E.merge_fold = merge_c
            d2 = dict(data)
            d2["k_cache"], d2["v_cache"] = kc0.clone(), vc0.clone()
            vis = {}
            out_c = E.emulate_k1(d2, n_sm=N_SM, cps=1, fold=True, visits=vis, min_chunk=mc)
            n_cov = E.check_coverage(d2, vis)
        finally:
            E.grid_x, E.plan, E.merge_fold = saved
        ok, msg = E.check_implementation(d2, out_c, d2["k_cache"], d2["v_cache"])
        same = bool(torch.equal(out_c, out_v4))
        dmax = float((out_c.float() - out_v4.float()).abs().max())
        res.append((name, ok, same, dmax, cs, n_cov, msg))
        assert ok, f"{name}: judge rule failed: {msg}"
    return res


# ------------------------------------------------------------------------------------------------
# Part C (--host, laptop only: torch without a GPU): k1_v9c's host logic against a fake extension
# ------------------------------------------------------------------------------------------------
def host_logic():
    """Imports k1_v9c (its CUDA build fails without nvcc -> _MOD None), swaps in a fake _MOD that records k1_forward's
    arguments and reports a cluster occupancy table, then checks _k1's cluster pick, grid, padding, flags, buffers and
    the fallbacks (unlaunchable, capture before caching, table off, self-test verdict)."""
    import torch
    spec = __import__("importlib.util").util.spec_from_file_location("k1v9c_host", str(KFILE))
    m = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(m)
    calls = []
    gpcs = [16] * 6 + [18] * 2                          # a 132-SM GPC split (guess); co-resident clusters per size:
    occ_table = {c: sum(n // c for n in gpcs) for c in range(2, 17)}   # padded (1 CTA/SM); unpadded fits 2 per SM

    class Fake:
        def k1_occupancy(self, L, st):
            return 2

        def k1_cluster_occupancy(self, L, st, cs, pad):
            return occ_table[cs] if pad else sum((2 * n) // cs for n in gpcs) * (occ_table[cs] > 0)

        def k1_forward(self, *a):
            calls.append(a)
    m._MOD = Fake()
    m._SMS["cpu"] = N_SM
    capturing = [False]
    torch.cuda.is_current_stream_capturing = lambda: capturing[0]
    dev = torch.device("cpu")

    def case(B, L=1, layer="sliding", hkv=2):
        d = {"qkv": torch.zeros((B, L, (2 * 8 * hkv + 2 * hkv) * 64), dtype=torch.bfloat16),
             "seq_lens": torch.full((B,), 80000, dtype=torch.int32), "cache_loc": torch.zeros(B * L, dtype=torch.int64),
             "page_table": torch.zeros((B, 2), dtype=torch.int32), "k_cache": torch.zeros(4, dtype=torch.uint8),
             "v_cache": torch.zeros(4, dtype=torch.uint8), "k_scale": torch.ones(1), "v_scale": torch.ones(1)}
        d["config"] = {"batch": B, "q_len": L, "num_q_heads": 8 * hkv, "num_kv_heads": hkv, "head_dim": 64,
                       "page_size": 1, "window": 2048 if layer == "sliding" else 0,
                       "rope_theta": 5e5 if layer == "sliding" else 0.0, "qk_scale_factor": 5.47, "sm_scale": 0.125,
                       "kv_scale": 1.0}
        return d
    # k1_forward positional arguments (k1_v9c._k1's call)
    A = dict(part_o=9, gx=19, fold=23, err=26, cs=27, pad=28, flags=29)
    out = []
    # self-test verdict gates the default route
    m._STATE["cluster_ok"] = {1: False, 8: False}
    m._k1(case(1))
    assert calls[-1][A["cs"]] == 0, "cluster used before the self-test passed"
    m._STATE["cluster_ok"] = {1: True, 8: False}
    # (B, cs, padded): B=12 -> cs 11, but 12 padded 11-CTA clusters don't fit the modeled GPCs -> unpadded
    for B, want_cs, want_pad in ((1, 16, 1), (16, 8, 1), (32, 4, 1), (3, 16, 1), (40, 3, 1), (12, 11, 0), (67, 0, 0)):
        m._k1(case(B))
        a = calls[-1]
        got = (a[A["cs"]], a[A["gx"]], a[A["pad"]], a[A["flags"]], a[A["fold"]])
        assert a[A["cs"]] == want_cs, f"B={B}: cs {a[A['cs']]} != {want_cs}"
        if want_cs:
            assert a[A["gx"]] == B * want_cs and a[A["fold"]] == 1 and a[A["flags"]] == m.K1_C1_RELAX
            assert a[A["pad"]] == (m.K1_PAD1_BYTES if want_pad else 0), "sliding L=1 pads to 1 CTA/SM when it fits"
            assert a[A["part_o"]].numel() == 1, "no global partials on the cluster path"
        else:
            assert a[A["gx"]] == N_SM and a[A["part_o"]].numel() >= N_SM * 16 * 64, "v4 grid / partials"
        out.append(f"B={B}: cs={got[0]} grid={got[1]} pad={got[2] // 1024}K")
    # full layers / L=8 stay on the v4 merge unless asked (table), and lab overrides work
    m._k1(case(16, layer="full"))
    assert calls[-1][A["cs"]] == 0
    m._k1(case(16, layer="full"), cluster=True)
    assert calls[-1][A["cs"]] == 16 and calls[-1][A["pad"]] == 0, "full lab: cap 16 (cps 2), 16 co-resident unpadded"
    m._k1(case(16, L=8), cluster=True)
    assert calls[-1][A["cs"]] == 8, "L=8 sliding lab: cs capped at 32 tiles / min_chunk 4 = 8"
    m._k1(case(1), cluster=True, cs=4, relax=3)
    assert calls[-1][A["cs"]] == 4 and calls[-1][A["flags"]] == 3
    m._k1(case(1), fold=False)
    assert calls[-1][A["cs"]] == 0, "separate-combine path never clusters"
    # unlaunchable (no co-resident cluster at all) -> v4 merge; one wave short -> unpadded / smaller cs
    saved_occ = dict(occ_table)
    occ_table[16] = 0
    m._CL_OCC.clear()
    m._CL_PICK.clear()
    m._k1(case(1))
    assert calls[-1][A["cs"]] == 15, "cs 16 unlaunchable -> 15"
    occ_table.update({c: 0 for c in occ_table})
    m._CL_OCC.clear()
    m._CL_PICK.clear()
    m._k1(case(1))
    assert calls[-1][A["cs"]] == 0, "nothing launchable -> v4 merge"
    # first need under capture with nothing cached -> v4 merge for that call
    occ_table.update(saved_occ)
    m._CL_OCC.clear()
    m._CL_PICK.clear()
    capturing[0] = True
    m._k1(case(16))
    assert calls[-1][A["cs"]] == 0, "uncached occupancy under capture must not query"
    capturing[0] = False
    m._k1(case(16))
    assert calls[-1][A["cs"]] == 8
    return out


# ------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    n = source_anchor_test()
    print(f"[src] {n} anchors found in k1_v9c.py; merge_cluster per-item math == merge_finalize's")
    print(f"[plan] {plan_test()} random cluster plans (any CS 2..16, min_chunk 1/2/4, W 2048/0): tiles once, one last split,"
          f" S_b <= CS, splits <= C_b and non-empty")
    for r in scored_plan_test():
        print(f"[plan] {r}; cluster plan == v4 plan (same splits)")
    n_idx = 0
    for L in (1, 8):
        for CS in range(2, CL_MAX + 1):
            for S in range(2, CS + 1):
                ops = index_test(L, S, CS)
                assert ops == fold_operands(L, S), f"operand order differs from the v4 merge (L={L} S={S})"
                n_idx += 1
        gate_out_addr_test(L)
        gate_out_addr_test(L, 128, 16)
    print(f"[index] {n_idx} (L, S_b, CS) configs: rows x quads merged once, reads only ranks < S_b inside ost/lse,"
          f" warp-uniform predicates, half-warp rows; operand order == v4 fold merge (bit-identical items);"
          f" gate/out addresses == v4")
    neg_inf_rows_test()
    print("[num] -inf lse rows: finite, all -inf -> 0 (as merge_finalize)")
    for B, seed in ((1, 601), (16, 603), (32, 605)):
        pre, post, cs, S = numerics_case(B, 1, seed)
        assert pre < 1e-5 and post < 6e-3, (B, pre, post)
        print(f"[num] dec{B:<2d} sliding L=1 CS={cs} S_b={S}: merged O rel L2 {pre:.2e} vs fp64 attention; after norm-gate"
              f" (bf16) {post:.2e}")
    for B, seed in ((1, 701), (16, 703)):
        pre, post, cs, S = numerics_case(B, 8, seed)
        assert pre < 1e-5 and post < 6e-3, (B, pre, post)
        print(f"[num] ver{B:<2d} sliding L=8 CS={cs} S_b={S} (lab path): merged O rel L2 {pre:.2e}; after norm-gate {post:.2e}")
    # negative controls: each must trip an assertion or the numeric bound
    ctls = {"rows_by_CS": dict(B=1, L=1, seed=11, CS=16), "row_shift": dict(B=1, L=1, seed=12),
            "no_lse": dict(B=1, L=1, seed=13), "own_lse": dict(B=1, L=1, seed=14),
            "lane_pred": dict(B=1, L=1, seed=15, CS=16), "S_eq_CS": dict(B=1, L=1, seed=16, CS=16)}
    for name, kw in ctls.items():
        caught = None
        try:
            if name in ("rows_by_CS", "lane_pred", "S_eq_CS"):
                # short requests (S_b < CS) expose these; seq ~300 -> 5 tiles -> S_b 3 at min_chunk 2
                rng_seqs = [300]
                cs = kw["CS"]
                reqs = cluster_plan(rng_seqs, WINDOW, cs, 2)
                if name == "S_eq_CS":
                    reqs = [(cs, C, nt) for (S, C, nt) in reqs]
                    for c in range(cs):
                        wk = cta_work_cluster(c, cs, reqs)
                        b, rk, ta, tb, last = wk
                        assert tb > ta or reqs[b][2] == 0, "empty split for a request with tiles"
                elif name == "rows_by_CS":
                    index_test(1, reqs[0][0], cs, split_by=cs)
                else:
                    index_test(1, reqs[0][0], cs, uniform_pred=False)
                    # per-lane predicate: lanes past r_hi skip the shuffles their half-warp partners execute
                    S = reqs[0][0]
                    for rk in range(S):
                        wk_by_warp = {}
                        for tid, k, r, u, mine, wkp in merge_items(1, S, rk, uniform_pred=False):
                            wk_by_warp.setdefault(tid >> 5, set()).add(wkp)
                        assert all(len(v) == 1 for v in wk_by_warp.values()), "shuffle predicate not warp-uniform"
            else:
                pre, post, _, _ = numerics_case(ctl=name, **kw)
                assert pre < 1e-5 and post < 6e-3, f"numeric bound (pre {pre:.2e})"
        except AssertionError as ex:
            caught = str(ex).splitlines()[0][:70]
        assert caught, f"negative control {name} was NOT caught"
        print(f"[neg] {name:10s} caught: {caught}")
    print(f"part A done in {time.time() - t0:.1f}s: ALL PASS")
    if "--host" in sys.argv:
        for r in host_logic():
            print(f"[host] {r}")
        print("[host] k1_v9c._k1 host logic (fake extension): picks, grids, padding, flags, fallbacks OK")
    if "--no-e2e" in sys.argv:
        return
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        print("part B skipped (no torch here; run on the pod CPU: python3 -I kernels/muse_attention_decode/k1_c1_emulate.py)")
        return
    t1 = time.time()
    cases = [("dec1 sliding ctx80000", dict(batch=1, ctx=80000, heads="tp8", layer="sliding", seed=601, page=1, dist="soft")),
             ("dec16 sliding agentic", dict(batch=16, ctx=6000, heads="tp8", layer="sliding", seed=603, page=1, dist="soft", spread="agentic")),
             ("dec32 sliding agentic", dict(batch=32, ctx=6000, heads="tp8", layer="sliding", seed=605, page=1, dist="soft", spread="agentic")),
             ("B3 ctx300 sliding (idle ranks)", dict(batch=3, ctx=300, heads="tp8", layer="sliding", seed=607, page=1))]
    for name, ok, same, dmax, cs, n_cov, msg in e2e(cases):
        print(f"[e2e] {name:30s} CS={cs:2d} judge {'PASS' if ok else 'FAIL'} {msg[:60]} | == v4-plan fold output: {same}"
              f" (max|d| {dmax:.2e}) cov_rows={n_cov}")
    print(f"part B done in {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()

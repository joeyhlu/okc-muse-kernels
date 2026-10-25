"""k1_v8f consumer-loop emulation, numpy only (no torch, no GPU). See k1_pp_design.md section 7.5.

  python -I kernels/muse_attention_decode/k1_v8f_emulate.py [--quick]

What it models. One consumer warpgroup is one KV head: 128 threads, with w = tid>>5, g = (tid&31)>>2, t = tid&3.

  * The register maps:
    - the S/O accumulators (m64nNk32 f32 D: a thread holds rows 16w+g and 16w+g+8, columns 8j+2t+{0,1}; O has j = 8
      for the n72 row-sum columns);
    - the 8-bit A fragment of PV (a0..a3 = row g / g+8 x k 4t.. / 16+4t..);
    - f_pack's byte order;
    - the permuted PV key order key(s, k) = 16q + 8(i>>1) + 2t + (i&1) (k1_wg_design.md section 3.3).
    The wgmma itself is the matrix product on logical tiles; everything around it is done per thread, per register.
  * k1_pp's consumer control flow, transcribed from k1_v8f.py:
    - first tile, ring steps, new-tile step, tail PV;
    - (a) rolled and (b) unrolled x2;
    - masks (key_lim, causal, window) and the masked / unmasked choice;
    - the branchless rescale (a = 1.0 exactly when the max did not move);
    - optional ping-pong sections.
  * An async wgmma model. An issue snapshots the register operands and makes its accumulator unreadable (NaN) until
    the wait that retires it. Each of these is a HAZARD:
    - a non-wgmma write to any register of an in-flight wgmma;
    - a non-wgmma read of an in-flight accumulator;
    - a slot release while a wgmma still reads the slot.
  * A ptxas model of C7513: a non-wgmma definition of a wgmma input inside the open pipeline stage (first issue ->
    queue empty), or a stage still open at a loop back edge, is FLAGGED ("would serialize"). It is calibrated on
    k1_v8's order, which the box showed serialized (k1_pp_design.md section 6).
  * The ring: the producer (k1_v8's code, unchanged) and both consumer WGs (their emulated traces) under random
    interleavings, checking:
    - full/empty parities, with exact phase on every pass (no ABA);
    - slot reuse only after both releases;
    - that a consumer reads only published tiles;
    - the ping-pong named-barrier accounting.

The checks:
  * same-rounding reference per split: float64, E4M3 P at the per-tile running max. Tight: errors are fp32-only;
  * exact softmax after the split merge: judge level;
  * key coverage: every row attends exactly its key set, once.

Negative controls (each must fail its check):
  * NC1: v8's order -> ptxas model FLAG;
  * NC2: pack/rescale before wait_group 0 -> HAZARD;
  * NC3: a stale rescale factor -> tight numerics;
  * NC4: release before PV retires -> HAZARD;
  * NC5: a P-pack byte swap -> tight numerics.
"""
import sys
import time

import numpy as np

T = 128
TID = np.arange(T)
WQ = TID >> 5
G = (TID & 31) >> 2
TQ = TID & 3
TILE = 64
L = 8
P_OFFSET = np.float32(8.0)
LOG2E = 1.4426950408889634
NEG = np.float32(-np.inf)
F32 = np.float32

# ---------------------------------------------------------------------------------------------- register maps
R32 = np.arange(32)
S_ROW = 16 * WQ[:, None] + G[:, None] + 8 * ((R32[None, :] >> 1) & 1)        # S[4j + 2rb + e]
S_COL = 8 * (R32[None, :] >> 2) + 2 * TQ[:, None] + (R32[None, :] & 1)
S_RB = (R32 >> 1) & 1
R36 = np.arange(36)
O_ROW = 16 * WQ[:, None] + G[:, None] + 8 * ((R36[None, :] >> 1) & 1)
O_COL = 8 * (R36[None, :] >> 2) + 2 * TQ[:, None] + (R36[None, :] & 1)          # 64..71 for j = 8 (row sums)
O_RB = (R36 >> 1) & 1
PACK_SRC = np.zeros((2, 4, 4), dtype=np.int64)        # f_pack: P register (st, 2*h2 + rb), byte i <- S register
for _st in range(2):
    for _h2 in range(2):
        _q = 2 * _st + _h2
        _ja, _jb = 2 * _q, 2 * _q + 1
        for _rb in range(2):
            PACK_SRC[_st, 2 * _h2 + _rb] = [4 * _ja + 2 * _rb, 4 * _ja + 2 * _rb + 1, 4 * _jb + 2 * _rb, 4 * _jb + 2 * _rb + 1]
A_ROW = np.zeros((T, 4, 4), dtype=np.int64)           # A fragment (8-bit m64nNk32): register ri, byte i -> (row, k)
A_K = np.zeros((T, 4, 4), dtype=np.int64)
for _ri in range(4):
    for _i in range(4):
        A_ROW[:, _ri, _i] = 16 * WQ + G + 8 * (_ri & 1)
        A_K[:, _ri, _i] = 16 * (_ri >> 1) + 4 * TQ + _i


def key_of(s, k):
    """PV logical (step s, k in 0..31) -> tile key (k1_wg_design.md section 3.3)"""
    q = 2 * s + (k >= 16)
    t, i = (k % 16) // 4, k % 4
    return 16 * q + 8 * (i >> 1) + 2 * t + (i & 1)


KEY_OF = np.array([[key_of(s, k) for k in range(32)] for s in range(2)])


def layout_check():
    """f_pack bytes == the A fragment's (row, permuted key); A covers every (row, k) of each step exactly once."""
    for st in range(2):
        cover = np.zeros((64, 32), dtype=np.int64)
        for ri in range(4):
            for i in range(4):
                src = PACK_SRC[st, ri, i]
                row_s, key_s = S_ROW[:, src], S_COL[:, src]
                assert np.array_equal(row_s, A_ROW[:, ri, i]), "pack row != A fragment row"
                assert np.array_equal(key_s, KEY_OF[st, A_K[:, ri, i]]), "pack key != permuted PV key"
                np.add.at(cover, (A_ROW[:, ri, i], A_K[:, ri, i]), 1)
        assert (cover == 1).all(), "A fragment coverage"
    for rr, cc in ((S_ROW, S_COL), (O_ROW, O_COL)):
        cov = np.zeros((64, int(cc.max()) + 1), dtype=np.int64)
        np.add.at(cov, (rr, cc), 1)
        assert (cov == 1).all(), "accumulator coverage"


# ---------------------------------------------------------------------------------------------- e4m3 (RN, satfinite)
_POS = np.array([(m * 2.0 ** -9) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
                 for e in range(16) for m in range(8)][:127])           # codes 0x00..0x7E, increasing, max 448


def e4m3(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    a = np.minimum(np.abs(x), 448.0)
    idx = np.clip(np.searchsorted(_POS, a), 1, 126)
    lo, hi = _POS[idx - 1], _POS[idx]
    up = (a - lo > hi - a) | ((a - lo == hi - a) & (idx % 2 == 0))      # ties to the even code
    r = np.where(up, hi, lo)
    return np.where(np.isnan(x), np.nan, s * r)


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32)


# ---------------------------------------------------------------------------------------------- async wgmma machine
class Machine:
    """One consumer warpgroup: register files, async wgmma queue, ptxas stage model, slot ownership, event trace."""

    def __init__(self, Qf, tiles, h):
        self.reg = {}
        self.cur, self.pend = [], []
        self.stage_defs = set()
        self.flags, self.hazards = [], []
        self.held = {"new": "new"}
        self.trace = []          # ring / ping-pong events, replayed by ring_check
        self.acc = []            # register-access trace of one step (FA3 equivalence)
        self.Qf, self.tiles, self.h = Qf, tiles, h
        self.cover = {}          # row -> list of attended absolute keys

    def inflight(self):
        return [op for grp in self.pend for op in grp] + self.cur

    def write(self, name, val):
        for op in self.inflight():
            if name in op["ins"] or name in op["outs"]:
                self.hazards.append(f"non-wgmma write of {name} while {op['kind']}({op['tile']}) is in flight")
        if self.pend or self.cur:
            self.stage_defs.add(name)
        self.reg[name] = val
        self.acc.append(("w", name))

    def read(self, name):
        for op in self.inflight():
            if name in op["outs"]:
                self.hazards.append(f"non-wgmma read of {name} while {op['kind']}({op['tile']}) is in flight")
        self.acc.append(("r", name))
        return self.reg[name]

    def issue(self, kind, ins, outs, slot, tile, fn):
        """one asm block = wgmma.fence + 2 wgmma + commit_group (k1_v8f f_qk / f_pv), or one wgmma group"""
        if not (self.pend or self.cur):
            self.stage_defs = set()                                    # a new pipeline stage opens
        for n in ins:
            if n in self.stage_defs:
                self.flags.append(f"C7513: {kind}({tile}) input {n} defined by non-wgmma code inside the open stage")
        if self.held.get(slot) != tile:
            self.hazards.append(f"{kind}({tile}) issued on slot {slot} holding {self.held.get(slot)}")
        op = dict(kind=kind, ins=list(ins), outs=list(outs), slot=slot, tile=tile, fn=fn,
                  snap={n: self.reg[n].copy() for n in ins})
        for n in outs:
            self.reg[n] = np.full_like(self.reg[n], np.nan)            # undefined until the retiring wait
        self.cur.append(op)
        self.acc.append(("issue", kind))

    def commit(self):
        self.pend.append(self.cur)
        self.cur = []
        self.acc.append(("commit",))

    def wait(self, n):
        assert not self.cur
        while len(self.pend) > n:
            for op in self.pend.pop(0):
                if self.held.get(op["slot"]) != op["tile"]:
                    self.hazards.append(f"{op['kind']}({op['tile']}) retired after slot {op['slot']} was released")
                res = op["fn"](op["snap"])
                for nm in op["outs"]:
                    if not np.isnan(self.reg[nm]).all():
                        self.hazards.append(f"{nm} written by non-wgmma code while {op['kind']}({op['tile']}) ran")
                    self.reg[nm] = res
        if not self.pend:
            self.stage_defs = set()                                    # queue empty: the stage closes
        self.acc.append(("wait", n))

    def back_edge(self):
        if self.pend or self.cur:
            self.flags.append(f"pipeline stage open across the loop back edge ({len(self.pend)} groups pending)")

    def wait_full(self, slot, parity, tile):
        self.trace.append(("wait_full", slot, parity, tile))
        if self.held.get(slot) is not None:
            self.hazards.append(f"slot {slot} re-acquired for tile {tile} while still holding {self.held[slot]}")
        self.held[slot] = tile

    def release(self, slot, tile):
        for op in self.inflight():
            if op["slot"] == slot and op["tile"] == tile:
                self.hazards.append(f"release of slot {slot} (tile {tile}) while {op['kind']} still reads it")
        if self.held.get(slot) != tile:
            self.hazards.append(f"release of slot {slot} for tile {tile}, held {self.held.get(slot)}")
        self.held[slot] = None
        self.trace.append(("arrive_empty", slot, tile))

    # ---- the two GEMMs (semantics on logical tiles)
    def qk(self, sname, slot, tile):
        Kt = self.tiles[tile][0]
        Qf = self.Qf

        def fn(snap):                                                  # scale-d 0: the old S is not an input
            S = (Qf.astype(np.float32) @ Kt.astype(np.float32).T).astype(np.float32)
            return S[S_ROW, S_COL]
        self.issue("QK", [], [sname], slot, tile, fn)
        self.commit()

    def pv(self, pname, slot, tile):
        Vt = self.tiles[tile][1]
        B = np.ones((2, 32, 72))
        B[:, :, :64] = Vt[KEY_OF]                                      # V^T with permuted keys + ones rows

        def fn(snap):
            P, O = snap[pname], snap["O"]
            add = np.zeros((64, 72))
            for st in range(2):
                A = np.zeros((64, 32))
                A[A_ROW, A_K] = P[:, st]
                add += A @ B[st]
            return f32(O.astype(np.float64) + add[O_ROW, O_COL])
        self.issue("PV", [pname, "O"], ["O"], slot, tile, fn)
        self.commit()


# ---------------------------------------------------------------------------------------------- per-thread pieces
def f_softmax(M, sname, mask, ts, klim, seq, W, cl2, st, cover=True):
    """k1_v8f f_softmax<MASK> on registers; st: dict with m [T, 2], a [T, 2] (float32)"""
    S = M.read(sname).astype(np.float32).copy()
    ak = ts + S_COL
    jj = 2 * WQ[:, None] + S_RB[None, :]
    ok = (ak < klim) & (ak <= seq + jj) & ((W == 0) | (ak > seq + jj - W))
    if mask:
        S = np.where(ok, S, NEG)
    else:
        assert ok.all(), "need == False but some key of the tile is masked"
    if cover:
        rows = S_ROW[ok]
        for r, k in zip(rows.tolist(), ak[ok].tolist()):
            M.cover.setdefault(r, []).append(k)
    m, a = st["m"], st["a"]
    for rb in range(2):
        cols = np.nonzero(S_RB == rb)[0]
        x = S[:, cols].max(1)                                          # pairwise tree = exact max
        x = x.reshape(4, 8, 4).max(2, keepdims=True).repeat(4, 2).reshape(T)   # quad shuffles (xor 1, 2)
        mo = m[:, rb]
        n = np.maximum(mo, f32(x * cl2))
        u = np.where(n == NEG, F32(0), n).astype(np.float32)
        with np.errstate(invalid="ignore"):
            a[:, rb] = np.where(n == mo, F32(1), np.exp2(f32(mo - u)))
        bo = f32(u - P_OFFSET)
        m[:, rb] = n
        with np.errstate(invalid="ignore"):
            S[:, cols] = np.exp2(f32(S[:, cols].astype(np.float64) * float(cl2) - bo[:, None].astype(np.float64)))
    M.write(sname, S.astype(np.float32))


def f_pack(M, sname, pname, swap=False):
    S = M.read(sname)
    src = PACK_SRC.copy()
    if swap:                                                           # NC5: bytes 1 and 2 exchanged
        src[..., [1, 2]] = src[..., [2, 1]]
    M.write(pname, e4m3(S[:, src]).astype(np.float32))


def f_rescale(M, a):
    O = M.read("O")
    M.write("O", f32(O * a[:, O_RB]))


# ---------------------------------------------------------------------------------------------- consumer schedules
class Ctx:
    def __init__(self, ntl, last, ts_of, need_of, seq, W, cl2, ST, pingpong=False):
        self.ntl, self.last, self.ts_of, self.need_of = ntl, last, ts_of, need_of
        self.seq, self.W, self.cl2, self.ST, self.pp = seq, W, cl2, ST, pingpong


def consumer_v8f(M, c, unr, bug=None):
    """k1_v8f.py k1_pp consumer (UNR = unr), line by line. bug: None | 'pack_before_wait0' | 'stale_alpha' |
    'early_release' | 'pack_swap' (negative controls)."""
    ST, ntl, last = c.ST, c.ntl, c.last
    seq, W, cl2 = c.seq, c.W, c.cl2
    st = {"m": np.full((T, 2), NEG, dtype=np.float32), "a": np.ones((T, 2), dtype=np.float32)}
    M.reg["O"] = np.zeros((T, 36), dtype=np.float32)
    for n in ("S", "SA", "SB"):
        M.reg[n] = np.zeros((T, 32), dtype=np.float32)
    for n in ("P", "PA", "PB"):
        M.reg[n] = np.zeros((T, 2, 4, 4), dtype=np.float32)
    anyt = ntl > 0
    h = M.h
    pp = {"k": 0, "n": (2 + max(ntl - 1, 0) + (1 if (last and anyt) else 0)) if (anyt or last) else 0}
    if c.pp and h == 1 and pp["n"] > 0:
        M.trace.append(("narrive", 2))

    def pp_enter():
        if c.pp:
            M.trace.append(("nsync", 2 + h))

    def pp_leave():
        if c.pp:
            pp["k"] += 1
            if not (h == 1 and pp["k"] == pp["n"]):
                M.trace.append(("narrive", 3 - h))

    def first(mask, S0, P0, tile, slot, ts, klim):
        pp_enter()
        M.qk(S0, slot, tile)
        pp_leave()
        M.wait(0)
        f_softmax(M, S0, mask, ts, klim, seq, W, cl2, st)
        f_pack(M, S0, P0, swap=(bug == "pack_swap"))
        f_rescale(M, st["a"])

    def step(mask, Sn, Pc, Pn, tile_n, slot_n, tile_c, slot_c, ts_n, klim_n, rel):
        a_prev = st["a"].copy()
        pp_enter()
        M.qk(Sn, slot_n, tile_n)
        M.pv(Pc, slot_c, tile_c)
        if bug == "early_release" and rel is not None:
            M.release(*rel)
        pp_leave()
        M.wait(1)
        f_softmax(M, Sn, mask, ts_n, klim_n, seq, W, cl2, st)
        if bug == "pack_before_wait0":
            f_pack(M, Sn, Pn)
            f_rescale(M, st["a"])
        M.wait(0)
        if rel is not None and bug != "early_release":
            M.release(*rel)
        if bug != "pack_before_wait0":
            f_pack(M, Sn, Pn, swap=(bug == "pack_swap"))
            f_rescale(M, a_prev if bug == "stale_alpha" else st["a"])

    def ring_step(Sn, Pc, Pn, i):
        M.wait_full((i + 1) % ST, ((i + 1) // ST) & 1, i + 1)
        ts = c.ts_of(i + 1)
        step(c.need_of(ts), Sn, Pc, Pn, i + 1, (i + 1) % ST, i, i % ST, ts, seq, (i % ST, i))

    def new_step(Sn, Pc, Pn):
        step(True, Sn, Pc, Pn, "new", "new", ntl - 1, (ntl - 1) % ST, seq, seq + L, ((ntl - 1) % ST, ntl - 1))

    def first_tile(S0, P0):
        if anyt:
            M.wait_full(0, 0, 0)
        ts = c.ts_of(0) if anyt else seq
        klim = seq if anyt else seq + L
        tile, slot = (0, 0) if anyt else ("new", "new")
        first((not anyt) or c.need_of(ts), S0, P0, tile, slot, ts, klim)

    def tail_pv(Pc, tile, slot, rel):
        pp_enter()
        M.pv(Pc, slot, tile)
        pp_leave()
        M.wait(0)
        if rel is not None:
            M.release(*rel)

    last_ring = (ntl - 1, (ntl - 1) % ST)
    if anyt or last:
        if unr == 0:
            first_tile("S", "P")
            for i in range(ntl - 1):
                ring_step("S", "P", "P", i)
                M.back_edge()
            if last and anyt:
                new_step("S", "P", "P")
            if last:
                tail_pv("P", "new", "new", None)
            else:
                tail_pv("P", last_ring[0], last_ring[1], (last_ring[1], last_ring[0]))
        else:
            first_tile("SA", "PA")
            i = 0
            while i + 2 < ntl:
                ring_step("SB", "PA", "PB", i)
                ring_step("SA", "PB", "PA", i + 1)
                M.back_edge()
                i += 2
            if anyt and i + 1 < ntl:
                ring_step("SB", "PA", "PB", i)
                if last:
                    new_step("SA", "PB", "PA")
                    tail_pv("PA", "new", "new", None)
                else:
                    tail_pv("PB", last_ring[0], last_ring[1], (last_ring[1], last_ring[0]))
            else:
                if last and anyt:
                    new_step("SB", "PA", "PB")
                    tail_pv("PB", "new", "new", None)
                elif last:
                    tail_pv("PA", "new", "new", None)
                else:
                    tail_pv("PA", last_ring[0], last_ring[1], (last_ring[1], last_ring[0]))
    assert not (M.pend or M.cur)
    if c.pp:
        assert pp["k"] == pp["n"], f"ping-pong sections {pp['k']} != pp_n {pp['n']}"
    return st["m"]


def consumer_v8(M, c):
    """k1_v8.py k1_pp consumer (the box-serialized order): QK(i+1) issued before softmax(i), P and O written while
    QK(i+1) is in flight, PV(i) left in flight across the loop back edge (negative control NC1)."""
    ST, ntl, last = c.ST, c.ntl, c.last
    seq, W, cl2 = c.seq, c.W, c.cl2
    st = {"m": np.full((T, 2), NEG, dtype=np.float32), "a": np.ones((T, 2), dtype=np.float32)}
    M.reg["O"] = np.zeros((T, 36), dtype=np.float32)
    for n in ("SA", "SB"):
        M.reg[n] = np.zeros((T, 32), dtype=np.float32)
    for n in ("PA", "PB"):
        M.reg[n] = np.zeros((T, 2, 4, 4), dtype=np.float32)

    def softmax_core(sname, pname, mask, ts, klim):                   # v8: mask, max, exp2, pack; warp vote
        f_softmax(M, sname, mask, ts, klim, seq, W, cl2, st)
        f_pack(M, sname, pname)
        return (st["a"] != 1).reshape(4, 32 * 2).any(1)                # __any_sync per warp

    def rescale_if(resc):
        a = st["a"].copy()
        a[~np.repeat(resc, 32)] = 1                                    # warps whose vote is false skip the rescale
        f_rescale(M, a)

    def step(i, Sc, Sn, Pc):
        M.wait(1)
        if i + 1 < ntl:
            M.wait_full((i + 1) % ST, ((i + 1) // ST) & 1, i + 1)
            M.qk(Sn, (i + 1) % ST, i + 1)
        else:
            M.commit()                                                 # placeholder group
        ts = c.ts_of(i)
        resc = softmax_core(Sc, Pc, c.need_of(ts), ts, seq)
        M.wait(1)
        if i >= 1:
            M.release((i - 1) % ST, i - 1)
        if resc.any():
            rescale_if(resc)
        M.pv(Pc, i % ST, i)

    if ntl > 0:
        M.wait_full(0, 0, 0)
        M.qk("SA", 0, 0)
        M.commit()                                                     # placeholder for "PV(-1)"
        for i in range(0, ntl, 2):
            step(i, "SA", "SB", "PA")
            if i + 1 < ntl:
                step(i + 1, "SB", "SA", "PB")
            M.back_edge()
    M.wait(0)
    if ntl > 0:
        M.release((ntl - 1) % ST, ntl - 1)                             # (v8 never releases the last tile; harmless)
    if last:
        M.qk("SA", "new", "new")
        M.wait(0)
        resc = softmax_core("SA", "PA", True, seq, seq + L)
        if resc.any():
            rescale_if(resc)
        M.pv("PA", "new", "new")
        M.wait(0)
    return st["m"]


def fa3_fwd_step_trace():
    """Register-access trace of FA3's fwd_step (hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp:1170-1207, RS PV, hdim 64
    FP8: RescaleOBeforeGemm false), written against the same Machine with one S and one P."""
    tiles = {k: (np.zeros((64, 64)), np.zeros((64, 64))) for k in (0, 1)}
    M = Machine(np.zeros((64, 64)), tiles, 0)
    M.reg.update(O=np.zeros((T, 36), np.float32), S=np.zeros((T, 32), np.float32), P=np.zeros((T, 2, 4, 4), np.float32))
    M.held.update({0: 0, 1: 1})
    st = {"m": np.zeros((T, 2), np.float32), "a": np.ones((T, 2), np.float32)}
    M.acc = []
    M.qk("S", 1, 1)                     # gemm<zero_init, wg_wait=-1>(QK(n+1)) -> tSrS
    M.pv("P", 0, 0)                     # gemm<wg_wait=-1>(PV(n)): tOrP, tOrO
    M.wait(1)                           # warpgroup_wait<1>; release K
    f_softmax(M, "S", False, 0, 1 << 30, 1 << 30, 0, F32(1), st, cover=False)   # mask, max_get_scale, online_softmax
    M.wait(0)                           # warpgroup_wait<0>; release V
    f_pack(M, "S", "P")                 # convert_type_out(tSrS -> tOrP)
    f_rescale(M, st["a"])               # rescale_o (after the GEMM)
    return M.acc


def v8f_step_trace():
    tiles = {k: (np.zeros((64, 64)), np.zeros((64, 64))) for k in (0, 1)}
    M = Machine(np.zeros((64, 64)), tiles, 0)
    c = Ctx(2, False, lambda i: 64 * i, lambda ts: False, 1 << 30, 0, F32(1), 7)
    acc_marks = []
    orig = M.wait_full

    def wf(slot, parity, tile):
        orig(slot, parity, tile)
        acc_marks.append(len(M.acc))
    M.wait_full = wf
    consumer_v8f(M, c, 0)
    a0 = acc_marks[1]                                                  # the ring step's full wait
    seg = M.acc[a0:]
    end = seg.index(("issue", "PV"), seg.index(("issue", "PV")) + 1)   # stop before the tail PV
    return seg[:end]


# ---------------------------------------------------------------------------------------------- ring replay
def ring_check(ntl, ST, traces, rng, n_runs):
    """Producer (k1_v8.py code, unchanged in v8f) + consumer WG traces under random interleavings."""
    D = ST - 2

    def producer():
        for i in range(ntl + D):
            j = i - D
            if j >= 0:
                yield ("publish", j % ST, j)
            if i < ntl:
                sl = i % ST
                if i >= ST:
                    yield ("wait", ("empty", sl), ((i // ST) - 1) & 1, (i // ST) - 1)
                yield ("fill", sl, i)

    def consumer(tr):
        for ev in tr:
            if ev[0] == "wait_full":
                yield ("wait", ("full", ev[1]), ev[2], ev[3] // ST)
                yield ("use", ev[1], ev[3])
            elif ev[0] == "arrive_empty":
                yield ("release", ev[1], ev[2])
            elif ev[0] == "nsync":
                yield ("nsync", ev[1])
            elif ev[0] == "narrive":
                yield ("narrive", ev[1])

    for _ in range(n_runs):
        done_c = {("full", s): 0 for s in range(ST)}
        done_c.update({("empty", s): 0 for s in range(ST)})
        pend = dict.fromkeys(done_c, 0)
        expect = {k: (1 if k[0] == "full" else 2) for k in done_c}
        content, published = {}, {}
        released = [set(), set()]
        nb = {2: {"gen": 0, "units": [], "waiting": {}}, 3: {"gen": 0, "units": [], "waiting": {}}}
        agents = [("P", producer()), ("C0", consumer(traces[0])), ("C1", consumer(traces[1]))]
        cur = {}
        blocked = {}
        for name, gen in agents:
            cur[name] = next(gen, None)

        def arrive(b):
            pend[b] += 1
            if pend[b] == expect[b]:
                pend[b] = 0
                done_c[b] += 1

        def enabled(name):
            act = cur[name]
            if act is None:
                return False
            if name in blocked:
                bid, g = blocked[name]
                return nb[bid]["gen"] > g
            if act[0] == "wait":
                return (done_c[act[1]] & 1) != act[2]
            return True

        while True:
            ready = [n for n, _ in agents if enabled(n)]
            if not ready:
                stuck = [n for n, _ in agents if cur[n] is not None]
                if stuck:
                    raise AssertionError(f"ring deadlock (ntl={ntl}, ST={ST}): {[(n, cur[n]) for n in stuck]}")
                break
            name = ready[int(rng.integers(len(ready)))]
            gen = dict(agents)[name]
            act = cur[name]
            if name in blocked:                                        # a finished nsync
                del blocked[name]
                cur[name] = next(gen, None)
                continue
            k = act[0]
            if k == "wait":
                assert done_c[act[1]] == act[3] + 1, f"ABA on {act[1]}: phase {done_c[act[1]]}, wanted {act[3] + 1}"
            elif k == "publish":
                assert content.get(act[1]) == act[2], "publishing a slot that does not hold the tile"
                published[act[1]] = act[2]
                arrive(("full", act[1]))
            elif k == "fill":
                if act[2] >= ST:
                    for hh in range(2):
                        assert act[2] - ST in released[hh], f"slot {act[1]} refilled before C{hh} released tile {act[2] - ST}"
                content[act[1]] = act[2]
            elif k == "use":
                assert published.get(act[1]) == act[2] and content.get(act[1]) == act[2], "read of an unpublished tile"
            elif k == "release":
                released[int(name[1])].add(act[2])
                arrive(("empty", act[1]))
            elif k in ("nsync", "narrive"):
                b = nb[act[1]]
                b["units"].append((name, k))
                g = b["gen"]
                if len(b["units"]) == 2:
                    kinds = sorted(u[1] for u in b["units"])
                    whos = {u[0] for u in b["units"]}
                    assert kinds == ["narrive", "nsync"] and len(whos) == 2, f"named barrier {act[1]} completed by {b['units']}"
                    b["units"] = []
                    b["gen"] += 1
                if k == "nsync" and b["gen"] == g:
                    blocked[name] = (act[1], g)
                    continue
            cur[name] = next(gen, None)
        for b in nb.values():
            assert not b["units"], f"named barrier left with {b['units']}"


# ---------------------------------------------------------------------------------------------- cases
QK_SM = 5.4730064863875 * 0.125


def make_case(rng, seq, W, dist, kscale=1.0):
    """One (request, KV head): e4m3 Q rows (token j = row >> 3), K / V for keys 0 .. seq+L-1 (the last L are new)."""
    n = seq + L
    q = rng.standard_normal((64, 64))
    q = q / np.sqrt((q * q).mean(1, keepdims=True))
    u = rng.standard_normal(64)
    k = 0.7 * u + 0.7 * rng.standard_normal((n, 64))
    pos = np.arange(n)[:, None] / max(n, 1)
    qd = q.mean(0) / np.linalg.norm(q.mean(0)) * 8
    if dist == "rising":            # the row max moves on (almost) every tile
        k = 0.3 * rng.standard_normal((n, 64)) + 2.5 * pos * qd
    elif dist == "falling":         # the max is in the first tile: a == 1.0 on every later tile
        k = 0.3 * rng.standard_normal((n, 64)) + 2.5 * (1 - pos) * qd
    elif dist == "edges":           # dominant keys at the window start, the last old key and a new key
        k = 0.5 * rng.standard_normal((n, 64))
        start = max(0, seq - W + 1) if W else 0
        for kk in {start, max(seq - 1, 0), seq + 3}:
            if kk < n:
                k[kk] = 3 * qd / 8 * 4
    elif dist == "soft":
        k = 0.05 * rng.standard_normal((n, 64))
    v = rng.standard_normal((n, 64))
    Qf = e4m3(q).astype(np.float32)
    Kf = e4m3(k / kscale).astype(np.float32)
    Vf = e4m3(v / kscale).astype(np.float32)
    cl2 = F32(np.float32(QK_SM) * np.float32(kscale) * np.float32(LOG2E))
    return Qf, Kf, Vf, cl2


def split_tiles(Kf, Vf, seq, W, nsplit, s_idx):
    n_old = min(seq, W - 1) if W else seq
    nt = (max(n_old, 0) + TILE - 1) // TILE
    S_b = max(1, min(nsplit, nt)) if nt else 1
    kstart = max(0, seq - W + 1) if W else 0
    t0 = s_idx * nt // S_b
    ntl = (s_idx + 1) * nt // S_b - t0
    tiles = {}
    for i in range(ntl):
        ts = kstart + (t0 + i) * TILE
        Kt, Vt = np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)
        idx = ts + np.arange(64)
        ok = idx < seq
        Kt[ok], Vt[ok] = Kf[idx[ok]], Vf[idx[ok]]                      # cp.async zero-fill beyond seq
        tiles[i] = (Kt, Vt)
    Kn, Vn = np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)
    Kn[:L], Vn[:L] = Kf[seq:seq + L], Vf[seq:seq + L]
    tiles["new"] = (Kn, Vn)
    return tiles, ntl, t0, kstart, S_b, nt


def ref_split(Qf, tiles, ntl, last, ts_of, seq, W, cl2):
    """Same-rounding reference: float32 S and p exactly as the kernel, E4M3 P at the per-tile running max, float64 O."""
    m = np.full(64, NEG, np.float32)
    O = np.zeros((64, 64))
    lsum = np.zeros(64)
    order = [(i, ts_of(i), seq) for i in range(ntl)] + ([("new", seq, seq + L)] if last else [])
    for tile, ts, klim in order:
        Kt, Vt = tiles[tile]
        S = (Qf.astype(np.float32) @ Kt.astype(np.float32).T).astype(np.float32)
        ak = ts + np.arange(64)[None, :]
        jj = (np.arange(64) >> 3)[:, None]
        ok = (ak < klim) & (ak <= seq + jj) & ((W == 0) | (ak > seq + jj - W))
        S = np.where(ok, S, NEG)
        n = np.maximum(m, f32(S.max(1) * cl2))
        u = np.where(n == NEG, F32(0), n).astype(np.float32)
        with np.errstate(invalid="ignore"):
            a = np.where(n == m, 1.0, np.exp2(f32(m - u)).astype(np.float64))
            p = np.exp2(f32(S.astype(np.float64) * float(cl2) - f32(u - P_OFFSET)[:, None].astype(np.float64)))
        p8 = e4m3(p)
        O = O * a[:, None] + p8 @ Vt.astype(np.float64)
        lsum = lsum * a + p8.sum(1)
        m = n
    with np.errstate(divide="ignore", invalid="ignore"):
        Os = np.where(lsum[:, None] > 0, O / lsum[:, None], 0.0)
        lse = np.where(lsum > 0, m.astype(np.float64) + np.log2(lsum) - 8.0, -np.inf)
    return Os, lse


def ref_exact(Qf, Kf, Vf, seq, W, cl2):
    out = np.zeros((64, 64))
    for r in range(64):
        j = r >> 3
        p = seq + j
        st = max(0, p - W + 1) if W else 0
        s = (Qf[r].astype(np.float64) @ Kf[st:p + 1].astype(np.float64).T) * float(cl2)
        w = np.exp2(s - s.max())
        out[r] = w @ Vf[st:p + 1].astype(np.float64) / w.sum()
    return out


def epilogue(M, m):
    """k1_pp epilogue: O / l from the n72 row-sum column, lse = m + log2(l) - 8, as logical [64 rows] arrays."""
    O = M.reg["O"].astype(np.float64)
    Ol = np.zeros((64, 72))
    Ol[O_ROW, O_COL] = O
    ml = np.zeros(64, np.float32)
    ml[S_ROW[:, 0]] = m[:, 0]
    ml[S_ROW[:, 2]] = m[:, 1]
    lt = Ol[:, 64]
    with np.errstate(divide="ignore", invalid="ignore"):
        Os = np.where(lt[:, None] > 0, Ol[:, :64] / lt[:, None], 0.0)
        lse = np.where(lt > 0, ml.astype(np.float64) + np.log2(lt) - 8.0, -np.inf)
    assert np.allclose(Ol[:, 64:72], lt[:, None], rtol=1e-6), "row-sum columns disagree"
    return Os, lse


def run_case(rng, seq, W, nsplit, dist, ST, unr, schedule="v8f", bug=None, pingpong=False, kscale=1.0, ring_runs=3):
    Qf, Kf, Vf, cl2 = make_case(rng, seq, W, dist, kscale)
    tol_tight = 2e-5
    res = dict(flags=[], hazards=[], tight=0.0, exact=0.0, ntl=[])
    nt_any = None
    parts = []
    for s_idx in range(max(1, nsplit)):
        tiles, ntl, t0, kstart, S_b, nt = split_tiles(Kf, Vf, seq, W, nsplit, s_idx)
        if s_idx >= S_b:
            break
        nt_any = nt
        last = s_idx == S_b - 1
        ts_of = (lambda kstart, t0: (lambda i: kstart + (t0 + i) * TILE))(kstart, t0)
        need_of = lambda ts: (ts + TILE > seq) or (W > 0 and ts < seq + L - W)      # noqa: E731
        traces = []
        for h in range(2):                     # both consumer WGs (same control flow; head 1 = a second draw of Q)
            Qh = Qf if h == 0 else Qf[::-1].copy()
            M = Machine(Qh, tiles, h)
            c = Ctx(ntl, last, ts_of, need_of, seq, W, cl2, ST, pingpong)
            m = consumer_v8(M, c) if schedule == "v8" else consumer_v8f(M, c, unr, bug)
            res["flags"] += M.flags
            res["hazards"] += M.hazards
            traces.append(M.trace)
            if not M.hazards:
                Os, lse = epilogue(M, m)
                Or, lr = ref_split(Qh, tiles, ntl, last, ts_of, seq, W, cl2)
                fin = np.isfinite(lr)
                err = float(np.abs(Os - Or).max() / max(np.abs(Or).max(), 1e-30))
                lerr = float(np.abs(lse[fin] - lr[fin]).max()) if fin.any() else 0.0
                assert np.array_equal(np.isfinite(lse), fin), "lse support differs"
                res["tight"] = max(res["tight"], err, lerr)
                if h == 0:
                    parts.append((Os, lse, M.cover))
        res["ntl"].append(ntl)
        ring_check(ntl, ST, traces, rng, ring_runs)
    if res["hazards"] or not parts:
        return res
    lse = np.stack([p[1] for p in parts])
    Mx = lse.max(0)
    w = np.exp2(lse - np.where(np.isfinite(Mx), Mx, 0)[None])
    O = (w[:, :, None] * np.stack([p[0] for p in parts])).sum(0) / w.sum(0)[:, None]
    ref = ref_exact(Qf, Kf, Vf, seq, W, cl2)
    rn = np.linalg.norm(ref, axis=1)
    res["exact"] = float((np.linalg.norm(O - ref, axis=1) / np.maximum(rn, 0.1 * rn.mean())).max())
    res["exact_l2"] = float(np.linalg.norm(O - ref) / np.linalg.norm(ref))
    for r in range(64):                        # coverage: each row attends exactly its window, each key once
        p = seq + (r >> 3)
        want = list(range(max(0, p - W + 1) if W else 0, p + 1))
        got = sorted(k for part in parts for k in part[2].get(r, []))
        assert got == want, f"coverage row {r}: {len(got)} keys, want {len(want)}"
    res["tol_tight"] = tol_tight
    res["nt"] = nt_any
    return res


def main():
    t_start = time.time()
    quick = "--quick" in sys.argv
    layout_check()
    print("[layout] f_pack bytes == PV A-fragment (row, permuted key); A, S and O (n72) maps cover each element once")
    fa3, v8f = fa3_fwd_step_trace(), v8f_step_trace()
    assert fa3 == v8f, f"v8f step != FA3 fwd_step:\n{fa3}\n{v8f}"
    print(f"[fa3] v8f rolled step register-access trace == FA3 fwd_step ({len(v8f)} events: "
          f"{' '.join(e[0] + ('(' + str(e[1]) + ')' if len(e) > 1 else '') for e in v8f)})")
    rng = np.random.default_rng(1234)
    # (seq, W, nsplit, dist, kscale): ntl 0..many, odd/even, first/last tile masks, sliding edges, split counts
    cases = [
        (0, 0, 1, "normal", 1.0),          # ntl = 0: the new tile only
        (1, 0, 1, "normal", 1.0),          # ntl = 1, one key
        (20, 0, 1, "edges", 1.0),
        (63, 0, 2, "normal", 1.0),
        (64, 0, 1, "rising", 1.0),
        (65, 0, 1, "falling", 1.0),        # ntl = 2
        (130, 0, 3, "normal", 1.0),        # 3 tiles over 3 splits: every split ntl = 1
        (200, 0, 1, "rising", 1.0),        # ntl = 4 (even)
        (300, 0, 1, "edges", 1.0),         # ntl = 5 (odd)
        (700, 0, 2, "soft", 1.0),
        (1000, 0, 3, "falling", 1.0),
        (1500, 0, 2, "rising", 1.0 / 64),
        (2047, 2048, 1, "edges", 1.0),     # sliding, first tile partial + masked window edge
        (2100, 2048, 2, "normal", 1.0),
        (3000, 2048, 3, "rising", 1.0),
        (5000, 2048, 5, "soft", 1.0),
        (2500, 2048, 1, "falling", 1.0 / 64),
    ]
    if quick:
        cases = cases[:9] + cases[12:14]
    n_fail = 0
    worst = dict(tight=0.0, exact=0.0)
    for (seq, W, ns, dist, ksc) in cases:
        for ST in (6, 7):
            for unr in (0, 1):
                pp = (ST == 7) == (unr == 1)                     # ping-pong accounting: ST 7 unrolled, ST 6 rolled
                t1 = time.time()
                r = run_case(rng, seq, W, ns, dist, ST, unr, pingpong=pp, kscale=ksc, ring_runs=3 if not quick else 2)
                ok = not r["flags"] and not r["hazards"] and r["tight"] < r["tol_tight"] and r["exact"] < 6e-2
                n_fail += int(not ok)
                worst["tight"] = max(worst["tight"], r["tight"])
                worst["exact"] = max(worst["exact"], r["exact"])
                if not ok or (ST == 7 and unr == 0):
                    print(f"[{'PASS' if ok else 'FAIL'}] seq={seq:5d} W={W:4d} splits={ns} {dist:7s} ks={ksc:g} ST={ST} "
                          f"{'unrolled' if unr else 'rolled  '}{' pp' if pp else '   '} ntl={r['ntl']} | tight {r['tight']:.1e}"
                          f" (tol {r['tol_tight']:.0e}) | exact row {r['exact']:.2e} L2 {r.get('exact_l2', 0):.2e}"
                          f" | flags {len(r['flags'])} hazards {len(r['hazards'])} ({time.time() - t1:.1f}s)"
                          + ("" if ok else f"\n        {r['flags'][:2]} {r['hazards'][:2]}"))
    n_runs = len(cases) * 4
    print(f"[v8f] {n_runs - n_fail}/{n_runs} runs pass (cases x ST 6/7 x rolled/unrolled; ring replay with random "
          f"interleavings; ping-pong accounting on ST 7 unrolled / ST 6 rolled); worst tight {worst['tight']:.1e}, worst exact row "
          f"{worst['exact']:.2e}")

    # ---- ring sweep: traces of every control-flow shape, many random interleavings
    n_ring = 0
    zt = lambda n: {**{i: (np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32)) for i in range(n)},   # noqa: E731
                    "new": (np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32))}
    for ntl in list(range(0, 21)) + [37, 60]:
        for ST in (6, 7):
            for last in ((True,) if ntl == 0 else (False, True)):
                for unr in (0, 1):
                    for pp in (False, True):
                        trs = []
                        for h in range(2):
                            M = Machine(np.zeros((64, 64), np.float32), zt(ntl), h)
                            c = Ctx(ntl, last, lambda i: 64 * i, lambda ts: True, 64 * ntl, 0, F32(1), ST, pp)
                            consumer_v8f(M, c, unr)
                            assert not M.hazards and not M.flags, (M.hazards[:2], M.flags[:2])
                            trs.append(M.trace)
                        ring_check(ntl, ST, trs, rng, 4 if quick else 12)
                        n_ring += 1
    print(f"[ring] {n_ring} control-flow shapes (ntl 0..20, 37, 60 x ST 6/7 x last x rolled/unrolled x ping-pong) x "
          f"{4 if quick else 12} random interleavings: no deadlock, exact phases (no ABA), slot reuse after both releases,"
          f" named-barrier instances = one sync + one arrive")

    # ---- negative controls: each must fail its own check
    nc = []
    r = run_case(rng, 1000, 0, 1, "rising", 7, 0, schedule="v8")
    nc.append(("NC1 v8 order (box: C7513)", bool(r["flags"]) and not r["hazards"] and r["tight"] < r["tol_tight"],
               f"ptxas-model flags {len(r['flags'])} (e.g. {r['flags'][0] if r['flags'] else '-'}); hazards "
               f"{len(r['hazards'])}; tight {r['tight']:.1e} (numerics correct, as on the box)"))
    r = run_case(rng, 1000, 0, 1, "rising", 7, 0, bug="pack_before_wait0")
    nc.append(("NC2 pack/rescale before wait_group 0", bool(r["hazards"]),
               f"hazards {len(r['hazards'])} (e.g. {r['hazards'][0] if r['hazards'] else '-'})"))
    r = run_case(rng, 1000, 0, 1, "rising", 7, 0, bug="stale_alpha")
    nc.append(("NC3 stale rescale factor", not r["hazards"] and r["tight"] > 1e-2,
               f"tight {r['tight']:.2e} vs tol {r['tol_tight']:.0e}; exact row {r['exact']:.2e}"))
    r = run_case(rng, 1000, 0, 1, "rising", 7, 1, bug="early_release")
    nc.append(("NC4 release before PV retires", bool(r["hazards"]),
               f"hazards {len(r['hazards'])} (e.g. {r['hazards'][0] if r['hazards'] else '-'})"))
    r = run_case(rng, 1000, 0, 1, "normal", 7, 0, bug="pack_swap")
    nc.append(("NC5 P-pack byte swap", not r["hazards"] and r["tight"] > 1e-2,
               f"tight {r['tight']:.2e}; exact row {r['exact']:.2e}"))
    trs = []
    for h in range(2):
        M = Machine(np.zeros((64, 64), np.float32), zt(20), h)
        consumer_v8f(M, Ctx(20, False, lambda i: 64 * i, lambda ts: True, 64 * 20, 0, F32(1), 7), 0)
        trs.append(M.trace)
    for name, mut in (("NC6 a missing release (tile 0)", "drop_release"), ("NC7 a wrong full parity (tile 7)", "parity")):
        bad = []
        for tr in trs:
            t2 = list(tr)
            if mut == "drop_release":
                t2.remove(next(e for e in t2 if e[0] == "arrive_empty"))
            else:
                k = next(i for i, e in enumerate(t2) if e[0] == "wait_full" and e[3] == 7)
                e = t2[k]
                t2[k] = (e[0], e[1], e[2] ^ 1, e[3])
            bad.append(t2)
        try:
            ring_check(20, 7, bad, rng, 4)
            nc.append((name, False, "ring check passed (should not)"))
        except AssertionError as ex:
            nc.append((name, True, f"ring check: {str(ex)[:90]}"))
    for name, failed_as_required, info in nc:
        print(f"[{'NEG-OK' if failed_as_required else 'NEG-BROKEN'}] {name}: {info}")
    n_bad_nc = sum(not x[1] for x in nc)
    print(f"done in {time.time() - t_start:.1f}s; positive {n_runs - n_fail}/{n_runs}; negative controls failing as "
          f"required {len(nc) - n_bad_nc}/{len(nc)}")
    if n_fail or n_bad_nc:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

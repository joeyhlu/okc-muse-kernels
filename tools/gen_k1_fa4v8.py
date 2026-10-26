"""Generate kernels/muse_attention_decode/k1_fa4v8.py (LAB copy): K1-FA4 v7 (tools/gen_k1_fa4v7.py) + two per-call
overhead levers for the sliding-window (and every other) route:
  * K1FA4_PREWAIT (FP8-PV producer): the page-table loads, and the first tile's K/V copies when that tile holds no new
    key, are issued BEFORE griddepcontrol.wait, so they overlap the prologue (only the prologue's writes - the E4M3 q
    and the new tokens' cache rows - are ordered by the wait; the Q TMA load moves after it).
  * K1FA4_COMB_HB: the combine + RMS-norm + sigmoid gate with HB heads per program (fewer, fatter programs; the v6
    combine runs one 64-thread program per (token, head), whose dispatch dominates at 2-4k programs).
  python tools/gen_k1_fa4v8.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_k1_fa4v6 as G6  # noqa: E402
import gen_k1_fa4v7 as G7  # noqa: E402

NL = "\n"
OUT = G6.KDIR / "k1_fa4v8.py"

MORE_TAIL = [
    ('K1FA4_L2PF = {"k": False, "v": False, "vh": False}' + NL,
     'K1FA4_L2PF = {"k": False, "v": False, "vh": False}' + NL
     + "# v8: FP8-PV producer issues the page-table loads (and the first tile's K/V when it holds no new key) before" + NL
     + "# griddepcontrol.wait, overlapping the prologue" + NL
     + "K1FA4_PREWAIT = False  # measured: sliding -0.0..-0.2 us, full B16/B32 +1.9/+2.5 us (box_out/sl_job2.log)" + NL
     + "# v8: combine heads per program: -1 = auto (largest power of 2 with NS_P2 x HB x 64 <= K1FA4_COMB_ELEMS, at most" + NL
     + "# K1FA4_COMB_CAP[q_len > 1]; measured best per shape, box_out/sl_job2.log), 0 = v6's" + NL
     + "# one program per (token, head), n = n heads per program" + NL
     + "K1FA4_COMB_HB = -1" + NL + "K1FA4_COMB_ELEMS = 4096" + NL + "K1FA4_COMB_CAP = {False: 4, True: 8}" + NL
     + "K1FA4_COMB_WARPS = 4" + NL
     + "K1FA4_PRO_BM = 0  # prologue rows per program; 0 = v6's (4 / 8 / 16 by row count)" + NL
     + "K1FA4_PRO_WARPS = 4" + NL),
    ("    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else 16)" + NL,
     "    block_m = K1FA4_PRO_BM or (4 if n_rows < 1024 else (8 if n_rows < 8192 else 16))" + NL),
    ("HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=4, **pdl_kw)", "HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=K1FA4_PRO_WARPS, **pdl_kw)"),
    ("    _k1fa4_combine_gate[(rows,)](" + NL
     + "        o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,"
     + NL + "        H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **pdl_kw)"
     + NL,
     "    hb = k1fa4_comb_hb(hq, triton.next_power_of_2(ns), d, length)" + NL
     + "    if hb > 1:" + NL
     + "        _k1fa4_combine_gate_hb[(tokens * (hq // hb),)](" + NL
     + "            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,"
     + NL + "            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), HB=hb, USE_PDL=pdl, num_warps=K1FA4_COMB_WARPS,"
     + NL + "            **pdl_kw)" + NL
     + "    else:" + NL
     + "        _k1fa4_combine_gate[(rows,)](" + NL
     + "            o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, K1FA4_RMS_EPS,"
     + NL + "            H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **pdl_kw)"
     + NL),
    ("# K1-FA4 v7 (team joeyhlu): FA4 v2 / v4 cores", "# K1-FA4 v8 (team joeyhlu): FA4 v2 / v4 cores"),
]

MORE_EXTRA = '''

def k1fa4_comb_hb(h, ns_p2, d, length):
    """Heads per combine program (K1FA4_COMB_HB); 1 = v6's combine kernel."""
    if K1FA4_COMB_HB == 0:
        return 1
    if K1FA4_COMB_HB > 0:
        return min(K1FA4_COMB_HB, h)
    hb = min(h, K1FA4_COMB_CAP[length > 1])
    while hb > 1 and ns_p2 * hb * d > K1FA4_COMB_ELEMS:
        hb //= 2
    return hb


@triton.jit
def _k1fa4_combine_gate_hb(OP, LSEP, QKV, OUT, stride_qkv, gate_off, n_rows, NS, B, L, eps,
                           H: tl.constexpr, D: tl.constexpr, NS_P2: tl.constexpr, HB: tl.constexpr,
                           USE_PDL: tl.constexpr):
    """_k1fa4_combine_gate for HB heads of one token per program (program = t * (H // HB) + head block); same
    arithmetic per (token, head) row. The gate load is issued before the PDL wait (qkv is not written by the core)."""
    pid = tl.program_id(0)
    t = pid // (H // HB)
    hh = (pid % (H // HB)) * HB + tl.arange(0, HB)
    b = t // L
    l = t % L
    d = tl.arange(0, D)
    g = tl.load(QKV + t * stride_qkv + gate_off + hh[:, None] * D + d[None, :]).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    s = tl.arange(0, NS_P2)
    smask = s < NS
    lrow = (b * H + hh) * L + l
    lse = tl.load(LSEP + s[:, None] * (B * H * L) + lrow[None, :], mask=smask[:, None], other=float("-inf"))
    r = t * H + hh
    o = tl.load(OP + s[:, None, None] * (n_rows * D) + r[None, :, None] * D + d[None, None, :],
                mask=smask[:, None, None], other=0.0)
    m = tl.max(lse, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - m[None, :])
    den = tl.sum(w, axis=0)
    acc = tl.sum(w[:, :, None] * o, axis=0) / tl.where(den > 0, den, 1.0)[:, None]
    acc = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(acc * acc, axis=1) / D
    y = acc * tl.rsqrt(var + eps)[:, None]
    y = y * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + t * (H * D) + hh[:, None] * D + d[None, :], y.to(tl.bfloat16))
'''

QLOAD = ("pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)" + NL)
POST = [
    ('        if const_expr(k1fa4_K["pdl"]):' + NL
     + "            cute.arch.griddepcontrol_wait()  # the prologue's E4M3 q and new K/V rows are visible from here on" + NL
     + '            if const_expr(k1fa4_K["trig"] == "start"):' + NL
     + "                cute.arch.griddepcontrol_launch_dependents()" + NL
     + "        storage = self._k1_storage" + NL + "        f8 = cutlass.Float8E4M3FN" + NL,
     '        pre = k1fa4_K["pdl"] and K1FA4_PREWAIT  # page tables (+ first tile K/V without new keys) before the wait'
     + NL + '        if const_expr(k1fa4_K["pdl"] and not pre):' + NL
     + "            cute.arch.griddepcontrol_wait()  # the prologue's E4M3 q and new K/V rows are visible from here on" + NL
     + '        if const_expr(k1fa4_K["pdl"] and k1fa4_K["trig"] == "start"):' + NL
     + "            cute.arch.griddepcontrol_launch_dependents()" + NL
     + "        storage = self._k1_storage" + NL + "        f8 = cutlass.Float8E4M3FN" + NL),
    ("            mK_h = mK[None, None, head_idx, None]" + NL
     + "            if warp_idx_in_wg == 0:" + NL
     + "                " + QLOAD
     + "                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))" + NL
     + "            q_producer_phase ^= 1" + NL
     + "            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)" + NL
     + "            nblk = n_block_max - n_block_min" + NL
     + "            if nblk > 0:" + NL
     + "                pg = cute.make_rmem_tensor((4,), Int32)" + NL
     + "                pkm.load_page_table(n_block_max - 1)" + NL
     + "                self._k1fa4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block_max - 1, rbase, pg)" + NL
     + "                for it in cutlass.range(nblk, unroll=1):" + NL
     + "                    n_block = n_block_max - 1 - it" + NL
     + "                    pipeline_k.producer_acquire(k_state)" + NL
     + '                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")' + NL
     + "                    pipeline_k.producer_commit(k_state)" + NL
     + "                    k_state.advance()" + NL
     + "                    self._k1fa4_gather_v(mV_h, pg, c, tidx, sVr, it % nraw, atom)" + NL
     + "                    cute.arch.cp_async_commit_group()" + NL,
     "            mK_h = mK[None, None, head_idx, None]" + NL
     + "            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)" + NL
     + "            nblk = n_block_max - n_block_min" + NL
     + "            if const_expr(not pre):" + NL
     + "                if warp_idx_in_wg == 0:" + NL
     + "                    " + QLOAD
     + "                    load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))" + NL
     + "            else:" + NL
     + "                if nblk <= 0:" + NL
     + "                    cute.arch.griddepcontrol_wait()" + NL
     + "                    if warp_idx_in_wg == 0:" + NL
     + "                        " + QLOAD
     + "                        load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))" + NL
     + "            if nblk > 0:" + NL
     + "                pg = cute.make_rmem_tensor((4,), Int32)" + NL
     + "                pkm.load_page_table(n_block_max - 1)" + NL
     + "                self._k1fa4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block_max - 1, rbase, pg)" + NL
     + "                if const_expr(pre):" + NL
     + "                    if n_block_max * self.tile_n > seqlen.seqlen_k - seqlen.seqlen_q:  # tile 0 holds a new key" + NL
     + "                        cute.arch.griddepcontrol_wait()" + NL
     + "                for it in cutlass.range(nblk, unroll=1):" + NL
     + "                    n_block = n_block_max - 1 - it" + NL
     + "                    pipeline_k.producer_acquire(k_state)" + NL
     + '                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")' + NL
     + "                    pipeline_k.producer_commit(k_state)" + NL
     + "                    k_state.advance()" + NL
     + "                    self._k1fa4_gather_v(mV_h, pg, c, tidx, sVr, it % nraw, atom)" + NL
     + "                    cute.arch.cp_async_commit_group()" + NL
     + "                    if const_expr(pre):" + NL
     + "                        if it == 0:" + NL
     + "                            cute.arch.griddepcontrol_wait()  # q and the new K/V rows are visible from here on" + NL
     + "                            if warp_idx_in_wg == 0:" + NL
     + "                                " + QLOAD
     + "                                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))" + NL),
    ("                    self._k1fa4_transpose_v(sVr32, tidx, j % nraw, vt_ptr + v_state.index * vt_stage_bytes, c, g,"
     + NL + "                                            rot)" + NL
     + "                    cute.arch.fence_view_async_shared()" + NL
     + "                    cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))" + NL
     + "                    v_state.advance()" + NL
     + "            tile_scheduler.prefetch_next_work()" + NL,
     "                    self._k1fa4_transpose_v(sVr32, tidx, j % nraw, vt_ptr + v_state.index * vt_stage_bytes, c, g,"
     + NL + "                                            rot)" + NL
     + "                    cute.arch.fence_view_async_shared()" + NL
     + "                    cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))" + NL
     + "                    v_state.advance()" + NL
     + "            q_producer_phase ^= 1" + NL
     + "            tile_scheduler.prefetch_next_work()" + NL),
    ("tuple(sorted(K1FA4_L2PF.items())))", "tuple(sorted(K1FA4_L2PF.items())), K1FA4_PREWAIT)"),
]


def main():
    out, n_edits, mapping = G7.build(
        more_tail=MORE_TAIL, post=POST, more_extra=MORE_EXTRA,
        title="K1-FA4 v8 attention (team joeyhlu, LAB). Generated by tools/gen_k1_fa4v8.py")
    OUT.write_bytes(out.encode("utf-8"))
    hits = G6.scan(out)
    long_lines = [i + 1 for i, ln in enumerate(out.split(NL)) if len(ln) > 400]
    print(f"[gen8] {OUT.name}: {len(out.encode())} bytes ({len(out.splitlines())} lines); lines > 400: {long_lines or 'none'}")
    print(f"[gen8] scan {OUT.name}: {'CLEAN' if not hits else hits}")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())

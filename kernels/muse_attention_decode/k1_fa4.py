"""K1-FA4: FP8-load / F16-compute L=8 full-layer attention built on the judge image's FA4 CuTe-DSL sm90 forward.

Reference code (open source, read at run time from the image, not copied here):
  sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py   (FA4 Hopper forward: consumer warpgroup,
      wgmma pipelines, pack-GQA, split-KV)            == repos/sglang/python/sglang/kernels/ops/attention/...
  .../paged_kv.py (cp.async page gather), .../interface.py (_flash_attn_fwd host path + split combine),
  .../flash_fwd_sm100.py dequant_v (the FP8 smem -> 16-bit smem upcast pattern this producer follows).

What this file changes (the image file is patched as text at import and written to /tmp so CuTe's AST
preprocessor can read it; the interface gets the same treatment):
  * launch min_blocks_per_mp 1 -> 2 and setmaxnreg (256, 56) -> (200, 56) for the 1-MMA-warpgroup config, so
    two CTAs (two consumer warpgroups) share an SM: at M = 64 rows (8 tokens x 8 q heads per KV head) one
    consumer chain per SM is latency-bound (k1_fa4_lab: F16, L2-resident v32 303 -> 257 us);
  * K/V stay FP8 E4M3 in HBM. A new producer (K1FA4Sm90.load) gathers FP8 rows with FA4's cp.async
    PagedKVManager into a 2-slot raw FP8 ring, then upcasts each 128-key tile to F16 (exact for E4M3) into the
    swizzled sK/sV stages the stock consumer reads, fences (generic -> async proxy) and arrives on the stock
    full barriers. QK and PV run as F16 wgmma with FP32 accumulation; P is F16 (stock FA3: E4M3 Q and P).
  * Only the shape family it was measured on is supported: page size 1, head dim 64, pack-GQA, causal,
    split-KV (num_splits >= 2, output via FA4's combine kernel), tile (64, 128), no intra-WG overlap.
"""
import importlib
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("SGLANG_CUTE_AOT_CACHE_DIR", "")  # process-local CuTe JIT cache (patched kernels)

import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import torch  # noqa: E402
from cutlass import Int32, const_expr, pipeline  # noqa: E402
from cutlass.cute import FastDivmodDivisor  # noqa: E402
from quack import copy_utils  # noqa: E402

VENDOR = "sglang.kernels.ops.attention.flash_attn.cute"
TMP = Path(os.environ.get("K1FA4_TMP", "/tmp/k1fa4"))
RAW_STAGES = 2
BAR_FREE, BAR_LANDED = 9, 10  # named barriers among the 128 producer threads (FA4 sm90 uses 1..6)


def _patched(modname, outname, reps, header=""):
    src = importlib.import_module(VENDOR + "." + modname)
    txt = Path(src.__file__).read_text()
    for a, b in reps:
        assert txt.count(a) == 1, ("patch anchor", modname, a, txt.count(a))
        txt = txt.replace(a, b)
    TMP.mkdir(parents=True, exist_ok=True)
    f = TMP / (outname + ".py")
    f.write_text(header + txt)
    spec = importlib.util.spec_from_file_location(outname, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[outname] = m
    spec.loader.exec_module(m)
    return m


SM90 = _patched("flash_fwd_sm90", "k1fa4_sm90", [
    ("min_blocks_per_mp=1,", 'min_blocks_per_mp=_K1["minb"],'),
    ("1: (256, 56),", '1: _K1["regs1"],'),
    # K/V smem stages are 16-bit (self.dtype) even when the gmem K/V are FP8
    ("(mK.element_type, (self.tile_n, self.tile_hdim), self.num_stages),",
     "(self.dtype, (self.tile_n, self.tile_hdim), self.num_stages),"),
    ("(mV.element_type, (self.tile_n, self.tile_hdimv), self.num_stages),",
     "(self.dtype, (self.tile_n, self.tile_hdimv), self.num_stages),"),
    # type check: K/V may be FP8 while Q is F16
    ("                    mQ,\n                    mK,\n                    mV,\n                    mO,\n",
     "                    mQ,\n                    mQ,\n                    mQ,\n                    mO,\n"),
    # the FP8 producer needs the raw ring, which lives in the shared storage struct
    ("        storage = smem.allocate(SharedStorage)\n",
     "        storage = smem.allocate(SharedStorage)\n        self._k1_storage = storage\n"),
], header='_K1 = {"minb": 2, "regs1": (200, 56), "cvt": "direct", "map": "own"}\n')

IFACE = _patched("interface", "k1fa4_iface", [
    ('        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}\n',
     '        input_tensors = {"q": q, "qv": qv}\n'),
    ("            arch // 10 == 10\n        ), \"FP8 is only supported on SM100",
     "            arch // 10 in (9, 10)\n        ), \"FP8 is only supported on SM100"),
])

from k1fa4_sm90 import PagedKVManager  # noqa: E402  (the image's paged_kv module, as imported there)


class K1FA4Sm90(SM90.FlashAttentionForwardSm90):
    """FA4 sm90 forward with an FP8-gather / F16-upcast producer (see module docstring)."""

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        raw_n = RAW_STAGES * self.tile_n * self.tile_hdim
        sKr_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, raw_n], 1024]
        sVr_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, raw_n], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageK1:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sKr: sKr_struct
            sVr: sVr_struct

        return SharedStorageK1

    @cute.jit
    def _k1_upcast(self, tiled_s2r, tiled_r2s, tidx, sXr, sX):
        """One 128-key tile: FP8 raw smem (128, 64) -> 16-bit swizzled smem stage (128, 64)."""
        thr_s2r = tiled_s2r.get_slice(tidx)
        thr_r2s = tiled_r2s.get_slice(tidx)
        tRs = thr_s2r.partition_S(cute.group_modes(sXr, 0, 1))
        tXs = thr_r2s.partition_D(cute.group_modes(sX, 0, 1))
        for i in cutlass.range_constexpr(cute.size(tRs.shape[1])):
            r8 = cute.make_fragment_like(tRs[None, i, None])
            cute.copy(tiled_s2r, tRs[None, i, None], r8)
            r16 = cute.make_fragment_like(tXs[None, i, None])
            if const_expr(SM90._K1["cvt"] == "f32"):
                r16.store(r8.load().to(cutlass.Float32).to(self.dtype))
            else:
                r16.store(r8.load().to(self.dtype))
            cute.copy(tiled_r2s, r16, tXs[None, i, None])

    @cute.jit
    def _k1_upcast_own(self, pkm, tiled_r2s, tidx, sXr, sX):
        """Same tile, but each thread upcasts exactly the 16-byte chunks its own cp.async wrote (FA4
        PagedKVManager mapping: 4 threads x 16 elements per row), so no barrier is needed: the thread's
        cp.async.wait_group covers visibility, and program order covers raw-slot reuse."""
        tRs = pkm.gmem_thr_copy_KV.partition_D(cute.group_modes(sXr, 0, 1))
        tXs = tiled_r2s.get_slice(tidx).partition_D(cute.group_modes(sX, 0, 1))
        for m in cutlass.range_constexpr(cute.size(tRs.shape[1])):
            r8 = cute.make_fragment_like(tRs[None, m, None])
            cute.autovec_copy(tRs[None, m, None], r8)
            r16 = cute.make_fragment_like(tXs[None, m, None])
            r16v = cute.make_tensor(r16.iterator, r8.layout)
            r16v.store(r8.load().to(self.dtype))
            cute.copy(tiled_r2s, r16, tXs[None, m, None])

    @cute.jit
    def load(self, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, mBias, sBias, tma_atom_Bias,
             pipeline_k, pipeline_v, pipeline_bias, pipeline_q, gmem_tiled_copy_Q, mPageTable,
             blocksparse_tensors, block_info, SeqlenInfoCls, TileSchedulerCls, num_splits: Int32 = 1):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        nthr = self.num_threads_per_warp_group
        storage = self._k1_storage
        raw_layout = cute.make_layout((self.tile_n, self.tile_hdim, RAW_STAGES),
                                      stride=(self.tile_hdim, 1, self.tile_n * self.tile_hdim))
        sKr = storage.sKr.get_tensor(raw_layout)
        sVr = storage.sVr.get_tensor(raw_layout)
        # upcast copies: 8 threads per 64-element row, 8 elements each (FP8 64-bit loads, F16 128-bit stores)
        tpr = self.tile_hdim // 8
        thr_layout = cute.make_ordered_layout((nthr // tpr, tpr), order=(1, 0))
        val_layout = cute.make_layout((1, 8))
        tiled_s2r = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=64),
            thr_layout, val_layout)
        tiled_r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128),
            thr_layout, val_layout)
        own = const_expr(SM90._K1["map"] == "own")
        skip = const_expr(SM90._K1["cvt"] == "skip")  # timing diagnostic only: no upcast (wrong numerics)
        tiled_r2s_own = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128),
            cute.make_ordered_layout((nthr // 4, 4), order=(1, 0)), cute.make_layout((1, 16)))

        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = copy_utils.tma_get_copy_fn(tma_atom_Q, 0, cute.make_layout(1), gQ, sQ,
                                                      single_stage=True)
            pkm = PagedKVManager.create(mPageTable, mK, mV, FastDivmodDivisor(mK.shape[0]), batch_idx, head_idx,
                                        tidx, seqlen.seqlen_k, 0, self.tile_n, self.tile_hdim, self.tile_hdimv,
                                        nthr, mK.element_type, arch=90)
            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_producer_phase ^= 1
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            nblk = n_block_max - n_block_min
            if nblk > 0:
                pkm.load_page_table(n_block_max - 1)
                pkm.load_KV(n_block_max - 1, sKr[None, None, 0], "K")
                pkm.load_KV(n_block_max - 1, sVr[None, None, 0], "V")
                cute.arch.cp_async_commit_group()
                for it in cutlass.range(nblk, unroll=1):
                    # slot (it + 1) % 2 held tile it - 1, converted by every thread before this barrier
                    if const_expr(not own):
                        cute.arch.barrier(barrier_id=BAR_FREE, number_of_threads=nthr)
                    if it + 1 < nblk:
                        n_next = n_block_max - 2 - it
                        pkm.load_page_table(n_next)
                        pkm.load_KV(n_next, sKr[None, None, (it + 1) % RAW_STAGES], "K")
                        pkm.load_KV(n_next, sVr[None, None, (it + 1) % RAW_STAGES], "V")
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)  # this thread's copies of tile it landed
                    if const_expr(not own):
                        cute.arch.barrier(barrier_id=BAR_LANDED, number_of_threads=nthr)  # everyone's did
                    r = it % RAW_STAGES
                    pipeline_k.producer_acquire(kv_producer_state)
                    if const_expr(skip):
                        pass
                    elif const_expr(own):
                        self._k1_upcast_own(pkm, tiled_r2s_own, tidx, sKr[None, None, r],
                                            sK[None, None, kv_producer_state.index])
                    else:
                        self._k1_upcast(tiled_s2r, tiled_r2s, tidx, sKr[None, None, r],
                                        sK[None, None, kv_producer_state.index])
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(pipeline_k.sync_object_full.get_barrier(kv_producer_state.index))
                    pipeline_v.producer_acquire(kv_producer_state)
                    if const_expr(skip):
                        pass
                    elif const_expr(own):
                        self._k1_upcast_own(pkm, tiled_r2s_own, tidx, sVr[None, None, r],
                                            sV[None, None, kv_producer_state.index])
                    else:
                        self._k1_upcast(tiled_s2r, tiled_r2s, tidx, sVr[None, None, r],
                                        sV[None, None, kv_producer_state.index])
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(kv_producer_state.index))
                    kv_producer_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_v.producer_tail(kv_producer_state)


IFACE.FlashAttentionForwardSm90 = K1FA4Sm90
TILE = (64, 128)


def num_splits_for(batch, hkv, max_seqlen_k, num_sms=132, ctas_per_sm=2):
    """The stock FA4 heuristic, scaled by the CTAs per SM this build gets (2)."""
    nnb = (max_seqlen_k + TILE[1] - 1) // TILE[1]
    return max(2, IFACE.num_splits_heuristic(batch * hkv, num_sms, nnb, 128) * ctas_per_sm)


def set_cvt(mode, mapping="own"):
    """cvt: 'direct' (E4M3 -> F16 cvt), 'f32' (via FP32, FA4 sm100 dequant's path) or 'skip' (timing only);
    mapping: 'own' (thread upcasts its own cp.async bytes, no barriers) or 'bar' (8x8 mapping, 2 named
    barriers per tile). A change drops the compiled kernels."""
    if SM90._K1["cvt"] != mode or SM90._K1["map"] != mapping:
        SM90._K1["cvt"] = mode
        SM90._K1["map"] = mapping
        IFACE._flash_attn_fwd.compile_cache = IFACE._get_jit_cache("fwd")


def fwd(q16, k8, v8, page_table, seqused_k, max_seqlen_k, softmax_scale, num_splits):
    """q16 [B, L, Hq, 64] float16 (unit-RMS queries); k8/v8 [P, 1, Hkv, 64] float8_e4m3fn (page size 1);
    page_table [B, max_pages] int32; seqused_k [B] int32 (cache lengths incl. the block). Returns bf16
    [B, L, Hq, 64] (FA4 split combine)."""
    assert num_splits >= 2 and k8.shape[1] == 1 and k8.dtype == torch.float8_e4m3fn
    out, _ = IFACE._flash_attn_fwd(q16, k8, v8, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
                                   page_table=page_table, softmax_scale=softmax_scale, causal=True,
                                   tile_mn=TILE, num_splits=num_splits, pack_gqa=True, intra_wg_overlap=False)
    return out

"""K1-FA4 v2: FP8 QK / F16 PV attention for L=8 (verify) blocks, full or sliding-window layers, built on the judge
image's FA4 CuTe-DSL sm90 forward. Successor of k1_fa4.py (v1: F16 QK and PV, K and V both upcast in smem).

Reference code (open source, read at run time from the image, never copied into the repo):
  sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py (consumer warpgroup, wgmma pipelines, pack-GQA,
  split-KV, local-window masking), .../paged_kv.py (cp.async page gather), .../interface.py (host path, split
  combine); == repos/sglang/python/sglang/kernels/ops/attention/flash_attn/cute/.

v2 changes (on top of v1's 2 CTAs/SM: min_blocks_per_mp 2, setmaxnreg (200, 56)):
  * QK is an FP8 wgmma: Q is the E4M3 query (as stock FA3 rounds it, A16), loaded by FA4's pack-GQA TMA into an
    FP8 sQ; K is cp.async-gathered by FA4's PagedKVManager straight into the FP8 swizzled sK stages and signalled
    with FA4's own cp.async mbarrier arrive. No K upcast.
  * PV stays F16 (P in F16: finer than stock's E4M3 P). V is gathered into a 2-slot raw FP8 ring and each thread
    upcasts exactly the bytes its own cp.async wrote into the swizzled F16 sV stage (v1's barrier-free mapping).
  * Sliding-window layers use FA4's local mask (window_size_left = window - 1, causal), same semantics as r1's
    FA3 call (window (2047, 0)).
Packaging (notes/assumptions.md A19): the image's files are read, patched as text (every anchor asserted unique)
and loaded from a private mkdtemp dir under NEW module names (k1fa4v2_*); no existing module or sys.modules entry is
touched, no environment variable is written, and the CuTe JIT cache is process-local (cache_dir=None in the patched
interface's own cache factory).
"""
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, const_expr, pipeline
from cutlass.cute import FastDivmodDivisor
from quack import copy_utils

VENDOR = "sglang.kernels.ops.attention.flash_attn.cute"
TMP = Path(tempfile.mkdtemp(prefix="k1fa4v2_"))
RAW_STAGES = 2


def _patched(modname, outname, reps, header=""):
    assert outname not in sys.modules, outname
    src = importlib.import_module(VENDOR + "." + modname)
    txt = Path(src.__file__).read_text()
    for a, b in reps:
        assert txt.count(a) == 1, ("patch anchor", modname, a, txt.count(a))
        txt = txt.replace(a, b)
    f = TMP / (outname + ".py")
    f.write_text(header + txt)
    spec = importlib.util.spec_from_file_location(outname, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[outname] = m  # a new private name (A19)
    spec.loader.exec_module(m)
    return m


SM90 = _patched("flash_fwd_sm90", "k1fa4v2_sm90", [
    ("min_blocks_per_mp=1,", 'min_blocks_per_mp=_K2["minb"],'),
    ("1: (256, 56),", '1: _K2["regs1"],'),
    ("        super().__init__(*args, **kwargs)\n",
     '        kwargs["num_stages"] = _K2["stages"]\n        super().__init__(*args, **kwargs)\n'),
    # V smem stages are 16-bit (self.dtype) while gmem V is FP8; K and Q keep their (FP8) element type
    ("(mV.element_type, (self.tile_n, self.tile_hdimv), self.num_stages),",
     "(self.dtype, (self.tile_n, self.tile_hdimv), self.num_stages),"),
    # FA4's type check requires 16-bit Q/K/V; this build checks its own contract on the host instead
    ("        self._check_type(\n", "        _k2_skip_check(\n"),
    # QK^T as an E4M3 x E4M3 wgmma (both operands K-major)
    ("        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(\n            self.dtype,\n            self.dtype,\n",
     "        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(\n"
     "            cutlass.Float8E4M3FN,\n            cutlass.Float8E4M3FN,\n"),
    # the producer needs the raw V ring, which lives in the shared storage struct
    ("        storage = smem.allocate(SharedStorage)\n",
     "        storage = smem.allocate(SharedStorage)\n        self._k1_storage = storage\n"),
], header='_K2 = {"minb": 2, "regs1": (200, 56), "stages": 2}\n\n\ndef _k2_skip_check(*_a, **_k):\n    return None\n\n\n')

IFACE = _patched("interface", "k1fa4v2_iface", [
    ('        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}\n',
     '        input_tensors = {"q": q, "qv": qv}\n'),
    ("            arch // 10 == 10\n        ), \"FP8 is only supported on SM100",
     "            arch // 10 in (9, 10)\n        ), \"FP8 is only supported on SM100"),
    ("        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n    )\n",
     "        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n"
     "        cache_dir=None,\n    )\n"),
])

PagedKVManager = SM90.PagedKVManager  # the image's paged_kv class, as the sm90 module imported it


class K2Sm90(SM90.FlashAttentionForwardSm90):
    """FA4 sm90 forward, FP8 QK (E4M3 Q, gathered FP8 K) and F16 PV (upcast V), new producer (module docstring)."""

    def __init__(self, dtype, *args, **kwargs):
        # K/V pipeline depth per layer type (the compile key separates local from full, so both can coexist)
        SM90._K2["stages"] = STAGES["sliding" if kwargs.get("is_local") else "full"]
        # the interface passes Q's dtype (E4M3); P, V, O and the PV wgmma are F16
        super().__init__(cutlass.Float16, *args, **kwargs)

    def _get_shared_storage_cls(self):
        # sO (F16) aliases sQ (E4M3) in FA4's epilogue: size the Q buffer for both
        q_n = max(cute.cosize(self.sQ_layout), 2 * cute.cosize(self.sO_layout))
        sQ_struct = cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, q_n], 1024]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(self.sK_layout)], 1024]
        sV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        sVr_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Float8E4M3FN, RAW_STAGES * self.tile_n * self.tile_hdimv], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageK2:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sVr: sVr_struct

        return SharedStorageK2

    @cute.jit
    def _k2_upcast_own(self, pkm, tiled_r2s, tidx, sXr, sX):
        """One 128-key V tile: each thread upcasts (E4M3 -> F16, exact) the 16-byte chunks its own cp.async wrote
        (PagedKVManager mapping: 4 threads x 16 elements per row) into the swizzled F16 stage. No barrier: the
        thread's cp.async.wait_group covers visibility, program order covers raw-slot reuse."""
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
        sVr = storage.sVr.get_tensor(cute.make_layout(
            (self.tile_n, self.tile_hdimv, RAW_STAGES),
            stride=(self.tile_hdimv, 1, self.tile_n * self.tile_hdimv)))
        tiled_r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128),
            cute.make_ordered_layout((nthr // 4, 4), order=(1, 0)), cute.make_layout((1, 16)))

        q_producer_phase = Int32(1)
        k_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
        v_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
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
                for it in cutlass.range(nblk, unroll=1):
                    n_block = n_block_max - 1 - it
                    # K(it): FP8 straight into the wgmma stage, FA4's cp.async arrive signals the consumer
                    pipeline_k.producer_acquire(k_state)
                    pkm.load_page_table(n_block)
                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")
                    pipeline_k.producer_commit(k_state)
                    k_state.advance()
                    # V(it): FP8 into the raw ring; one cp.async group per iteration
                    pkm.load_KV(n_block, sVr[None, None, it % RAW_STAGES], "V")
                    cute.arch.cp_async_commit_group()
                    if it > 0:
                        cute.arch.cp_async_wait_group(1)  # group it - 1 (this thread's copies) landed
                        pipeline_v.producer_acquire(v_state)
                        self._k2_upcast_own(pkm, tiled_r2s, tidx, sVr[None, None, (it - 1) % RAW_STAGES],
                                            sV[None, None, v_state.index])
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                        v_state.advance()
                cute.arch.cp_async_wait_group(0)
                pipeline_v.producer_acquire(v_state)
                self._k2_upcast_own(pkm, tiled_r2s, tidx, sVr[None, None, (nblk - 1) % RAW_STAGES],
                                    sV[None, None, v_state.index])
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                v_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_v.producer_tail(v_state)


IFACE.FlashAttentionForwardSm90 = K2Sm90  # a global of the private interface copy only
TILE = (64, 128)
MAX_SPLITS = 128
STAGES = {"full": 3, "sliding": 2}  # measured: full 3 = -2.9 us at ver32 (k1_fa4_design.md section 8)


def set_stages(n, layer=None):
    """K/V pipeline depth for 'full', 'sliding' or both (None); 2 or 3 (3 still fits 2 CTAs/SM:
    8 + 24 + 48 + 16 KB). A change drops the compiled kernels."""
    new = dict(STAGES)
    for key in ((layer,) if layer else ("full", "sliding")):
        new[key] = n
    if new != STAGES:
        STAGES.update(new)
        IFACE._flash_attn_fwd.compile_cache = IFACE._get_jit_cache("fwd")


def num_splits_for(batch, hkv, keys, num_sms=132, ctas_per_sm=2):
    """FA4's heuristic scaled by the CTAs per SM of this build; keys = the longest key range a request reads
    (the context bound on full layers, window + L on sliding ones)."""
    nnb = (keys + TILE[1] - 1) // TILE[1]
    return max(2, min(MAX_SPLITS, IFACE.num_splits_heuristic(batch * hkv, num_sms, nnb, 128) * ctas_per_sm))


def fwd(q8, k8, v8, page_table, seqused_k, max_seqlen_k, softmax_scale, num_splits, window=0, iwo=False):
    """q8 [B, L, Hq, 64] float8_e4m3fn (unit-RMS queries, RoPE'd on sliding layers); k8/v8 [P, 1, Hkv, 64]
    float8_e4m3fn (page size 1, unit scales); page_table [B, max_pages] int32; seqused_k [B] int32 (cache lengths
    incl. the block); window 0 = full, else the sliding window (a token and the window - 1 before it).
    iwo: FA4's intra-warpgroup overlap (QK(i+1) issued before PV(i)); the producer order K(i+1), V(i) serves both.
    Returns bf16 [B, L, Hq, 64] (FA4 split combine)."""
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn and k8.shape[1] == 1 and num_splits >= 2
    assert q8.shape[-1] == 64 and q8.shape[-2] == 8 * k8.shape[-2]
    out, _ = IFACE._flash_attn_fwd(q8, k8, v8, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
                                   page_table=page_table, softmax_scale=softmax_scale, causal=True,
                                   window_size_left=(window - 1) if window else None,
                                   window_size_right=0 if window else None,
                                   tile_mn=TILE, num_splits=num_splits, pack_gqa=True, intra_wg_overlap=iwo)
    return out

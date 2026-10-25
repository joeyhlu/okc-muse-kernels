"""K1-FA4 v4: FP8 QK and FP8 PV (E4M3 P, as stock FA3 does for its PV wgmma: A16) attention for L=8 (verify)
blocks, full or sliding-window layers, on the judge image's FA4 CuTe-DSL sm90 forward. Successor of k1_fa4v2.py
(FP8 QK, F16 PV with V upcast); everything but the PV path is v2's.

Reference code (open source, read at run time from the image, never copied into the repo):
  sglang/kernels/ops/attention/flash_attn/cute/flash_fwd_sm90.py, .../paged_kv.py, .../interface.py, .../utils.py
  (prmt_b32); == repos/sglang/python/sglang/kernels/ops/attention/flash_attn/cute/.

FP8 PV on sm90: the E4M3 wgmma needs both operands K-major, so V is stored transposed, Vt[d][key], and P comes
from registers in the E4M3 A-fragment layout. Instead of shuffling P, the keys of each 32-key block are permuted:
  * register side: A value v = i + 4 rs + 8 kh (i: byte in the 4-key group, rs: row r / r+8, kh: k half) of k-block
    kb takes the S accumulator value c + 2 rs + 4 j with c = i % 2 and j = 4 kb + 2 kh + i // 2. That is a static
    register selection; two F32 values are packed per cvt.rn.satfinite.e4m3x2.f32.
  * smem side: virtual column 32 blk + 16 kh + 4 q + i of Vt holds key 32 blk + 16 kh + 2 q + (0, 1, 8, 9)[i].
The producer gathers V itself: thread (warp wv, lane l) takes the key quad g = 16 (wv//2) + l//2 (virtual columns
4g..4g+3, keys 32 (g//8) + 16 ((g%8)//4) + 2 (g%4) + (0, 1, 8, 9)) and head-dim chunk c = 2 (wv%2) + l%2. A lane pair
therefore fetches one full 32 B sector per key row. (One lane per 16 B chunk, the first build, read 1.5x the bytes:
ver32 396 us; lane pairs per 32 B sector: 273 us.) Final mapping: 4 lanes per 64 B key row, quads chosen so that a
warp's stores are bank-conflict free. Each thread issues 4 x 16 B cp.async into a thread-private raw slot. Once its own
cp.async group lands, it does 4 LDS.128, rotates the bytes by c, does a 4x4 byte transpose (12 PRMT per four dims)
and 16 STS.32 into the 128B-swizzled Vt stage. No barrier is needed. Out-of-range keys are clamped to the last
valid key: the data stays finite and P is 0 there.
Packaging (A19): text-patched copies under new module names (k1fa4v4_*) in a mkdtemp dir; no env writes; no existing
module or sys.modules entry touched; process-local CuTe JIT cache (cache_dir=None).
"""
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32, const_expr, pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from quack import copy_utils

VENDOR = "sglang.kernels.ops.attention.flash_attn.cute"
TMP = Path(tempfile.mkdtemp(prefix="k1fa4v4_"))
RAW = {"n": 2}  # raw V ring depth: 2 = transpose one tile behind the gather (3: two behind; measured
                # slower on sliding, equal on full: k1_fa4_design.md section 9)


def _patched(modname, outname, reps, header=""):
    assert outname not in sys.modules, outname
    src = importlib.import_module(VENDOR + "." + modname)
    txt = Path(src.__file__).read_text()
    for a, b, n in reps:
        assert txt.count(a) == n, ("patch anchor", modname, a, txt.count(a))
        txt = txt.replace(a, b)
    f = TMP / (outname + ".py")
    f.write_text(header + txt)
    spec = importlib.util.spec_from_file_location(outname, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[outname] = m  # a new private name (A19)
    spec.loader.exec_module(m)
    return m


SM90 = _patched("flash_fwd_sm90", "k1fa4v4_sm90", [
    ("min_blocks_per_mp=1,", 'min_blocks_per_mp=_K4["minb"],', 1),
    ("1: (256, 56),", '1: _K4["regs1"],', 1),
    ("        super().__init__(*args, **kwargs)\n",
     '        kwargs["num_stages"] = _K4["stages"]\n        super().__init__(*args, **kwargs)\n', 1),
    # V smem stages: E4M3, transposed (head dim x keys), K-major for the FP8 PV wgmma
    ("(mV.element_type, (self.tile_n, self.tile_hdimv), self.num_stages),",
     "(cutlass.Float8E4M3FN, (self.tile_hdimv, self.tile_n), self.num_stages),", 1),
    ("        sVt = layout_utils.transpose_view(sV)\n", "        sVt = sV\n", 1),
    ("        self._check_type(\n", "        _k4_skip_check(\n", 1),
    ("        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(\n            self.dtype,\n            self.dtype,\n",
     "        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(\n"
     "            cutlass.Float8E4M3FN,\n            cutlass.Float8E4M3FN,\n", 1),
    ("        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(\n            self.dtype,\n            self.dtype,\n"
     "            warpgroup.OperandMajorMode.K,\n            warpgroup.OperandMajorMode.MN,\n",
     "        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(\n"
     "            cutlass.Float8E4M3FN,\n            cutlass.Float8E4M3FN,\n"
     "            warpgroup.OperandMajorMode.K,\n            warpgroup.OperandMajorMode.K,\n", 1),
    # P -> E4M3 A fragment with the key permutation (see module docstring)
    ("utils.cvt_f16(tOrP_acc, tOrP_cur)", "_K4FN['cvt_p'](acc_S, tOrP_cur)", 2),
    ("        storage = smem.allocate(SharedStorage)\n",
     "        storage = smem.allocate(SharedStorage)\n        self._k1_storage = storage\n", 1),
], header=('_K4 = {"minb": 2, "regs1": (184, 72), "stages": 2}\n_K4FN = {}\n\n\n'
           'def _k4_skip_check(*_a, **_k):\n    return None\n\n\n'))

IFACE = _patched("interface", "k1fa4v4_iface", [
    ('        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}\n',
     '        input_tensors = {"q": q, "qv": qv}\n', 1),
    ("            arch // 10 == 10\n        ), \"FP8 is only supported on SM100",
     "            arch // 10 in (9, 10)\n        ), \"FP8 is only supported on SM100", 1),
    ("        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n    )\n",
     "        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n"
     "        cache_dir=None,\n    )\n", 1),
])

PagedKVManager = SM90.PagedKVManager
prmt = SM90.utils.prmt_b32  # the image's PTX prmt.b32 helper


@dsl_user_op
def _cvt_e4m3x4(a, b, c, d, *, loc=None, ip=None):
    """Four F32 -> one 32-bit word of E4M3 bytes (a in byte 0), RN, satfinite."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip),
         Float32(c).ir_value(loc=loc, ip=ip), Float32(d).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 lo, hi;\ncvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\ncvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n"
        "mov.b32 $0, {lo, hi};\n}",
        "=r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def _cvt_p(acc_S, tOrP):
    """S accumulator (F32, C layout of m64nN) -> E4M3 P in the FP8 A-fragment registers, keys permuted per block."""
    nkb = cute.size(acc_S) // 16
    words = cute.make_tensor(cute.recast_ptr(tOrP.iterator, dtype=Uint32), cute.make_layout((nkb * 4,)))
    for kb in cutlass.range_constexpr(nkb):
        for reg in cutlass.range_constexpr(4):
            rs, kh = reg % 2, reg // 2
            base = 2 * rs + 4 * (4 * kb + 2 * kh)
            # bytes i = 0..3 <- C values (c, j) = (0, 0), (1, 0), (0, 1), (1, 1) of the block's col groups
            words[4 * kb + reg] = _cvt_e4m3x4(acc_S[base], acc_S[base + 1], acc_S[base + 4], acc_S[base + 5])


SM90._K4FN["cvt_p"] = _cvt_p  # a global of the private module copy only


class K4Sm90(SM90.FlashAttentionForwardSm90):
    """FA4 sm90 forward: FP8 QK (E4M3 Q, gathered FP8 K), FP8 PV (E4M3 P, producer-transposed V)."""

    def __init__(self, dtype, *args, **kwargs):
        SM90._K4["stages"] = STAGES["sliding" if kwargs.get("is_local") else "full"]
        super().__init__(cutlass.Float16, *args, **kwargs)  # O / epilogue dtype; QK and PV are E4M3 (patched)

    def _get_shared_storage_cls(self):
        q_n = max(cute.cosize(self.sQ_layout), 2 * cute.cosize(self.sO_layout))  # sO (F16) aliases sQ
        f8 = cutlass.Float8E4M3FN
        sQ_struct = cute.struct.Align[cute.struct.MemRange[f8, q_n], 1024]
        sK_struct = cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(self.sK_layout)], 1024]
        sV_struct = cute.struct.Align[cute.struct.MemRange[f8, cute.cosize(self.sV_layout)], 1024]
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, 0], 1024]
        sVr_struct = cute.struct.Align[
            cute.struct.MemRange[f8, RAW["n"] * self.tile_n * self.tile_hdimv], 1024]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageK4:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sVr: sVr_struct

        return SharedStorageK4

    @cute.jit
    def _k4_v_pages(self, mPageTable, batch_idx, seqlen_k, n_block, rbase, pg):
        """Page (= cache row, page size 1) of keys n_block*128 + rbase + (0, 1, 8, 9), clamped to the last valid key.
        Issued one tile ahead, so the cp.async of the next tile does not wait on these loads."""
        for i in cutlass.range_constexpr(4):
            key = n_block * self.tile_n + rbase + (0, 1, 8, 9)[i]
            pg[i] = mPageTable[batch_idx, cutlass.min(key, seqlen_k - 1)]

    @cute.jit
    def _k4_gather_v(self, mV_h, pg, c, tidx, sVr, slot, atom):
        """4 x 16 B cp.async (pages pg, head-dim chunk 16c) into this thread's private raw slot."""
        for i in cutlass.range_constexpr(4):
            p = SM90.utils.elem_pointer(mV_h, (0, 16 * c, pg[i]))
            src = cute.make_tensor(cute.make_ptr(cutlass.Float8E4M3FN, p.toint(), cute.AddressSpace.gmem,
                                                 assumed_align=16), cute.make_layout((16,)))
            cute.copy(atom, src, sVr[None, tidx, i, slot])

    @cute.jit
    def _k4_transpose_v(self, sVr32, tidx, slot, vt_base, c, g, rot):
        """This thread's 4 keys x 16 dims (own bytes) -> 16 words Vt[16c + 4w + (e + c) % 4][4g .. 4g + 3] (128B
        swizzle); the input bytes are rotated by c so that step (w, e) of the 4 lanes of a key stores 4 rows with
        distinct (row % 8)."""
        rr = []
        for i in cutlass.range_constexpr(4):
            r = cute.make_rmem_tensor((4,), Uint32)
            cute.autovec_copy(sVr32[None, tidx, i, slot], r)
            rr.append(r)
        for w in cutlass.range_constexpr(4):
            x0 = prmt(rr[0][w], rr[0][w], rot)
            x1 = prmt(rr[1][w], rr[1][w], rot)
            x2 = prmt(rr[2][w], rr[2][w], rot)
            x3 = prmt(rr[3][w], rr[3][w], rot)
            t0 = prmt(x0, x1, Uint32(0x5140))
            t1 = prmt(x0, x1, Uint32(0x7362))
            t2 = prmt(x2, x3, Uint32(0x5140))
            t3 = prmt(x2, x3, Uint32(0x7362))
            outs = (prmt(t0, t2, Uint32(0x5410)), prmt(t0, t2, Uint32(0x7632)),
                    prmt(t1, t3, Uint32(0x5410)), prmt(t1, t3, Uint32(0x7632)))
            for e in cutlass.range_constexpr(4):
                d = 16 * c + 4 * w + ((e + c) % 4)
                off = (d * self.tile_n + 4 * g) ^ ((d % 8) * 16)  # 128B swizzle: bits 4-6 ^= row % 8
                cute.make_tensor(cute.recast_ptr(vt_base + off, dtype=Uint32), cute.make_layout((1,)))[0] = outs[e]

    @cute.jit
    def load(self, mQ, mK, mV, sQ, sK, sV, tma_atom_Q, tma_atom_K, tma_atom_V, mBias, sBias, tma_atom_Bias,
             pipeline_k, pipeline_v, pipeline_bias, pipeline_q, gmem_tiled_copy_Q, mPageTable,
             blocksparse_tensors, block_info, SeqlenInfoCls, TileSchedulerCls, num_splits: Int32 = 1):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        nthr = self.num_threads_per_warp_group
        storage = self._k1_storage
        # Vt stage layout (printed on the box): S<3,4,3> o ((8,8),(128,1),(1,S)):((128,1024),(1,0),(0,8192))
        f8 = cutlass.Float8E4M3FN
        nraw = RAW["n"]  # raw V ring depth; V(it) is transposed lag = nraw - 1 iterations after its gather
        lag = nraw - 1
        sVr = storage.sVr.get_tensor(cute.make_layout((16, nthr, 4, nraw),
                                                      stride=(1, 16, 16 * nthr, 64 * nthr)))
        sVr32 = cute.make_tensor(cute.recast_ptr(sVr.iterator, dtype=Uint32),
                                 cute.make_layout((4, nthr, 4, nraw), stride=(1, 4, 4 * nthr, 16 * nthr)))
        vt_ptr = storage.sV.data_ptr()
        vt_stage_bytes = self.tile_hdimv * self.tile_n
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), f8,
                                   num_bits_per_copy=128)
        # 4 lanes (c = 0..3) fetch the whole 64 B row of a key (as FA4's own gather does). The warp's 8 quads differ in
        # bits 0, 1, 4 of g, and the dim each lane stores per step is rotated by c (byte rotation before the
        # transpose), so the swizzled Vt stores of a warp hit 32 distinct banks.
        lane = tidx % 32
        wv = tidx // 32
        c = lane % 4
        gq = lane // 4
        g = (gq % 2) + 2 * ((gq // 2) % 2) + 4 * wv + 16 * (gq // 4)
        rot = (Uint32(0x32103210) >> (4 * c)) & Uint32(0xFFFF)
        rbase = 32 * (g // 8) + 16 * ((g % 8) // 4) + 2 * (g % 4)

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
            mV_h = mV[None, None, head_idx, None]
            if warp_idx_in_wg == 0:
                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_producer_phase ^= 1
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block, split_idx, num_splits)
            nblk = n_block_max - n_block_min
            if nblk > 0:
                pg = cute.make_rmem_tensor((4,), Int32)
                pkm.load_page_table(n_block_max - 1)
                self._k4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block_max - 1, rbase, pg)
                for it in cutlass.range(nblk, unroll=1):
                    n_block = n_block_max - 1 - it
                    pipeline_k.producer_acquire(k_state)
                    pkm.load_KV(n_block, sK[None, None, k_state.index], "K")
                    pipeline_k.producer_commit(k_state)
                    k_state.advance()
                    self._k4_gather_v(mV_h, pg, c, tidx, sVr, it % nraw, atom)
                    cute.arch.cp_async_commit_group()
                    if it + 1 < nblk:  # page loads for the next tile (K: FA4's per-thread rows; V: this thread's quad)
                        pkm.load_page_table(n_block - 1)
                        self._k4_v_pages(mPageTable, batch_idx, seqlen.seqlen_k, n_block - 1, rbase, pg)
                    if it >= lag:  # V(it - lag): this thread's copies landed (lag newer groups may be pending)
                        cute.arch.cp_async_wait_group(lag)
                        pipeline_v.producer_acquire(v_state)
                        self._k4_transpose_v(sVr32, tidx, (it - lag) % nraw,
                                             vt_ptr + v_state.index * vt_stage_bytes, c, g, rot)
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                        v_state.advance()
                cute.arch.cp_async_wait_group(0)
                for j in cutlass.range(cutlass.max(nblk - lag, 0), nblk, unroll=1):
                    pipeline_v.producer_acquire(v_state)
                    self._k4_transpose_v(sVr32, tidx, j % nraw, vt_ptr + v_state.index * vt_stage_bytes, c, g, rot)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(pipeline_v.sync_object_full.get_barrier(v_state.index))
                    v_state.advance()
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        pipeline_v.producer_tail(v_state)


IFACE.FlashAttentionForwardSm90 = K4Sm90  # a global of the private interface copy only
TILE = (64, 128)
MAX_SPLITS = 128
STAGES = {"full": 3, "sliding": 2}


def set_stages(n, layer=None):
    new = dict(STAGES)
    for key in ((layer,) if layer else ("full", "sliding")):
        new[key] = n
    if new != STAGES:
        STAGES.update(new)
        IFACE._flash_attn_fwd.compile_cache = IFACE._get_jit_cache("fwd")


def set_raw(n):
    """Raw V ring depth (2 or 3); drops compiled kernels on a change."""
    if RAW["n"] != n:
        RAW["n"] = n
        IFACE._flash_attn_fwd.compile_cache = IFACE._get_jit_cache("fwd")


def num_splits_for(batch, hkv, keys, num_sms=132, ctas_per_sm=2):
    nnb = (keys + TILE[1] - 1) // TILE[1]
    return max(2, min(MAX_SPLITS, IFACE.num_splits_heuristic(batch * hkv, num_sms, nnb, 128) * ctas_per_sm))


def fwd(q8, k8, v8, page_table, seqused_k, max_seqlen_k, softmax_scale, num_splits, window=0):
    """Same contract as k1_fa4v2.fwd: q8 [B, L, Hq, 64] E4M3; k8/v8 [P, 1, Hkv, 64] E4M3 (page size 1, unit
    scales); returns bf16 [B, L, Hq, 64]."""
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn and k8.shape[1] == 1 and num_splits >= 2
    assert q8.shape[-1] == 64 and q8.shape[-2] == 8 * k8.shape[-2]
    out, _ = IFACE._flash_attn_fwd(q8, k8, v8, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
                                   page_table=page_table, softmax_scale=softmax_scale, causal=True,
                                   window_size_left=(window - 1) if window else None,
                                   window_size_right=0 if window else None,
                                   tile_mn=TILE, num_splits=num_splits, pack_gqa=True, intra_wg_overlap=False)
    return out

"""K1-FA4 v5: the FA4 v2 attention core (FP8 QK, F16 PV; same kernel code as k1_fa4v2.py) between a fused prologue and
a fused epilogue, chained with programmatic dependent launch (PDL), for decode (L=1) and verify (L=8) steps on full
or sliding layers. Three launches per call instead of the v2 route's five (r1's FusedQKNorm + store_cache, FA4 main
+ FA4 combine, rms_norm_gate):
  1. prologue (Triton): per (token, head) row, SGLang's FusedQKNorm arithmetic verbatim (fp32 RMS, bf16 rounding,
     RoPE on sliding layers, +-448 clamp): writes the E4M3 query and stores the new FP8 K/V straight into the cache
     rows (cache_loc), with no staging copy. Unlike SGLang's kernel it also writes cache row 0, which the judge's
     inputs can use (the reason r1 stages). gdc_wait / gdc_launch_dependents (PDL).
  2. FA4 main (CuTe DSL, v2 kernel): launched with use_pdl; its producer warps execute griddepcontrol.wait before
     the first Q/K/V read and then trigger the dependents. The interface returns the split partials (its own
     combine is skipped).
  3. combine + RMS-norm + sigmoid gate (Triton, PDL secondary): merges the splits by LSE, rounds the attention output
     to bf16 as FA4's combine does, applies rms_norm_gate's math and writes the bf16 output.
Reference code (open source, read at run time from the image): sglang/kernels/ops/attention/flash_attn/cute/*
(FA4), sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py (_qk_rms_norm_fwd_kernel).
Packaging (A19): text-patched copies under new module names (k1fa4v5_*) in a mkdtemp dir; no env writes; no existing
module or sys.modules entry touched; process-local CuTe JIT cache.
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
TMP = Path(tempfile.mkdtemp(prefix="k1fa4v5_"))
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


SM90 = _patched("flash_fwd_sm90", "k1fa4v5_sm90", [
    ("min_blocks_per_mp=1,", 'min_blocks_per_mp=_K2["minb"], use_pdl=_K2["pdl"],'),
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
    # optional PDL trigger once the consumer's main loop is done (the combine kernel can launch while O is written)
    ("            self.epilogue(\n                acc_O,\n",
     "            if const_expr(_K2['trig'] == 'epi'):\n                cute.arch.griddepcontrol_launch_dependents()\n"
     "            self.epilogue(\n                acc_O,\n"),
], header='_K2 = {"minb": 2, "regs1": (200, 56), "stages": 2, "pdl": True, "trig": False}\n\n\ndef _k2_skip_check(*_a, **_k):\n    return None\n\n\n')

IFACE = _patched("interface", "k1fa4v5_iface", [
    ('        input_tensors = {"q": q, "k": k, "v": v, "qv": qv}\n',
     '        input_tensors = {"q": q, "qv": qv}\n'),
    ("            arch // 10 == 10\n        ), \"FP8 is only supported on SM100",
     "            arch // 10 in (9, 10)\n        ), \"FP8 is only supported on SM100"),
    ("        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n    )\n",
     "        source_paths=(os.path.dirname(os.path.abspath(__file__)),),\n        enable_tvm_ffi=True,\n"
     "        cache_dir=None,\n    )\n"),
    # return the split partials: the fused combine + norm-gate kernel replaces FA4's combine
    ("    if is_split_kv:\n        _flash_attn_fwd_combine(\n",
     "    if is_split_kv and _K5I['skip_combine']:\n        return out_partial, lse_partial\n"
     "    if is_split_kv:\n        _flash_attn_fwd_combine(\n"),
], header="_K5I = {'skip_combine': True}\n")

PagedKVManager = SM90.PagedKVManager  # the image's paged_kv class, as the sm90 module imported it


class K5Sm90(SM90.FlashAttentionForwardSm90):
    """FA4 sm90 forward, FP8 QK (E4M3 Q, gathered FP8 K) and F16 PV (upcast V), new producer (module docstring)."""

    def __init__(self, dtype, *args, **kwargs):
        # K/V pipeline depth per layer type (the compile key separates local from full, so both can coexist)
        SM90._K2["stages"] = STAGES["sliding" if kwargs.get("is_local") else "full"]
        SM90._K2["trig"] = TRIG["sliding" if kwargs.get("is_local") else "full"]
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
        if const_expr(SM90._K2["pdl"]):
            cute.arch.griddepcontrol_wait()  # the prologue's E4M3 q and new K/V rows are visible from here on
            if const_expr(SM90._K2["trig"] == "start"):
                cute.arch.griddepcontrol_launch_dependents()  # let the combine kernel's CTAs launch early
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


IFACE.FlashAttentionForwardSm90 = K5Sm90  # a global of the private interface copy only
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


TRIG = {"full": "epi", "sliding": "epi"}  # PDL trigger in FA4 main: False, "start" (producer), "epi"
# (after the consumer main loop; measured best: start hurt sliding decode by 1-1.5 us, epi -0.1..-0.9 us)
SPLITS = {("sliding", 1): 32}  # (layer, batch) -> num_splits (B=1 sliding: 32 beat the heuristic at L=1 and L=8)


def set_pdl(pdl, trig=None):
    """FA4 main kernel: PDL launch + griddepcontrol.wait in the producer (pdl); trig (None = keep): dependent
    trigger for both layer types. A change drops the compiled kernels."""
    new = dict(TRIG) if trig is None else {"full": trig, "sliding": trig}
    if SM90._K2["pdl"] != pdl or new != TRIG:
        SM90._K2["pdl"] = pdl
        TRIG.update(new)
        IFACE._flash_attn_fwd.compile_cache = IFACE._get_jit_cache("fwd")


def fwd_partials(q8, k8, v8, page_table, seqused_k, max_seqlen_k, softmax_scale, num_splits, window=0):
    """The v2 core; returns (out_partial f32 [S, B, L, Hq, 64], lse_partial f32 [S, B, Hq, L]) (natural log)."""
    assert q8.dtype == k8.dtype == v8.dtype == torch.float8_e4m3fn and k8.shape[1] == 1 and num_splits >= 2
    assert q8.shape[-1] == 64 and q8.shape[-2] == 8 * k8.shape[-2]
    return IFACE._flash_attn_fwd(q8, k8, v8, seqused_k=seqused_k, max_seqlen_k=max_seqlen_k,
                                 page_table=page_table, softmax_scale=softmax_scale, causal=True,
                                 window_size_left=(window - 1) if window else None,
                                 window_size_right=0 if window else None,
                                 tile_mn=TILE, num_splits=num_splits, pack_gqa=True, intra_wg_overlap=False)


# ----------------------------------------------------------------------------------------------------------------
# Fused prologue and epilogue (Triton)
# ----------------------------------------------------------------------------------------------------------------
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

RMS_EPS = 1e-5
ROPE_POSITIONS = 1 << 21
_ROPE = {}


@triton.jit
def _k5_prologue(QKV, Q_OUT, POS, COS_SIN, K_CACHE, V_CACHE, CACHE_LOC, eps, stride_qkv, n_rows,
                 NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                 BLOCK_M: tl.constexpr, HAS_ROPE: tl.constexpr, USE_PDL: tl.constexpr,
                 FP8_MAX_C: tl.constexpr = 448.0):
    """SGLang FusedQKNorm (_qk_rms_norm_fwd_kernel with HAS_KV_STORE and FP8_KV_STORE) arithmetic, verbatim. It
    stores every cache row (no reserved slot 0) and skips the unused bf16 K output."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < n_rows
    cols = tl.arange(0, BD)
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]
    HEADS: tl.constexpr = NUM_Q_HEADS + 2 * NUM_KV_HEADS
    token = rows // HEADS
    head = rows % HEADS
    is_q = head < NUM_Q_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    src = token * stride_qkv + head * D
    raw = tl.load(QKV + src[:, None] + cols[None, :], mask=mask, other=0.0)
    x = raw.to(tl.float32)
    var = tl.sum(tl.where(mask, x * x, 0.0), axis=1) / D
    rstd = tl.rsqrt(var + eps)
    x = x * rstd[:, None]
    if HAS_ROPE:
        x = x.to(QKV.dtype.element_ty).to(tl.float32)
        partner_cols = cols ^ 1
        xp = tl.load(QKV + src[:, None] + partner_cols[None, :], mask=mask, other=0.0).to(tl.float32)
        xp = (xp * rstd[:, None]).to(QKV.dtype.element_ty).to(tl.float32)
        pos = tl.load(POS + token, mask=row_mask, other=0).to(tl.int64)
        freq = cols // 2
        cs_mask = row_mask[:, None] & (cols < D)[None, :]
        cos = tl.load(COS_SIN + pos[:, None] * D + freq[None, :], mask=cs_mask, other=1.0)
        sin = tl.load(COS_SIN + pos[:, None] * D + D // 2 + freq[None, :], mask=cs_mask, other=0.0)
        is_even = (cols % 2 == 0)[None, :]
        x = tl.where(is_even, x * cos - xp * sin, xp * sin + x * cos)
    x = x.to(QKV.dtype.element_ty).to(tl.float32)
    x = tl.where(is_v[:, None], raw.to(tl.float32), x)
    x = tl.minimum(tl.maximum(x, -FP8_MAX_C), FP8_MAX_C)
    loc = tl.load(CACHE_LOC + token, mask=row_mask, other=0).to(tl.int64)
    kv_head = tl.where(is_v, head - NUM_Q_HEADS - NUM_KV_HEADS, head - NUM_Q_HEADS)
    kv_dst = loc * (NUM_KV_HEADS * D) + kv_head * D
    tl.store(K_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_k[:, None])
    tl.store(V_CACHE + kv_dst[:, None] + cols[None, :], x, mask=mask & is_v[:, None])
    q_dst = tl.where(is_q, (token * NUM_Q_HEADS + head) * D, 0)
    tl.store(Q_OUT + q_dst[:, None] + cols[None, :], x, mask=mask & is_q[:, None])


@triton.jit
def _k5_combine_gate(OP, LSEP, QKV, OUT, stride_qkv, gate_off, n_rows, NS, B, L, eps,
                     H: tl.constexpr, D: tl.constexpr, NS_P2: tl.constexpr, USE_PDL: tl.constexpr):
    """One program per (token, head) row r = (b * L + l) * H + h:
    out = rms_norm(bf16(sum_s w_s O_s / sum_s w_s)) * sigmoid(gate), w_s = exp(lse_s - max lse).
    All splits are loaded in one [NS, D] tile (no serial split loop). OP f32 [S, B, L, H, D]; LSEP f32 [S, B, H, L]."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    r = tl.program_id(0)
    t = r // H
    h = r % H
    b = t // L
    l = t % L
    lrow = (b * H + h) * L + l
    s = tl.arange(0, NS_P2)
    smask = s < NS
    lse = tl.load(LSEP + s * (B * H * L) + lrow, mask=smask, other=float("-inf"))
    m = tl.max(lse, axis=0)
    m = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse - m)
    den = tl.sum(w, axis=0)
    d = tl.arange(0, D)
    o = tl.load(OP + s[:, None] * (n_rows * D) + r * D + d[None, :], mask=smask[:, None], other=0.0)
    acc = tl.sum(w[:, None] * o, axis=0) / tl.where(den > 0, den, 1.0)
    acc = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(acc * acc, axis=0) / D
    y = acc * tl.rsqrt(var + eps)
    g = tl.load(QKV + t * stride_qkv + gate_off + h * D + d).to(tl.float32)
    y = y * (1.0 / (1.0 + tl.exp(-g)))
    tl.store(OUT + t * (H * D) + h * D + d, y.to(tl.bfloat16))


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, as SGLang's RotaryEmbedding (and r1) builds it, cached per theta."""
    key = (theta, str(device))
    cache = _ROPE.get(key)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _ROPE[key] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def route_ok(cfg):
    return (cfg["q_len"] in (1, 8) and cfg["page_size"] == 1 and cfg["kv_scale"] == 1.0 and cfg["head_dim"] == 64
            and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"])


def run(data, splits=None, pdl=True):
    """Full kernel_fn path (the attention problem's inputs): prologue -> FA4 v2 core -> combine + norm-gate."""
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    window = cfg["window"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    q8 = torch.empty((tokens, hq * d), dtype=torch.float8_e4m3fn, device=device)
    n_rows = tokens * (hq + 2 * hkv)
    block_m = 4 if n_rows < 1024 else (8 if n_rows < 8192 else 16)
    theta = cfg["rope_theta"]
    cos_sin = _cos_sin(theta, d, device) if theta else qkv
    pdl_kw = {"launch_pdl": True} if pdl else {}
    _k5_prologue[(triton.cdiv(n_rows, block_m),)](
        qkv, q8, data["positions"] if theta else qkv, cos_sin, k_cache.view(-1, hkv * d), v_cache.view(-1, hkv * d),
        data["cache_loc"], RMS_EPS, qkv.stride(0), n_rows, hq, hkv, d, triton.next_power_of_2(d), block_m,
        HAS_ROPE=bool(theta), USE_PDL=pdl, num_warps=4, **pdl_kw)
    max_k = data["page_table"].shape[1]
    keys = min(max_k, window + length) if window else max_k
    ns = splits or SPLITS.get(("sliding" if window else "full", batch)) or num_splits_for(batch, hkv, keys)
    set_pdl(pdl)
    o_part, lse_part = fwd_partials(q8.view(batch, length, hq, d), k_cache, v_cache, data["page_table"],
                                    data["cache_seqlens"], max_k, cfg["sm_scale"] * cfg["qk_scale_factor"], ns,
                                    window=window)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    rows = tokens * hq
    _k5_combine_gate[(rows,)](
        o_part, lse_part, qkv, out, qkv.stride(0), (hq + 2 * hkv) * d, rows, ns, batch, length, RMS_EPS,
        H=hq, D=d, NS_P2=triton.next_power_of_2(ns), USE_PDL=pdl, num_warps=2 if ns <= 8 else 4, **pdl_kw)
    return out
